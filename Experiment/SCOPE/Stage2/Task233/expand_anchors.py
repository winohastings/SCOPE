from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task233.gate0 import execute, MODEL_PYTHON, WebsocketClientPolicy
from Experiment.SCOPE.Stage2.Task233.analyze import summarize_anchor, write_csv

SOURCE = Path("/home/data/czh/Third_Party/tasks/R2R-CE/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")


def route(record):
    payload = record.get("payload", record)
    return Path(record["scene_id"]).stem, str(payload["trajectory_id"])


def stats(records):
    return {"episodes": len(records), "scenes": len({Path(record["scene_id"]).stem for record in records}),
            "trajectories": len({route(record) for record in records})}


def select_full_train(full, excluded, seed=233256):
    generator = random.Random(seed)
    shuffled = list(full)
    generator.shuffle(shuffled)
    groups = defaultdict(list)
    seen = set(excluded)
    for record in shuffled:
        identity = route(record)
        if identity not in seen:
            groups[identity[0]].append(record)
            seen.add(identity)
    scenes = sorted(groups)
    generator.shuffle(scenes)
    selected = []
    while any(groups.values()):
        for scene in scenes:
            if groups[scene]:
                selected.append(groups[scene].pop())
    return selected


def convert(record):
    payload = copy.deepcopy(record)
    identifier = str(payload["episode_id"])
    instruction = payload["instruction"]["instruction_text"].strip()
    points = payload.get("reference_path", [])
    length = float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum()) if len(points) > 1 else 0.
    payload.update(episode_id=identifier, instruction=instruction, task_name="r2r", split="train",
                   gt_path_length=length, reference_points=[[-point[2], point[0], -point[1]] for point in points])
    return {"episode_uid": f"task233_full_train:{identifier}", "source_episode_id": identifier,
            "scene_id": Path(payload["scene_id"]).stem, "instruction": instruction, "source": "vlnce_r2r",
            "input_namespace": "task233_full_train", "input_root": "/home/data/czh/Third_Party/tasks", "payload": payload}


def validate_parent(parent):
    if json.loads((parent / "PROGRESS_GATE.json").read_text())["Gate0"] != "PASS":
        raise ValueError("Gate0 PASS required; anchor expansion cannot conceal failed Gate0")
    audit = json.loads((parent / "teacher_signal_audit.json").read_text())
    if sha256(parent / "candidate_results.jsonl") != audit["result_sha256"]:
        raise ValueError("parent evidence changed")
    rows = [json.loads(line) for line in (parent / "candidate_results.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["anchor_id"]].append(row)
    if len(groups) != 72 or len({row["anchor_id"] for row in rows if row["partition"] != "calibration"}) != 64:
        raise ValueError("expected audited 8 calibration + 64 evaluation anchors")
    for row in rows:
        if row["variant"] == "stop":
            if not Path(row["diagnostic"]["state_path"]).is_file():
                raise FileNotFoundError("parent ActorState missing")
    return rows


def combine(output, parent, parent_rows, requested):
    fresh = [json.loads(line) for line in (output / "candidate_results.jsonl").read_text().splitlines()]
    old = [row for row in parent_rows if row["partition"] != "calibration"]
    groups = defaultdict(list)
    for row in old + fresh:
        groups[row["anchor_id"]].append(row)
    if len(groups) != requested:
        raise ValueError("unique count does not equal user target")
    thresholds = json.loads((parent / "FROZEN_THRESHOLDS.json").read_text())
    quality = [summarize_anchor(records, thresholds) for records in groups.values()]
    anchors = []
    physical_keys = set()
    for identifier, records in groups.items():
        variants = [row["variant"] for row in records]
        if len(variants) != len(set(variants)) or any(f"candidate_{index}" not in variants for index in range(8)):
            raise ValueError("missing/duplicated candidate evidence")
        baseline = next(row for row in records if row["variant"] == "stop")
        identity = baseline["restoration_identity"]
        physical = (baseline["scene"], str(baseline["trajectory_id"]), tuple(identity["pose"]), identity["images"])
        if physical in physical_keys:
            raise ValueError("duplicate physical navigation anchor")
        physical_keys.add(physical)
        if any(not row["restoration_valid"] or row["split"] != "train" or row["ppo_eligible"]
               or row["diagnostic"]["current_tokens"] != 64 for row in records):
            raise ValueError("restoration/split/teacher/token invariant failed")
        state = Path(baseline["diagnostic"]["state_path"])
        anchors.append({"anchor_id": identifier, "episode_uid": baseline["episode_uid"], "trajectory_id": baseline["trajectory_id"],
                        "scene": baseline["scene"], "partition": baseline["partition"], "state_source": baseline["state_source"],
                        "j": baseline["j"], "state_path": str(state), "state_sha256": sha256(state),
                        "evidence": str(parent / "candidate_results.jsonl") if baseline in old else str(output / "candidate_results.jsonl")})
    train_routes = {(row["scene"], row["trajectory_id"]) for row in anchors if row["partition"] == "teacher_train"}
    heldout_routes = {(row["scene"], row["trajectory_id"]) for row in anchors if row["partition"] == "teacher_heldout"}
    if train_routes & heldout_routes:
        raise ValueError("train/heldout trajectory leakage")
    write_json(output / "ANCHOR_MANIFEST_256.json", {"anchors": anchors, "unique_noncalibration_anchors": len(anchors),
               "calibration_anchors_separate": 8, "parent": str(parent), "training_updates": 0})
    write_csv(output / "candidate_quality_256.csv", quality)
    result = {"status": "COMPLETE", "target": requested, "unique_noncalibration_anchors": len(anchors),
              "new_anchors": len({row["anchor_id"] for row in fresh}), "retained_parent_noncalibration_anchors": 64,
              "scenes": len({row["scene"] for row in anchors}), "trajectories": len({(row["scene"], row["trajectory_id"]) for row in anchors}),
              "partitions": {label: sum(row["partition"] == label for row in anchors) for label in ("teacher_train", "teacher_heldout")},
              "state_sources": {label: sum(row["state_source"] == label for row in anchors) for label in ("expert_start", "learner_induced")},
              "credible_labels": sum(row["confident"] for row in quality), "credible_label_count_not_target": True,
              "all_61_scenes_represented": len({row["scene"] for row in anchors}) == 61,
              "parent_unchanged": True, "no_training_or_group": True, "new_branch_count": len(fresh)}
    if not result["all_61_scenes_represented"]:
        raise ValueError("full-train scene coverage incomplete")
    write_json(output / "EXPANSION_AUDIT.json", result)
    write_json(output / "PROGRESS_GATE.json", {"Gate0": "PASS_PARENT", "anchor_expansion": "COMPLETE",
               "Gate1": "NOT_RUN", "Gate2": "NOT_RUN", "Gate3": "NOT_RUN", "optimizer_updates": 0,
               "note": "256 anchors collected, not 256 high-confidence labels; no claim Gate1 passed"})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--port", type=int, default=10233)
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--habitat-gpu", type=int, default=1)
    parser.add_argument("--wall-hours", type=float, default=3.)
    args = parser.parse_args()
    args.output, args.parent = args.output.resolve(), args.parent.resolve()
    if ROOT not in args.output.parents or args.anchors != 256 or not 0 < args.wall_hours <= 3:
        parser.error("Task233 new output, exactly 256 noncalibration anchors, at most three hours")
    parent_rows = validate_parent(args.parent)
    old_manifest = json.loads((args.parent / "DATA_MANIFEST.json").read_text())["train"]
    with gzip.open(SOURCE, "rt") as stream:
        full = json.load(stream)["episodes"]
    records = select_full_train(full, {route(record) for record in old_manifest})
    selection = [convert(record) for record in records]
    args.target_anchors = args.anchors - 64
    args.anchor_prefix = "fulltrain256_"
    args.partition_by_episode = {record["episode_uid"]: "teacher_heldout" if index % 4 == 3 else "teacher_train"
                                 for index, record in enumerate(selection)}
    args.output.mkdir(parents=True, exist_ok=False)
    old_pool = json.loads((ROOT.parent / "Task23/artifacts/ppo4096/DATA_MANIFEST.json").read_text())["train"]
    scope = {"full_train": stats(full), "previous_pool": stats(old_pool), "previous_selection": stats(old_manifest),
             "new_reserve": stats(selection), "excluded_old_trajectories": len({route(record) for record in old_manifest}),
             "source": str(SOURCE), "source_sha256": sha256(SOURCE), "seed": 233256,
             "newly_authorized_train_episodes": len(full) - len(old_pool), "validation_splits_accessed": False,
             "historical_train_dev_note": "7 former train-development scenes now authorized for teacher data; no longer blind train-heldout evidence"}
    write_json(args.output / "TRAIN_SCOPE_AUDIT.json", scope)
    write_json(args.output / "DATA_MANIFEST.json", {"train": selection, "val_seen": [], "scope": scope,
               "partition_by_episode": args.partition_by_episode, "selection": "full reserve frozen before evaluation; scene round-robin; unique routes"})
    write_json(args.output / "RUN_CONFIG.json", {"target_noncalibration_total": 256, "retained": 64, "new_target": 192,
               "calibration_separate": 8, "wall_hours": args.wall_hours, "partition_rule": "every fourth fresh episode heldout",
               "replacement_rule": "next frozen reserve only for unavailable second live state or duplicate physical state; never outcome gain",
               "criterion": "any complete valid anchor counts; retain low/negative/margin-tied samples", "optimizer_updates": 0,
               "protocol_sha256": sha256(ROOT / "EXPANSION_PROTOCOL.md")})
    checkpoint = ROOT.parent / "Task23/artifacts/ppo4096/runtime/checkpoints/update_0014.pt"
    trusted = json.loads((args.parent / "TRUSTED_ASSET_AUDIT.json").read_text())
    expected = next(record["actual"] for record in trusted["assets"] if record["path"].endswith("update_0014.pt"))
    if sha256(checkpoint) != expected:
        raise ValueError("trusted checkpoint changed")
    inputs = [checkpoint, SOURCE, ROOT.parent / "Task21/artifacts/release/adapter.pt", ROOT / "EXPANSION_PROTOCOL.md"] + list(ROOT.glob("*.py"))
    parent_files = [path for path in args.parent.rglob("*") if path.is_file()]
    hashes = {str(path): sha256(path) for path in inputs + parent_files}
    write_json(args.output / "SOURCE_INTEGRITY_BEFORE.json", hashes)
    with (args.output / "server.log").open("w") as log:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.model_gpu, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        server = subprocess.Popen([MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(args.output / "runtime"),
                                   "--manifest", str(args.output / "DATA_MANIFEST.json"), "--checkpoint", str(checkpoint),
                                   "--port", str(args.port)], stdout=log, stderr=subprocess.STDOUT, env=environment)
        client = None
        try:
            deadline = time.monotonic() + 600
            while not (args.output / "runtime/ready.json").exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("model startup failed")
                time.sleep(1)
            client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
            client._ws.ping_timeout = 1200
            execute(args, client, selection)
            after = {path: sha256(path) for path in hashes}
            write_json(args.output / "SOURCE_INTEGRITY_AFTER.json", {"hashes": after, "unchanged": after == hashes})
            if after != hashes:
                raise ValueError("source/parent evidence changed during collection")
            combine(args.output, args.parent, parent_rows, args.anchors)
        except BaseException as error:
            write_json(args.output / "INCOMPLETE.json", {"error": repr(error), "optimizer_updates": 0})
            raise
        finally:
            if client is not None:
                client.close()
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
