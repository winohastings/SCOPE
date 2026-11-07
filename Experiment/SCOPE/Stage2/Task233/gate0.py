from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.run import MODEL_PYTHON, WebsocketClientPolicy
from Experiment.SCOPE.Stage2.real_rollout import frame_from, image_hash
from Experiment.SCOPE.Stage2.data_guard import STAGE1, sha256, write_json
from Experiment.SCOPE.Stage2.Task233.teacher_candidates import action, candidates, random_action
from Experiment.SCOPE.Stage2.Task233.expert_scoring import expert_at_pose, score
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False, default=lambda value: value.tolist()).encode()).hexdigest()


def select_episodes(manifest):
    generator = random.Random(233)
    records = list(manifest["train"])
    generator.shuffle(records)
    groups = defaultdict(list)
    seen = set()
    for record in records:
        route = record["payload"]["trajectory_id"]
        if route not in seen:
            groups[record["scene_id"]].append(record)
            seen.add(route)
    selected = []
    while len(selected) < 36:
        for scene in sorted(groups):
            if groups[scene]:
                selected.append(groups[scene].pop())
                if len(selected) == 36:
                    break
        if not any(groups.values()) and len(selected) < 36:
            raise ValueError("insufficient disjoint train trajectories")
    return selected


def progress(output, status, reason, **extra):
    write_json(output / "PROGRESS_GATE.json", {"Gate0": status, "Gate1": "NOT_RUN", "Gate2": "NOT_RUN", "Gate3": "NOT_RUN",
               "downstream_locked": status != "PASS", "reason": reason, "optimizer_updates": 0, **extra})


def identity(backend, observation, frames, instruction):
    environment = backend.runtime.env
    return {"images": image_hash(observation["images"]), "pose": np.asarray(observation["body_pose"]).tolist(),
            "history": digest([(image_hash(frame["images"]), np.asarray(frame["body_pose"]).tolist()) for frame in frames]),
            "instruction": hashlib.sha256(instruction.encode()).hexdigest(),
            "metrics": digest(observation["metrics"]), "elapsed_steps": int(environment._elapsed_steps),
            "episode_over": bool(environment.episode_over),
            "stop_called": bool(getattr(environment.task, "is_stop_called", False))}


def restore(backend, runtime, episode, navigation_prefix):
    backend.start_episode(episode, runtime.initial_pose(episode))
    observation = backend.get_observation()
    frames = [frame_from(observation, episode, "train")]
    for actions in navigation_prefix:
        executed = backend.apply_action(observation["pose"], np.asarray(actions))
        observation = executed.observation
        if executed.data_done:
            raise RuntimeError("navigation prefix reached terminal state")
        for intermediate in executed.action_observations:
            frames.append(frame_from(intermediate, episode, "train", len(frames)))
    return observation, frames


def freeze_thresholds(output, rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["anchor_id"]][row["variant"]] = row
    progress_noise, gt_noise = [], []
    for records in groups.values():
        for original, duplicate in (("stop", "stop_repeat"), ("candidate_0", "candidate_repeat")):
            if original not in records or duplicate not in records:
                raise ValueError("incomplete independent noise calibration")
            progress_noise.append(abs(records[original]["progress"] - records[duplicate]["progress"]))
            gt_noise.append(abs(records[original]["score_gt"] - records[duplicate]["score_gt"]))
    if len(groups) != 8:
        raise ValueError("calibration requires eight anchors")
    thresholds = {"calibration_anchors": len(groups), "progress_noise_max": max(progress_noise),
                  "gt_noise_max": max(gt_noise), "min_progress_m": max(.02, 5 * max(progress_noise)),
                  "min_gt_margin": max(.0001, 5 * max(gt_noise)), "minimum_confident_fraction": .25,
                  "minimum_evaluation_anchors": 64, "minimum_heldout_anchors": 32,
                  "paired_ci_level": 1 - .05 / 3, "bootstrap_episode_replicates": 10000,
                  "seed": 233, "calibration_rows_sha256": digest(rows)}
    write_json(output / "FROZEN_THRESHOLDS.json", thresholds)
    return thresholds


def execute(args, client, selection):
    raw = yaml.safe_load((STAGE1 / "configs/G1.yaml").read_text())
    raw["model"].update(uri=f"ws://127.0.0.1:{args.port}", seed=233)
    raw["benchmark"]["kwargs"].update(split="train", distance_metric="geodesic")
    raw["input"]["split"] = "train"
    raw["env"]["kwargs"].update(split="train", max_episode_steps=200)
    raw["output"].update(root=str(args.output), run_name="gate0", save_step_artifacts=False)
    (args.output / "environment.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_eval_config(args.output / "environment.yaml")
    runtime = build_benchmark_runtime(cfg)
    backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=args.habitat_gpu)
    rows, exclusions = [], []
    began = time.monotonic()
    native_steps = 0
    completed_anchors = 0
    target_anchors = getattr(args, "target_anchors", None)
    seen_navigation_states = set()
    try:
        for ordinal, record in enumerate(selection):
            partition_map = getattr(args, "partition_by_episode", None)
            partition = partition_map[record["episode_uid"]] if partition_map is not None else "calibration" if ordinal < 4 else "teacher_train" if ordinal < 20 else "teacher_heldout"
            episode = EvalEpisode(**record)
            navigation_prefix = []
            for navigation_index in range(2):
                if time.monotonic() - began > args.wall_hours * 3600:
                    raise TimeoutError("preregistered Gate0 wall-time budget exhausted")
                anchor_id = f"{getattr(args, 'anchor_prefix', '')}episode_{ordinal:02d}_state_{navigation_index}"
                if target_anchors is not None:
                    preview, preview_frames = restore(backend, runtime, episode, navigation_prefix)
                    native_steps += 8 * len(navigation_prefix)
                    navigation_key = digest({"scene": episode.scene_id, "trajectory": record["payload"]["trajectory_id"],
                                             "pose": np.asarray(preview["body_pose"]).tolist(), "images": image_hash(preview["images"])})
                    if navigation_key in seen_navigation_states:
                        exclusions.append({"episode": episode.episode_uid, "state": navigation_index,
                                           "reason": "duplicate physical navigation state; history or j alone cannot inflate anchors"})
                        break
                microstep = ordinal % 4
                prefix = [action(camera, [0., 0.]) for camera in range(microstep)]
                candidate_set = candidates(233000 + ordinal * 10 + navigation_index)
                variants = [("stop", None, False, prefix)]
                variants += [(f"candidate_{index}", candidate, False, prefix) for index, candidate in enumerate(candidate_set)]
                variants += [("learned", None, True, prefix)]
                variants += [(f"random_{index}", random_action(2330000 + ordinal * 100 + navigation_index * 10 + index), False, prefix)
                             for index in range(3)]
                if prefix:
                    variants.append(("glance_only", None, False, []))
                if partition == "calibration":
                    variants += [("stop_repeat", None, False, prefix), ("candidate_repeat", candidate_set[0], False, prefix)]
                reference = None
                expert = None
                next_actions = None
                next_done = None
                for variant, candidate, learned, perception_prefix in variants:
                    observation, frames = restore(backend, runtime, episode, navigation_prefix)
                    native_steps += 8 * len(navigation_prefix)
                    before_identity = identity(backend, observation, frames, episode.instruction)
                    current_expert = expert_at_pose(backend, observation)
                    if expert is None:
                        expert = current_expert
                    elif current_expert != expert:
                        raise RuntimeError("expert label changed after exact reset/prefix replay")
                    frames[-1]["task233"] = {"prefix": perception_prefix, "candidate": candidate, "learned": learned,
                                              "anchor_id": anchor_id, "save_state": variant == "stop"}
                    identifier = anchor_id + "-" + variant
                    client.reset(session_id=identifier, episode_id=episode.source_episode_id,
                                 statistics_key="vln_train_train", seed=233)
                    prediction = client.predict(frames=frames, instruction=episode.instruction)
                    diagnostic = prediction["metadata"]["task233"]
                    client.close_session(identifier)
                    before_identity.update(context=diagnostic["context_sha256"], grids=diagnostic["context_grids"])
                    state_sha = diagnostic["anchor_state_sha256"]
                    if reference is None:
                        reference = {"environment": copy.deepcopy(before_identity), "anchor": state_sha}
                    valid = before_identity == reference["environment"] and (variant == "glance_only" or state_sha == reference["anchor"])
                    if not valid:
                        write_json(args.output / "RESTORE_FAILURE.json", {"anchor_id": anchor_id, "variant": variant,
                                   "expected": reference, "actual": before_identity, "state_sha": state_sha})
                        raise RuntimeError("Gate0 exact state restoration failed")
                    actions = np.asarray(prediction["actions"])
                    motion = float(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean())
                    stopped = bool(prediction.get("stop")) or motion <= cfg.raw["stop_threshold"]
                    before = float(observation["metrics"]["geodesic_distance_to_goal"])
                    if stopped:
                        backend._latest_payload = backend.runtime.step({"action": "STOP", "action_args": {}})
                        after_observation = backend.get_observation()
                        terminal, truncated = True, False
                        executed_steps = 1
                    else:
                        execution = backend.apply_action(observation["pose"], actions)
                        after_observation = execution.observation
                        executed_steps = execution.diagnostics["executed_native_action_count"]
                        truncated = bool(execution.data_done and backend.runtime.env._elapsed_steps >= 200)
                        terminal = bool(execution.data_done and not truncated)
                    native_steps += executed_steps
                    after = float(after_observation["metrics"]["geodesic_distance_to_goal"])
                    row = {"anchor_id": anchor_id, "ordinal": ordinal, "episode_uid": episode.episode_uid,
                           "trajectory_id": record["payload"]["trajectory_id"], "scene": episode.scene_id,
                           "partition": partition, "split": "train", "state_source": "expert_start" if navigation_index == 0 else "learner_induced",
                           "j": microstep, "variant": variant, "data_kind": "teacher_candidate", "ppo_eligible": False,
                           "navigation_prefix": copy.deepcopy(navigation_prefix), "perception_prefix": copy.deepcopy(perception_prefix),
                           "restoration_valid": valid, "restoration_identity": before_identity, "expert": expert,
                           "distance_before": before, "distance_after": after, "progress": before - after,
                           "navigation_stopped": stopped, "terminated": terminal, "truncated": truncated,
                           "terminal_success": terminal and after < 3, "native_steps": executed_steps,
                           "actions": actions.tolist(), "diagnostic": diagnostic, **score(actions, stopped, expert)}
                    rows.append(row)
                    with (args.output / "candidate_results.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    if variant == "glance_only" or (variant == "stop" and not prefix):
                        next_actions, next_done = actions.tolist(), terminal or truncated
                print(json.dumps({"anchor": anchor_id, "partition": partition, "j": microstep,
                                  "branches": len(variants), "rows": len(rows), "wall_seconds": time.monotonic() - began}), flush=True)
                completed_anchors += 1
                if target_anchors is not None:
                    seen_navigation_states.add(navigation_key)
                    if completed_anchors >= target_anchors:
                        break
                if next_done or next_actions is None:
                    if navigation_index == 0:
                        exclusions.append({"episode": episode.episode_uid, "reason": "no-Focus learner prefix terminated; no live second anchor"})
                    break
                navigation_prefix.append(next_actions)
            write_json(args.output / "execution_progress.json", {"episodes_completed": ordinal + 1, "branches": len(rows),
                       "completed_anchors": completed_anchors, "target_anchors": target_anchors,
                       "native_steps_including_replay": native_steps, "wall_seconds": time.monotonic() - began, "exclusions": exclusions})
            if target_anchors is not None and completed_anchors >= target_anchors:
                break
            if partition_map is None and ordinal == 3:
                thresholds = freeze_thresholds(args.output, rows)
                persisted = [json.loads(line) for line in (args.output / "candidate_results.jsonl").read_text().splitlines()]
                if digest(persisted) != thresholds["calibration_rows_sha256"]:
                    raise RuntimeError("calibration memory/persisted evidence mismatch")
            if partition_map is None and ordinal < 4 and len(exclusions):
                raise RuntimeError("independent calibration anchor shortfall; no adaptive substitution")
        if target_anchors is not None and completed_anchors != target_anchors:
            raise RuntimeError("full train reserve exhausted before requested unique-anchor count")
        write_json(args.output / "COMPLETE.json", {"branches": len(rows), "native_steps_including_replay": native_steps,
                   "completed_anchors": completed_anchors,
                   "wall_seconds": time.monotonic() - began, "exclusions": exclusions, "audit": client._request("audit", {})})
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10233)
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--habitat-gpu", type=int, default=1)
    parser.add_argument("--wall-hours", type=float, default=3.)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if ROOT not in args.output.parents or not 0 < args.wall_hours <= 3:
        parser.error("new Task233 output only; maximum three-hour Gate0 budget")
    args.output.mkdir(parents=True, exist_ok=False)
    source = ROOT.parent / "Task23/artifacts/ppo4096"
    manifest_path = source / "DATA_MANIFEST.json"
    checkpoint = source / "runtime/checkpoints/update_0014.pt"
    selection = select_episodes(json.loads(manifest_path.read_text()))
    write_json(args.output / "DATA_MANIFEST.json", {"train": selection, "val_seen": [], "calibration_episodes": 4,
               "teacher_train_episodes": 16, "teacher_heldout_episodes": 16, "selection_seed": 233,
               "selection": "unique trajectories, scene round-robin, no outcome-based replacement"})
    sources = [checkpoint, manifest_path, ROOT.parent / "Task21/artifacts/release/adapter.pt",
               ROOT.parent / "Task232/REPORT.md", ROOT.parent / "Task232/C_REPORT.md", ROOT.parent / "Task232/C_MECHANISM.md",
               ROOT.parent / "Task231/REPORT.md", ROOT / "TEACHER_PROTOCOL.md"] + list(ROOT.glob("*.py"))
    source_hashes = {str(path): sha256(path) for path in sources}
    write_json(args.output / "SOURCE_INTEGRITY_BEFORE.json", source_hashes)
    historical_manifest = source.parent / "ARTIFACT_MANIFEST.json"
    trusted = {record["path"]: record["sha256"] for record in json.loads(historical_manifest.read_text())["files"]}
    verified = []
    for path in (checkpoint, manifest_path):
        relative = str(path.relative_to(ROOT.parents[3]))
        verified.append({"path": relative, "expected": trusted.get(relative), "actual": source_hashes[str(path)]})
    passed = all(row["expected"] == row["actual"] for row in verified)
    write_json(args.output / "TRUSTED_ASSET_AUDIT.json", {"passed": passed, "assets": verified,
               "historical_manifest_sha256": sha256(historical_manifest)})
    if not passed:
        progress(args.output, "BLOCKED", "historical asset hashes mismatch")
        raise RuntimeError("historical asset hashes mismatch")
    write_json(args.output / "RUN_CONFIG.json", {"seed": 233, "selection_episodes": 36, "target_evaluation_anchors": 64,
               "calibration_anchors": 8, "candidates_per_anchor": 8, "max_predictions": 1008,
               "max_wall_hours": args.wall_hours, "group_sampling": False, "optimizer_updates": 0,
               "role_policy": "user explicitly reauthorized execution; missing agents/exec-agent.md disclosed; Task233 governs",
               "protocol_sha256": sha256(ROOT / "TEACHER_PROTOCOL.md"), "source_hashes": source_hashes})
    progress(args.output, "NOT_RUN", "preflight complete; Gate0 evidence collection beginning")
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
                    raise RuntimeError("teacher model server startup failed; inspect server.log")
                time.sleep(1)
            client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
            client._ws.ping_timeout = 1200
            execute(args, client, selection)
            after = {path: sha256(path) for path in source_hashes}
            write_json(args.output / "SOURCE_INTEGRITY_AFTER.json", {"hashes": after, "unchanged": after == source_hashes})
            if after != source_hashes:
                raise RuntimeError("source integrity changed during audit")
            from Experiment.SCOPE.Stage2.Task233.analyze import analyze
            analyze(args.output)
        except BaseException as error:
            write_json(args.output / "INCOMPLETE.json", {"error": repr(error), "optimizer_updates": 0})
            progress(args.output, "BLOCKED", repr(error))
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
