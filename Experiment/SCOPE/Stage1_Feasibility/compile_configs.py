from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import yaml
import numpy as np

from policies import MODES, TOKENS

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SIMPLENAV = Path("/home/data/czh/Third_Party/baselines/SimpleNav")
SOURCE = ROOT.parent / "Stage1/STAGE1_GROUP_CONFIGS"


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def materialize():
    partition_path = SOURCE / "manifests/r2r_selector_train_dev.json"
    partition = json.loads(partition_path.read_text())
    for source in partition["sources"]:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("changed source annotations")
    with gzip.open(partition["sources"][0]["path"], "rt") as stream:
        annotations = json.load(stream)["episodes"]
    records = {str(record["episode_id"]): record for record in annotations}
    identifiers = partition["development_evaluation_episode_ids"]
    if len(identifiers) != 64 or len(set(identifiers)) != 64:
        raise ValueError("expected exactly 64 unique development episodes")
    train_scenes = set(partition["training_scenes"])
    dev_scenes = set(partition["development_scenes"])
    if train_scenes & dev_scenes or set(identifiers) & set(partition["training_episode_ids"]):
        raise ValueError("train/dev overlap")
    evaluation_scenes = set()
    for name in ("r2r_val_unseen.json", "rxr_val_unseen_guide_en.json"):
        manifest = json.loads((SOURCE / "manifests" / name).read_text())
        evaluation_scenes.update(row["scene_id"] for row in manifest["episodes"])
    if dev_scenes & evaluation_scenes or train_scenes & evaluation_scenes:
        raise ValueError("evaluation scene leakage")
    episodes = []
    for ordinal, identifier in enumerate(identifiers):
        record = dict(records[identifier])
        scene = Path(record["scene_id"]).stem
        if scene not in dev_scenes:
            raise ValueError("non-development scene")
        instruction = record["instruction"]["instruction_text"].strip()
        reference = np.asarray(record.get("reference_path", []), dtype=np.float32)
        path_length = float(np.linalg.norm(reference[1:, :3] - reference[:-1, :3], axis=1).sum()) if len(reference) >= 2 else 0.0
        record.update(episode_id=identifier, instruction=instruction, task_name="r2r", split="train",
                      gt_path_length=path_length,
                      reference_points=[[-point[2], point[0], -point[1]] for point in record.get("reference_path", [])])
        episodes.append({"episode_uid": f"scope_feasibility_dev:{identifier}", "source_episode_id": identifier,
                         "scene_id": scene, "instruction": instruction, "source": "vlnce_r2r",
                         "input_namespace": "scope_feasibility_dev", "input_root": "/home/data/czh/Third_Party/tasks",
                         "payload": record})
    write_json(ROOT / "DEV_EPISODES.json", episodes)
    write_json(ROOT / "DATA_ISOLATION.json", {"passed": True, "count": 64,
               "ordered_episode_ids": identifiers, "train_scenes": sorted(train_scenes),
               "dev_scenes": sorted(dev_scenes), "val_unseen_scenes": sorted(evaluation_scenes),
               "partition_sha256": sha256(partition_path), "source": partition["sources"],
               "warning": "dev from original train; not isolated from pretrained backbone data"})
    return episodes


def compile_bundle():
    if (ROOT / "EXECUTION_MANIFEST.json").exists():
        raise FileExistsError("already frozen; do not silently overwrite execution manifest")
    episodes = materialize()
    common = yaml.safe_load((SOURCE / "common.yaml").read_text())
    checkpoint = common["checkpoint"]
    if sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    configs = {}
    for group, mode in MODES.items():
        config = yaml.safe_load((SIMPLENAV / "benchmark/vlnce/r2r/config_standard_local.yaml").read_text())
        config["policy_mode"] = mode
        config["benchmark"].update(max_samples=64, name="SCOPE-Stage1-Feasibility-v1")
        config["benchmark"]["kwargs"]["split"] = "train"
        config["input"].update(split="train", namespace="scope_feasibility_dev")
        config["model"].update(uri="ws://127.0.0.1:10095", seed=42)
        config["env"]["kwargs"].update(split="train", image_size=512, seed=0)
        config["parallel"]["gpu_ids"] = [1]
        config["output"].update(root=str(ROOT / "runs"), run_name=group)
        config.pop("driver_paths_file", None)
        path = ROOT / "configs" / f"{group}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        write_json(ROOT / "manifests" / f"{group}.json", {"group": group, "mode": mode,
                   "current_tokens": TOKENS[group], "run_seed": 42,
                   "episodes": [{"ordinal": ordinal, "episode_id": episode["source_episode_id"],
                                 "scene_id": episode["scene_id"], "session_seed": 42 + ordinal}
                                for ordinal, episode in enumerate(episodes)]})
        configs[group] = {"config": str(path), "sha256": sha256(path), "mode": mode,
                          "manifest_sha256": sha256(ROOT / "manifests" / f"{group}.json")}
    write_json(ROOT / "EXECUTION_MANIFEST.json", {"version": "feasibility-v1.1", "authorized": True,
               "training_authorized": False, "protocol_sha256": sha256(ROOT / "STAGE1_FEASIBILITY_PROTOCOL.md"),
               "checkpoint": checkpoint, "episodes_sha256": sha256(ROOT / "DEV_EPISODES.json"),
               "planned_per_group": 64, "groups": configs})


def validate_freeze():
    manifest = json.loads((ROOT / "EXECUTION_MANIFEST.json").read_text())
    if manifest["protocol_sha256"] != sha256(ROOT / "STAGE1_FEASIBILITY_PROTOCOL.md"):
        raise ValueError("protocol changed after freeze")
    if manifest["episodes_sha256"] != sha256(ROOT / "DEV_EPISODES.json"):
        raise ValueError("episodes changed after freeze")
    for group, config in manifest["groups"].items():
        if sha256(config["config"]) != config["sha256"]:
            raise ValueError("configuration changed after freeze")
        if sha256(ROOT / "manifests" / f"{group}.json") != config["manifest_sha256"]:
            raise ValueError("group manifest changed after freeze")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate_freeze()
    else:
        compile_bundle()
