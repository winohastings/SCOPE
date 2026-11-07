from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, "/home/data/czh/Third_Party/baselines/SimpleNav")
from Experiment.SCOPE.Stage2.data_guard import PARTITION, STAGE1, episodes, partition, sha256, write_json
from benchmark.common.config import InputConfig
from benchmark.vlnce.inputs import VLNCEInputAdapter


def create_manifest(path, config):
    path = Path(path)
    if path.exists():
        raise FileExistsError("do not replace an experiment's frozen data manifest")
    training = episodes("train", len(partition()["training_episode_ids"]))
    rng = random.Random(config["seed"])
    rng.shuffle(training)
    val_path = Path("/home/data/czh/Third_Party/tasks/R2R-CE/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz")
    inputs = InputConfig(type="vlnce_r2r", adapter_class_path="benchmark.vlnce.inputs:VLNCEInputAdapter",
                         path=Path("/home/data/czh/Third_Party/tasks"), namespace="agm_ppo_val_seen",
                         raw={"task_name": "r2r", "split": "val_seen"})
    validation = [asdict(episode) for episode in VLNCEInputAdapter().load_episodes(inputs, max_samples=None)]
    rng.shuffle(validation)
    by_scene = defaultdict(list)
    for record in validation:
        by_scene[record["scene_id"]].append(record)
    selected = []
    while len(selected) < min(config["validation_episodes"], len(validation)):
        for records in by_scene.values():
            if records and len(selected) < config["validation_episodes"]:
                selected.append(records.pop())
    manifest = {"schema": config["schema"], "seed": config["seed"], "train": training, "val_seen": selected,
                "partition_sha256": sha256(PARTITION), "val_seen_source": str(val_path), "val_seen_sha256": sha256(val_path),
                "val_seen_full_count": len(validation), "val_seen_selection": config["validation_selection"],
                "train_source": partition()["sources"][0], "val_unseen_accessed": False}
    write_json(path, manifest)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest["partition_sha256"] != sha256(PARTITION):
        raise ValueError("training partition changed")
    if sha256(manifest["val_seen_source"]) != manifest["val_seen_sha256"]:
        raise ValueError("val_seen source changed")
    if sha256(manifest["train_source"]["path"]) != manifest["train_source"]["sha256"]:
        raise ValueError("training annotations changed")
    if not manifest["train"] or not manifest["val_seen"] or manifest["val_unseen_accessed"]:
        raise ValueError("invalid PPO-only data manifest")
    return manifest
