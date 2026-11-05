from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
STAGE1 = ROOT.parent / "Stage1_Feasibility"
PARTITION = ROOT.parent / "Stage1/STAGE1_GROUP_CONFIGS/manifests/r2r_selector_train_dev.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def partition():
    payload = json.loads(PARTITION.read_text())
    if set(payload["training_scenes"]) & set(payload["development_scenes"]):
        raise ValueError("train/dev scene overlap")
    if set(payload["training_episode_ids"]) & set(payload["development_evaluation_episode_ids"]):
        raise ValueError("train/dev episode overlap")
    return payload


def check_episode(episode_id, scene_id, purpose):
    payload = partition()
    if purpose not in ("train", "dev"):
        raise PermissionError("Stage2 excludes held-out evaluation")
    identifiers = payload["training_episode_ids"] if purpose == "train" else payload["development_evaluation_episode_ids"]
    scenes = payload["training_scenes"] if purpose == "train" else payload["development_scenes"]
    if str(episode_id) not in identifiers or scene_id not in scenes:
        raise PermissionError("episode/scene outside authorized partition")


def episodes(purpose, count):
    payload = partition()
    if purpose == "dev":
        records = json.loads((STAGE1 / "DEV_EPISODES.json").read_text())[:count]
    elif purpose == "train":
        source = payload["sources"][0]
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("annotation source changed")
        with gzip.open(source["path"], "rt") as stream:
            by_id = {str(record["episode_id"]): record for record in json.load(stream)["episodes"]}
        records = []
        for identifier in payload["training_episode_ids"][:count]:
            record = dict(by_id[identifier])
            instruction = record["instruction"]["instruction_text"].strip()
            points = record.get("reference_path", [])
            import numpy as np
            length = float(np.linalg.norm(np.diff(np.asarray(points), axis=0), axis=1).sum())
            record.update(episode_id=identifier, instruction=instruction, task_name="r2r", split="train",
                          gt_path_length=length, reference_points=[[-point[2], point[0], -point[1]] for point in points])
            records.append({"episode_uid": f"scope_stage2_train:{identifier}", "source_episode_id": identifier,
                            "scene_id": Path(record["scene_id"]).stem, "instruction": instruction, "source": "vlnce_r2r",
                            "input_namespace": "scope_stage2_train", "input_root": "/home/data/czh/Third_Party/tasks",
                            "payload": record})
    else:
        raise PermissionError("only train/dev allowed")
    for record in records:
        check_episode(record["source_episode_id"], record["scene_id"], purpose)
    return records


def protected_snapshot():
    paths = []
    for directory in (ROOT.parent / "Stage1", STAGE1, PROJECT / "docs/Logs/Task1"):
        paths.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    manifest = json.loads((STAGE1 / "EXECUTION_MANIFEST.json").read_text())
    paths.append(Path(manifest["checkpoint"]["path"]))
    return {str(path): sha256(path) for path in sorted(paths)}
