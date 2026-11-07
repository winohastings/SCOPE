from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.data import InputConfig, VLNCEInputAdapter, load_manifest
from Experiment.SCOPE.Stage2.Task23.audit import compare_nested
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save


def main():
    source = ROOT.parent / "Task23/artifacts/ppo4096"
    checkpoint = source / "runtime/checkpoints/update_0014.pt"
    original_records = json.loads((source.parent / "ARTIFACT_MANIFEST.json").read_text())["files"]
    expected = {record["path"]: record["sha256"] for record in original_records}
    relative = str(checkpoint.relative_to(ROOT.parents[3]))
    assert sha256(checkpoint) == expected[relative]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["collected"] == 1033 and payload["trainer"]["transitions_seen"] == 1033
    assert payload["episode_boundary"] and not payload["buffer_boundaries"]
    manifest = load_manifest(source / "DATA_MANIFEST.json")
    config = json.loads((source / "config.json").read_text())
    original_validation = copy.deepcopy(manifest["val_seen"])
    old_ids = {record["source_episode_id"] for record in original_validation}
    old_routes = {record["payload"]["trajectory_id"] for record in original_validation}
    inputs = InputConfig(type="vlnce_r2r", adapter_class_path="benchmark.vlnce.inputs:VLNCEInputAdapter",
                         path=Path("/home/data/czh/Third_Party/tasks"), namespace="agm_ppo_val_seen",
                         raw={"task_name": "r2r", "split": "val_seen"})
    candidates = [asdict(episode) for episode in VLNCEInputAdapter().load_episodes(inputs, max_samples=None)]
    candidates = [record for record in candidates if record["source_episode_id"] not in old_ids
                  and record["payload"]["trajectory_id"] not in old_routes]
    random.Random(232).shuffle(candidates)
    groups = defaultdict(list)
    for record in candidates:
        groups[record["scene_id"]].append(record)
    independent = [groups[scene][0] for scene in sorted(groups)][:32]
    assert len(independent) == 32
    assert not old_routes & {record["payload"]["trajectory_id"] for record in independent}
    manifest.update(val_seen=original_validation + independent, historical_val_seen=original_validation,
                    independent_val_seen=independent, task232_selection_seed=232,
                    task232_selection="scene-balanced; excludes historical32 episode AND trajectory; no outcome selection",
                    historical_manifest_sha256=sha256(source / "DATA_MANIFEST.json"))
    assert manifest["train"] == load_manifest(source / "DATA_MANIFEST.json")["train"]
    output = ROOT / "frozen"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "DATA_MANIFEST.json", manifest)
    probe_path = ROOT.parent / "Task231/checkpoint_policy_probe.csv"
    import csv
    probe_rows = list(csv.DictReader(probe_path.open()))
    fixed = [{"split": row["split"], "path": row["state_path"], "sha256": sha256(row["state_path"])}
             for row in probe_rows if row["transitions"] == "0"]
    geometry = []
    for episode_path in sorted((source / "episodes").glob("val_001_*/result.json")):
        episode = json.loads(episode_path.read_text())
        for path in episode["training"]["trace_paths"][:2]:
            trace = torch.load(path, map_location="cpu", weights_only=True)
            if trace["focus_count"] == 4:
                geometry.append({"path": path, "sha256": sha256(path), "scene": episode["scene_id"]})
    write_json(output / "PROBE_MANIFEST.json", {"pre_focus": fixed, "geometry": geometry,
                                               "geometry_scope": "frozen historical forced-four intermediate states; not fresh fused-feature rollout"})
    for branch in ("U", "A", "B"):
        settings = config | {"budget": 512, "continuation_budgets": [1024], "validation_episodes": 64,
                             "validation_every_transitions": 256, "task232_branch": branch,
                             "focus_cost": .0025 if branch == "A" else .01, "fork_transitions": 1033,
                             "parent_checkpoint_sha256": sha256(checkpoint), "protocol_sha256": sha256(ROOT / "PROTOCOL.md"),
                             "probe_manifest_sha256": sha256(output / "PROBE_MANIFEST.json")}
        if branch == "B":
            settings.update(rollout_minimum=128, actor_learning_rate=3e-5, critic_learning_rate=1e-4)
        write_json(output / f"config_{branch}.json", settings)
        fork = copy.deepcopy(payload)
        fork["trainer"]["config"] = settings
        fork["trainer"]["actor_optimizer"]["param_groups"][0]["lr"] = settings["actor_learning_rate"]
        fork["trainer"]["critic_optimizer"]["param_groups"][0]["lr"] = settings["critic_learning_rate"]
        fork["provenance"] = payload["provenance"] | {"data_manifest_sha256": sha256(output / "DATA_MANIFEST.json"),
                                                     "task232_config_sha256": sha256(output / f"config_{branch}.json")}
        fork["cursor"] = dict(payload["cursor"], next_validation_at=1289, validation_round=0)
        fork["best"] = None
        fork["checkpoint_history"] = []
        for key in ("actor", "critic", "old_actor", "old_critic", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "minibatch_rng"):
            compare_nested(payload["trainer"][key], fork["trainer"][key], key)
        for key in ("actor_optimizer", "critic_optimizer"):
            compare_nested(payload["trainer"][key]["state"], fork["trainer"][key]["state"], key)
        atomic_save(output / f"fork_{branch}.pt", fork)
    write_json(output / "FORK_AUDIT.json", {"passed": True, "parent": str(checkpoint), "parent_sha256": sha256(checkpoint),
               "new_manifest_sha256": sha256(output / "DATA_MANIFEST.json"), "train_order_identical": True,
               "independent_episodes": 32, "independent_trajectory_overlap": 0, "train_cursor": payload["cursor"]["train_episode_index"],
               "optimizer_rule": "retain all moments and steps; B changes only param-group LR", "buffer_empty": True,
               "actor_critic_old_snapshots_rng_bitwise_equal": True, "probe_states": len(fixed), "geometry_states": len(geometry),
               "diagnosis_sha256": sha256(ROOT.parent / "Task231/diagnosis.json"),
               "diagnosis_report_sha256": sha256(ROOT.parent / "Task231/REPORT.md")})
    print(json.dumps({"prepared": str(output), "independent_scenes": len(groups), "fork": str(checkpoint)}))


if __name__ == "__main__":
    main()
