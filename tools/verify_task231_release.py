from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import json
import math

from verify_release import ROOT, verify


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_evidence(root=ROOT):
    task = root / "Experiment/SCOPE/Stage2/Task231"
    diagnosis = read(task / "diagnosis.json")
    require(diagnosis["new_training_updates"] == 0 and diagnosis["original_training_modified"] is False
            and diagnosis["formal_optimization_started"] is False, "Unexpected training modification")
    integrity = read(task / "source_integrity.json")
    require(integrity["original_run_unchanged"] and not integrity["artifact_failures"]
            and not integrity["source_failures"], "Historical source audit failed")
    groups = defaultdict(list)
    for shard in (0, 1):
        directory = task / f"artifacts/counterfactual32_shard{shard}"
        complete = read(directory / "COMPLETE.json")
        require(complete["episodes"] == 96 and complete["training_updates_performed"] == 0
                and all(complete["freeze_audit"].values()), "Counterfactual completion gate failed")
        for record in read(directory / "counterfactual_results.json"):
            require(record["failure"] is None and not record["training"]["updated"], "Counterfactual failed or trained")
            groups[(record["transitions"], record["mode"])].append(record)
    expected_groups = {(step, mode) for step in (1033, 2054, 4096) for mode in ("native", "forced4")}
    require(set(groups) == expected_groups, "Counterfactual groups mismatch")
    identities = None
    summaries = {(record["transitions"], record["mode"]): record for record in diagnosis["counterfactual"]["groups"]}
    for key, records in groups.items():
        episode_ids = {record["episode_uid"] for record in records}
        require(len(records) == len(episode_ids) == 32, "Counterfactual denominator mismatch")
        if identities is None:
            identities = episode_ids
        require(identities == episode_ids, "Counterfactual pairing mismatch")
        for metric in ("SR", "SPL", "nDTW", "focus_mean", "oracle_success"):
            actual = sum(record[metric] for record in records) / 32
            require(math.isclose(actual, summaries[key][metric], abs_tol=1e-12), f"Summary mismatch: {key} {metric}")
    exported = rows(task / "counterfactual.csv")
    require(len(exported) == 192, "Exported counterfactual denominator mismatch")
    indexed = {(record["transitions"], record["mode"], record["episode_uid"]): record
               for records in groups.values() for record in records}
    require(len({(row["transitions"], row["mode"], row["episode_uid"]) for row in exported}) == 192,
            "Duplicate exported counterfactual")
    for row in exported:
        original = indexed[(int(row["transitions"]), row["mode"], row["episode_uid"])]
        require(all(math.isclose(float(row[key]), original[key], abs_tol=1e-12)
                    for key in ("SR", "SPL", "nDTW", "focus_mean")), "CSV differs from raw results")
    matched = read(task / "artifacts/matched8/matched_results.json")
    require(len(matched) == 128 and all(record["restoration_valid"] for record in matched), "Matched restoration failure")
    require(len({(record["episode_uid"], record["state_index"]) for record in matched}) == 16,
            "Matched state denominator mismatch")
    require(len({record["episode_uid"] for record in matched}) == 8, "Matched episode denominator mismatch")
    numerical = read(task / "artifacts/offline/numerical_audit.json")
    require(numerical["transitions"] == 4096 and len(numerical["updates"]) == 56
            and all(value == 0 for value in numerical["errors"].values()), "Historical numerical audit failed")
    require(len(rows(task / "artifacts/offline/numerical_transitions.csv")) == 4096, "Numerical CSV denominator mismatch")
    probes = defaultdict(list)
    for record in rows(task / "checkpoint_policy_probe.csv"):
        probes[(int(record["transitions"]), record["split"])].append(record["state_path"])
    require(set(probes) == {(step, split) for step in (0, 1033, 2054, 3100, 4096) for split in ("train", "val_seen")},
            "Probe groups mismatch")
    for (step, split), states in probes.items():
        require(len(states) == len(set(states)) == 128 and set(states) == set(probes[(0, split)]), "Probe states mismatch")
    historical = rows(task / "paired_episodes.csv")
    require(len(historical) == 160, "Historical pairing denominator mismatch")
    for step in (0, 1033, 2054, 3100, 4096):
        selected = [row for row in historical if int(row["transitions"]) == step]
        require(len(selected) == 32 and {row["episode_uid"] for row in selected} == identities,
                "Historical pairing identities mismatch")
    for name in ("counterfactual32_shard0.log", "counterfactual32_shard1.log", "matched8.log", "offline.log",
                 "artifacts/counterfactual32_shard0/server.log", "artifacts/counterfactual32_shard1/server.log"):
        require((task / name).is_file() and (task / name).stat().st_size > 0, f"Missing key log: {name}")
    with gzip.open(task / "FILES_MANIFEST.tsv.gz", "rt") as stream:
        originals = {record["path"]: record for record in csv.DictReader(stream, delimiter="\t")}
    placeholders = read(root / "docs/release/TASK231_CHECKPOINT_PLACEHOLDERS.json")
    require(len(placeholders) == len(set(placeholders)) == 23, "Placeholder denominator mismatch")
    require(set(placeholders) == {str(path.relative_to(root)) for path in task.rglob("*.placeholder.json")},
            "Placeholder inventory mismatch")
    for relative in placeholders:
        record = read(root / relative)
        require(record["uploaded"] is False and relative == record["checkpoint"] + ".placeholder.json",
                "Invalid checkpoint placeholder")
        original = originals[str((root / record["checkpoint"]).relative_to(task))]
        require(record["sha256"] == original["sha256"] and record["bytes"] == int(original["bytes"]),
                "Checkpoint placeholder hash mismatch")
    require(not list(task.rglob("*.pt")), "Task231 tensors must remain local")
    return {"status": "PASS", "counterfactual_episodes": 192, "matched_states": 16, "matched_branches": 128,
            "actor_probes": 1280, "audited_train_transitions": 4096, "new_training_updates": 0,
            "checkpoint_placeholders": 23, "full_trace_replay": "requires local originals"}


def verify_task231(root=ROOT):
    archive = verify(root)
    return {"archive_integrity": archive["integrity"], **verify_evidence(root)}


if __name__ == "__main__":
    print(json.dumps(verify_task231(), indent=2))
