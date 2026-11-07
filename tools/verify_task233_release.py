from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from verify_release import ROOT, digest, verify


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(root, original):
    return root / ("Experiment/" + original.split("/Experiment/", 1)[1])


def verify_evidence(root=ROOT):
    task = root / "Experiment/SCOPE/Stage2/Task233"
    gate0 = task / "artifacts/gate0_v3"
    expanded = task / "artifacts/fulltrain256_v1"
    warmup = task / "artifacts/gate1_v1"
    teacher = read(gate0 / "teacher_signal_audit.json")
    require(teacher["status"] == "PASS" and all(teacher["checks"].values())
            and teacher["anchors"] == 72 and teacher["branches"] == 1006, "Gate0 evidence mismatch")
    for directory, filename in ((gate0, "VERIFICATION.json"), (expanded, "EXPANSION_VERIFICATION.json"),
                                (warmup, "INDEPENDENT_VERIFICATION.json")):
        audit = read(directory / filename)
        require(audit["passed"] and all(audit["checks"].values()), f"Historical audit failed: {filename}")
    evidence = {}
    for directory, expected_count in ((gate0, 1006), (expanded, 2640)):
        with gzip.open(directory / "candidate_results.jsonl.gz", "rt") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        require(len(records) == expected_count, "Candidate denominator mismatch")
        complete = read(directory / "COMPLETE.json")
        require(complete["branches"] == complete["audit"]["model_predictions"] == len(records), "Branch accounting mismatch")
        require(complete["audit"]["optimizer_updates"] == complete["audit"]["critic_training_samples"] == 0
                and all(complete["audit"]["checks"].values()), "Teacher collection trained a model")
        require(sum(record["diagnostic"]["focus_count"] for record in records) == complete["audit"]["focus_encoder_calls"],
                "Focus accounting mismatch")
        grouped = defaultdict(list)
        for record in records:
            require(record["split"] == "train" and record["data_kind"] == "teacher_candidate"
                    and record["ppo_eligible"] is False and record["restoration_valid"], "Teacher data boundary violation")
            require(record["diagnostic"]["current_tokens"] == 64, "Token budget mismatch")
            grouped[record["anchor_id"]].append(record)
        evidence[directory.name] = grouped
    anchors = read(expanded / "ANCHOR_MANIFEST_256.json")["anchors"]
    require(len(anchors) == len({record["anchor_id"] for record in anchors}) == 256, "Anchor denominator mismatch")
    require(len({record["scene"] for record in anchors}) == 61, "Scene denominator mismatch")
    require(Counter(record["partition"] for record in anchors) == {"teacher_train": 176, "teacher_heldout": 80},
            "Partition denominator mismatch")
    require(Counter(record["state_source"] for record in anchors) == {"expert_start": 128, "learner_induced": 128},
            "State-source denominator mismatch")
    trajectories = defaultdict(set)
    by_id = {}
    branch_count = 0
    for anchor in anchors:
        by_id[anchor["anchor_id"]] = anchor
        trajectories[anchor["partition"]].add((anchor["scene"], str(anchor["trajectory_id"])))
        path = Path(str(local_path(root, anchor["state_path"])) + ".gz")
        with gzip.open(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        require(actual == anchor["state_sha256"], "Teacher state hash mismatch")
        directory = local_path(root, anchor["evidence"]).parent.name
        selected = evidence[directory][anchor["anchor_id"]]
        variants = {record["variant"] for record in selected}
        require(len(variants) == len(selected) and {f"candidate_{index}" for index in range(8)} <= variants,
                "Missing or repeated teacher candidate")
        require(all(record["partition"] == anchor["partition"] for record in selected), "Candidate partition mismatch")
        branch_count += len(selected)
    require(not trajectories["teacher_train"] & trajectories["teacher_heldout"], "Trajectory leakage")
    require(branch_count == 3520, "Noncalibration branch denominator mismatch")
    progress = read(warmup / "PROGRESS_GATE.json")
    require(progress["Gate0"] == "PASS" and progress["Gate1"] == "INCONCLUSIVE"
            and progress["Gate2"] == progress["Gate3"] == "NOT_RUN" and progress["downstream_locked"],
            "Unsupported downstream acceptance")
    audit = read(warmup / "OFFLINE_AUDIT.json")
    require(audit["optimizer_updates"] == 186 and audit["optimizer_updates_by_phase"] ==
            {"task22_reset": 27, "update14_reset": 27, "warmup": 132}, "Optimizer denominator mismatch")
    training = rows(warmup / "training.csv")
    require(Counter(record["phase"] for record in training) == audit["optimizer_updates_by_phase"], "Training log count mismatch")
    require(all(math.isfinite(float(record[key])) for record in training for key in ("loss", "grad_norm")),
            "Nonfinite training log")
    for phase in audit["optimizer_updates_by_phase"]:
        require([int(record["update"]) for record in training if record["phase"] == phase] ==
                list(range(1, audit["optimizer_updates_by_phase"][phase] + 1)), "Training update cursor mismatch")
    for stage in ("before", "after"):
        predictions = rows(warmup / f"{stage}_predictions.csv")
        require(len(predictions) == len({record["anchor_id"] for record in predictions}) == 256,
                "Prediction denominator mismatch")
        for partition in ("teacher_train", "teacher_heldout"):
            for confident in (False, True):
                selected = [record for record in predictions if record["partition"] == partition
                            and (not confident or record["confident"] == "True")]
                summary = audit[stage][partition + ("_confident" if confident else "_all")]
                require(len(selected) == summary["count"], "Supervision denominator mismatch")
                weights = sum(float(record["weight"]) for record in selected)
                metrics = {"weighted_nll": sum(float(record["weight"]) * float(record["nll"]) for record in selected) / weights,
                           "top1": sum(float(record["top1"]) for record in selected) / len(selected),
                           "camera_hit": sum(float(record["camera_hit"]) for record in selected) / len(selected)}
                require(all(math.isclose(value, summary[key], abs_tol=1e-10) for key, value in metrics.items()),
                        "Supervision metric mismatch")
    train_manifest = read(warmup / "TRAIN_MANIFEST.json")
    require(len(train_manifest["train"]) == len(set(train_manifest["train"])) == 81, "Training anchor denominator mismatch")
    require(all(by_id[identifier]["partition"] == "teacher_train" for identifier in train_manifest["train"]),
            "Heldout sample used in training")
    require(set(train_manifest["teacher_heldout"]) ==
            {identifier for identifier, record in by_id.items() if record["partition"] == "teacher_heldout"},
            "Heldout manifest mismatch")
    require(audit["after"]["teacher_heldout_confident"]["weighted_nll"] > audit["before"]["teacher_heldout_confident"]["weighted_nll"]
            and audit["after"]["teacher_heldout_confident"]["top1"] < audit["before"]["teacher_heldout_confident"]["top1"],
            "Heldout degradation evidence changed")
    verification = read(warmup / "INDEPENDENT_VERIFICATION.json")
    for original, substitution in verification["post_run_source_substitutions"].items():
        require(digest(local_path(root, original)) == substitution["current_sha256"]
                and digest(local_path(root, substitution["executed_source_snapshot"])) == substitution["executed_sha256"],
                "Executed/current source snapshot mismatch")
    placeholders = read(root / "docs/release/TASK233_CHECKPOINT_PLACEHOLDERS.json")
    require(len(placeholders) == len(set(placeholders)) == 5 and set(placeholders) ==
            {str(path.relative_to(root)) for path in task.rglob("*.placeholder.json")}, "Placeholder inventory mismatch")
    for relative in placeholders:
        record = read(root / relative)
        require(record["uploaded"] is False and relative == record["checkpoint"] + ".placeholder.json",
                "Invalid checkpoint placeholder")
        if record["checkpoint"].endswith("warmup_last.pt"):
            require(record["sha256"] == verification["checkpoint_sha256"], "Final checkpoint hash mismatch")
    require(not list(task.rglob("*.pt")), "Raw tensor/checkpoint must not be uploaded")
    for name in ("gate0_v3.log", "fulltrain256_v1.log", "gate1_v1.log", "artifacts/gate1_v1/training.csv"):
        require((task / name).is_file() and (task / name).stat().st_size > 0, f"Missing key log: {name}")
    return {"status": "PASS", "Gate0": "PASS", "Gate1": "INCONCLUSIVE", "Gate2": "NOT_RUN", "Gate3": "NOT_RUN",
            "noncalibration_anchors": 256, "credible_train_anchors": 81, "supervised_optimizer_updates": 186,
            "checkpoint_placeholders": 5, "new_PPO_or_Group_training": False}


def verify_task233(root=ROOT):
    archive = verify(root)
    return {"archive_integrity": archive["integrity"], **verify_evidence(root)}


if __name__ == "__main__":
    print(json.dumps(verify_task233(), indent=2))
