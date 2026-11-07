from __future__ import annotations

import gzip
import json

from verify_release import ROOT, digest, verify


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_evidence(root=ROOT):
    task = root / "Experiment/SCOPE/Stage2/Task23"
    run = task / "artifacts/ppo4096"
    for relative in ("driver.log", "server.log", "summary/training.csv", "summary/audit.log",
                     "runtime/scope.jsonl.gz", "summary/loss.svg", "summary/kl.svg", "summary/entropy.svg"):
        require((run / relative).is_file() and (run / relative).stat().st_size > 0,
                f"Missing key training log: {relative}")
    complete = read(run / "COMPLETE.json")
    for key, expected in {"target": 4096, "collected": 4096, "trained": 4096,
                          "buffer_transitions": 0, "updates": 56, "optimizer_steps": 570,
                          "validation_transitions": 2452}.items():
        require(complete[key] == expected, f"Training denominator mismatch: {key}")
    require(all(complete["freeze_audit"].values()), "Frozen models changed")
    for record in read(task / "SOURCE_MANIFEST.json")["files"]:
        source = root / record["path"]
        if record["path"] == "docs/index.md":
            source = root / "docs/release/task23_source_index.md"
        require(digest(source) == record["sha256"], f"Task23 source mismatch: {record['path']}")
    updates = [read(path) for path in sorted((run / "runtime/updates").glob("update_*.json"))]
    require(len(updates) == 56, "Update count mismatch")
    transitions = optimizer_steps = 0
    for number, update in enumerate(updates, 1):
        transitions += update["transitions"]
        optimizer_steps += len(update["minibatches"])
        require(update["update"] == number and update["total_transitions"] == transitions,
                "Update transition cursor mismatch")
        require(update["optimizer_steps"] == optimizer_steps, "Optimizer cursor mismatch")
        require(update["initial_ratio_max_error"] == 0 and not update["diagnostic"], "PPO ratio mismatch")
        require(update["actor_parameter_l2_change"] > 0 and update["critic_parameter_l2_change"] > 0,
                "Actor/Critic not updated")
        require(all(update["frozen_audit"].values()), "Update freeze audit failed")
    require((transitions, optimizer_steps) == (4096, 570), "Update denominator mismatch")
    episodes = read(run / "episodes_summary.json")
    require(len(episodes) == 436 and all(record["failure"] is None for record in episodes),
            "Episode count or execution failure")
    for purpose, count, steps in (("train", 276, 4096), ("val_seen", 160, 2452)):
        selected = [record for record in episodes if record["purpose"] == purpose]
        require(len(selected) == count and sum(record["training"]["transitions"] for record in selected) == steps,
                f"Episode denominator mismatch: {purpose}")
    traces = read(run / "runtime/trace_summary.json.gz")
    require(len(traces) == len({record["transition_id"] for record in traces}) == 6548,
            "Trace denominator or duplicate ID")
    require(all(record["current_visual_tokens"] == 64 for record in traces), "Token budget mismatch")
    require({record["trace_path"] for record in traces} ==
            {path for record in episodes for path in record["training"]["trace_paths"]}, "Trace IDs differ")
    history = read(run / "runtime/validation_history.json")
    require([record["transitions"] for record in history] == [0, 1033, 2054, 3100, 4096],
            "Validation schedule mismatch")
    require(all(record["episodes"] == 32 and record["split"] == "val_seen" for record in history),
            "Validation denominator mismatch")
    require(history[0]["SR"] == history[-1]["SR"] == 0.34375 and history[1]["SR"] == 0.5625,
            "Validation SR mismatch")
    summary = read(run / "summary/SUMMARY.json")
    require(summary["complete"] and summary["trace_audit"]["checked"], "Historical audit incomplete")
    require(summary["validation_history"] == history, "Summary history differs")
    require(summary["validation_focus"][-1]["focus_histogram"] == {"0": 498}, "Final collapse evidence differs")
    execution = read(run / "EXECUTION_AUDIT.json")
    require(execution["training_and_validation_completed"] and execution["driver_exit_code"] == 134
            and execution["final_policy_effectiveness_accepted"] is False, "Execution caveats changed")
    final = read(run / "FINAL_CHECKPOINT_AUDIT.json")
    require(final["passed"] and final["all_four_models_and_two_optimizers_exact"], "Historical restore audit failed")
    require(final["cursor"] == {"train_episode_index": 276, "validation_round": 5, "next_validation_at": 5120},
            "Final checkpoint cursor mismatch")
    manifest = {record["path"]: record for record in read(task / "artifacts/ARTIFACT_MANIFEST.json.gz")["files"]}
    placeholders = read(root / "docs/release/TASK23_CHECKPOINT_PLACEHOLDERS.json")
    actual = {str(path.relative_to(root)) for path in task.rglob("*.pt.placeholder.json")}
    require(len(placeholders) == len(set(placeholders)) == 11 and set(placeholders) == actual,
            "Checkpoint placeholder count mismatch")
    for relative in placeholders:
        record = read(root / relative)
        require(record["uploaded"] is False and relative == record["checkpoint"] + ".placeholder.json",
                "Invalid checkpoint placeholder")
        require(not (root / record["checkpoint"]).exists(), "Checkpoint must remain local")
        original = manifest[record["checkpoint"]]
        require(record["sha256"] == original["sha256"] and record["bytes"] == original["bytes"],
                "Checkpoint placeholder digest mismatch")
    require(not list(task.rglob("*.pt")), "Unexpected Task23 tensor/checkpoint upload")
    return {"status": "PASS", "train_transitions": 4096, "validation_transitions": 2452,
            "updates": 56, "optimizer_steps": 570, "checkpoint_placeholders": 11,
            "checkpoint_weights_uploaded": False, "full_trace_replay": "not included",
            "driver_exit_code": 134, "final_policy_effectiveness_accepted": False}


def verify_task23(root=ROOT):
    archive = verify(root)
    return {"archive_integrity": archive["integrity"], **verify_evidence(root)}


if __name__ == "__main__":
    print(json.dumps(verify_task23(), indent=2))
