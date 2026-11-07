from __future__ import annotations

import csv
import gzip
import json
import math

from verify_release import ROOT, verify


def read(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_evidence(root=ROOT):
    task = root / "Experiment/SCOPE/Stage2/Task232"
    manifest = read(task / "frozen/DATA_MANIFEST.json.gz")
    independent, historical = manifest["independent_val_seen"], manifest["historical_val_seen"]
    require(len(independent) == len(historical) == 32, "Evaluation manifest denominator mismatch")
    for key in ("source_episode_id", "trajectory_id"):
        sets = [{record[key] if key == "source_episode_id" else record["payload"][key] for record in group}
                for group in (independent, historical)]
        require(not sets[0] & sets[1], "Independent evaluation overlaps historical set")
    fork = read(task / "frozen/FORK_AUDIT.json")
    require(fork["passed"] and fork["buffer_empty"] and fork["train_cursor"] == 70
            and fork["actor_critic_old_snapshots_rng_bitwise_equal"], "Historical fork audit failed")
    require(read(task / "frozen/C_FORK_AUDIT.json")["full_state_bitwise_equal"], "Historical C fork audit failed")
    gate = read(task / "artifacts/unchanged_gate/TRAINING_GATE.json")
    require(gate["passed"] and gate["fresh_transitions"] == 64 and gate["historical_forward_bitwise_equal"]
            and gate["restore_bitwise_equal"], "Unchanged fork gate failed")
    parent = read(root / "Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096/runtime/updates/update_0014.json")
    fresh_total = 0
    for branch, directory, budget, updates_count in (("U", "unchanged_gate", 64, 1), ("A", "branch_A512", 512, 6),
                                                   ("B", "branch_B512", 512, 2), ("C0", "branch_C0512", 512, 6),
                                                   ("C", "branch_C512", 512, 6)):
        run = task / "artifacts" / directory
        complete = read(run / "COMPLETE.json")
        summary = read(run / "analysis/summary.json")
        require(complete["new_training_transitions"] == budget,
                f"Training denominator mismatch: {branch}")
        if branch != "U":
            require(complete["fork_added_trained_transitions"] == budget, f"Fork budget mismatch: {branch}")
        require(complete["status"]["trained"] == 1033 + budget and complete["status"]["buffer_transitions"] == 0,
                f"Training cursor mismatch: {branch}")
        require(all(complete["freeze_audit"].values()) and read(run / "RESTORE_GATE.json")["passed"],
                f"Freeze/restore gate failed: {branch}")
        updates = [read(path) for path in sorted((run / "runtime/updates").glob("update_*.json"))]
        require(len(updates) == updates_count == summary["updates"], f"Update denominator mismatch: {branch}")
        trained, steps = 1033, parent["optimizer_steps"]
        for number, update in enumerate(updates, 15):
            trained += update["transitions"]
            steps += len(update["minibatches"])
            require(update["update"] == number and update["total_transitions"] == trained
                    and update["optimizer_steps"] == steps, f"Update cursor mismatch: {branch}")
            require(update["initial_ratio_max_error"] == 0 and all(update["frozen_audit"].values())
                    and update["actor_parameter_l2_change"] > 0 and update["critic_parameter_l2_change"] > 0,
                    f"PPO update audit failed: {branch}")
            require((run / f"runtime/updates/targets_{number:04d}.pt.gz").is_file(), "Missing fixed target data")
        require(trained == 1033 + budget and steps - parent["optimizer_steps"] == summary["new_optimizer_steps"],
                f"Update totals mismatch: {branch}")
        episodes = read(run / "episodes_summary.json")
        require(all(record["failure"] is None for record in episodes), "Failed navigation episode")
        require(sum(record["training"]["transitions"] for record in episodes if record["purpose"] == "train") == budget,
                f"Episode training denominator mismatch: {branch}")
        traces = read(run / "runtime/trace_summary.json.gz")
        require(len(traces) == len({record["transition_id"] for record in traces}), "Duplicate trace IDs")
        require(all(record["current_visual_tokens"] == 64 for record in traces), "Token budget mismatch")
        require({record["trace_path"] for record in traces} ==
                {path for episode in episodes for path in episode["training"]["trace_paths"]}, "Episode/trace mismatch")
        for count in ((0, 64) if branch == "U" else (0, 256, 512)):
            require((run / f"runtime/probes/added_{count:04d}.json").is_file(), "Missing milestone probe")
        if branch != "U":
            for subset, selection in (("independent", independent), ("historical", historical)):
                evaluations = read(run / f"evaluation_{subset}.json")
                require(len(evaluations) == len({record["episode_uid"] for record in evaluations}) == 32,
                        f"Evaluation denominator mismatch: {branch} {subset}")
                require({str(record["source_episode_id"]) for record in evaluations} ==
                        {str(record["source_episode_id"]) for record in selection}, "Evaluation selection mismatch")
                require(all(record["failure"] is None and not record["training"]["updated"] for record in evaluations),
                        "Evaluation failed or trained")
                for metric in ("SR", "SPL", "nDTW", "focus_mean"):
                    actual = sum(record[metric] for record in evaluations) / 32
                    require(math.isclose(actual, summary["subsets"][subset][metric], abs_tol=1e-12),
                            f"Evaluation summary mismatch: {branch} {metric}")
            require("GL::Context::current(): no current context" in (task / f"{directory}.log").read_text(),
                    "Missing cleanup failure evidence")
        for path in (run / "server.log", task / f"{directory}.log"):
            require(path.is_file() and path.stat().st_size > 0, f"Missing key training log: {path}")
        fresh_total += budget
    decision = read(task / "repair_decision.json")
    require(fresh_total == decision["total_fresh_training_including_new_C_control"] == 2112, "Fresh budget mismatch")
    require(decision["qualified_for_extension"] == [] and not decision["A_plus_B"]["train_A_plus_B_now"],
            "Unsupported expansion claim")
    require(all(not result["eligible"] for result in read(task / "extension_eligibility.json").values()),
            "Unexpected extension eligibility")
    novelty = read(task / "C_DECISION.json")
    require(not novelty["exploration_screen_passed"] and not novelty["automatic_extension"]
            and not novelty["Task233_configuration_approved"], "Unsupported C acceptance claim")
    for branch in ("C", "C0"):
        run = task / f"artifacts/branch_{branch}512"
        audit = read(run / "analysis/C_REWARD_AUDIT.json")
        require(audit["passed"] and audit["fresh_transitions_updated_once"] == 512
                and max(audit["max_abs_errors"].values()) < 1e-12, "Historical reward/GAE audit failed")
        require(int((run / "PROCESS_EXIT_CODE.txt").read_text()) == 134, "Cleanup exit code mismatch")
    baseline = read(task / "artifacts/baseline_independent/evaluation_independent.json")
    require(len(baseline) == 32 and {str(record["source_episode_id"]) for record in baseline} ==
            {str(record["source_episode_id"]) for record in independent}, "Baseline selection mismatch")
    require(all(record["failure"] is None and not record["training"]["updated"] for record in baseline),
            "Baseline failed or trained")
    threshold_rows = read(task / "artifacts/thresholds/threshold_results.json")
    require(len(threshold_rows) == 24 and all(record["failure"] is None for record in threshold_rows),
            "Threshold evaluation denominator mismatch")
    for group in read(task / "threshold_summary.json")["groups"]:
        if group["threshold"] == 0.5:
            selection = [read(root / f"Experiment/SCOPE/Stage2/Task23/artifacts/ppo4096/episodes/val_004_{index:03d}/result.json")
                         for index in sorted({row["manifest_index"] for row in threshold_rows})]
        else:
            selection = [row for row in threshold_rows if row["stop_threshold"] == group["threshold"]]
        require(len(selection) == len({row["episode_uid"] for row in selection}) == group["episodes"] == 12,
                "Threshold group denominator mismatch")
        require(all(math.isclose(sum(row[metric] for row in selection) / 12, group[metric], abs_tol=1e-12)
                    for metric in ("SR", "SPL", "nDTW", "focus_mean")), "Threshold summary mismatch")
    integrity = read(task / "SOURCE_INTEGRITY.json")
    require(not integrity["failures"] and not integrity["actor_source_failures"]
            and integrity["Task231_diagnosis_report_unchanged"], "Historical source integrity failure")
    with gzip.open(task / "FILES_MANIFEST.tsv.gz", "rt") as stream:
        originals = {record["path"]: record for record in csv.DictReader(stream, delimiter="\t")}
    placeholders = read(root / "docs/release/TASK232_CHECKPOINT_PLACEHOLDERS.json")
    require(len(placeholders) == len(set(placeholders)) == 83, "Placeholder count mismatch")
    require(set(placeholders) == {str(path.relative_to(root)) for path in task.rglob("*.placeholder.json")},
            "Placeholder inventory mismatch")
    for relative in placeholders:
        record = read(root / relative)
        require(record["uploaded"] is False and relative == record["checkpoint"] + ".placeholder.json",
                "Invalid checkpoint placeholder")
        original = originals[str((root / record["checkpoint"]).relative_to(task))]
        require(record["sha256"] == original["sha256"] and record["bytes"] == int(original["bytes"]),
                "Checkpoint placeholder hash mismatch")
    require(not list(task.rglob("*.pt")), "Task232 tensors must remain local")
    for record in read(root / "docs/release/TASK232_SOURCE_MAP.json"):
        if record["source"].startswith("Experiment/SCOPE/Stage2/Task232/"):
            relative = record["source"].split("Task232/", 1)[1]
            if relative in originals:
                require(record["source_sha256"] == originals[relative]["sha256"], "Original source map mismatch")
    return {"status": "PASS", "fresh_training_transitions": fresh_total, "branch_training_transitions": 512,
            "unchanged_gate_transitions": 64, "branch_evaluation_episodes": 256, "checkpoint_placeholders": 83,
            "extension_approved": False, "full_trace_replay": "requires local originals"}


def verify_task232(root=ROOT):
    archive = verify(root)
    return {"archive_integrity": archive["integrity"], **verify_evidence(root)}


if __name__ == "__main__":
    print(json.dumps(verify_task232(), indent=2))
