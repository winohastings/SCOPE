from __future__ import annotations

import json
from pathlib import Path

from verify_release import ROOT, historical_source, verify


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_stage2(root=ROOT):
    archive = verify(root)
    stage = root / "Experiment/SCOPE/Stage2"
    artifacts = stage / "artifacts"
    final = read(artifacts / "FINAL_VALIDATION.json")
    complete = read(artifacts / "release/COMPLETE.json")
    require(final["passed"] and complete["passed"], "Stage2 completion gate failed")
    require(not complete["full_training_started"] and not complete["heldout_evaluation_started"],
            "Unexpected full training or heldout evaluation")
    for relative, expected in read(artifacts / "FINAL_SOURCE_HASHES.json").items():
        historical_source(root, relative, expected)
    records = []
    for phase in ("train_A", "train_B", "train_C"):
        records.extend(read(artifacts / "release" / phase / "results.json"))
    primary = sum(record["training"]["transitions"] for record in records)
    require(primary == final["primary_real_training_transitions"] == 24, "Primary training denominator mismatch")
    resumed = read(artifacts / "resume_train_check/results.json")
    require(len(resumed) == 1 and resumed[0]["source_episode_id"] == final["resume_episode_id"] == "3",
            "Resume cursor episode mismatch")
    transitions = primary + sum(record["training"]["transitions"] for record in resumed)
    require(transitions == final["total_final_validation_real_training_transitions"] == 26,
            "Final training denominator mismatch")
    for record in records + resumed:
        update = record["training"]
        require(record["failure"] is None and update["updated"] and update["backbone_gradients_none"],
                "Failed update or unfrozen backbone")
        require(update["statistics"][0]["ratio_mean"] == 1.0 and update["actor_parameter_l2_change"] > 0,
                "PPO ratio or parameter update evidence mismatch")
    require(len(records) == final["primary_ppo_updates"] == 4 and len(resumed) == final["supplemental_resume_updates"] == 1,
            "PPO update denominator mismatch")
    diagnostics = [record for purpose in ("train", "dev") for record in
                   read(artifacts / "release" / f"diagnostics_{purpose}" / "diagnostics.json")]
    require(len(diagnostics) == final["diagnostic_branch_comparisons"] == 54, "Diagnostic denominator mismatch")
    require(final["matched_history_native_scatter_action_max_abs"] == 0 and final["native_cached_feature_max_abs"] == 0,
            "Cache or native scatter equivalence failed")
    metrics = ("source_episode_id", "success", "oracle_success", "final_distance", "path_length", "steps", "failure", "termination_reason", "nDTW")
    after = read(artifacts / "release/dev_after/results.json")
    restored = read(artifacts / "release/dev_restored/results.json")
    require(len(after) == len(restored) == 2, "Unexpected fixed development denominator")
    require([{key: record[key] for key in metrics} for record in after] ==
            [{key: record[key] for key in metrics} for record in restored], "Restored development metrics differ")
    return {"archive_integrity": archive["integrity"], "diagnostics": len(diagnostics),
            "real_training_transitions": transitions, "ppo_updates": len(records) + len(resumed),
            "restored_development_metrics": "EXACT", "status": "PASS"}


if __name__ == "__main__":
    print(json.dumps(verify_stage2(), indent=2))
