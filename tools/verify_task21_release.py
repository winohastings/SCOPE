from __future__ import annotations

import json

from verify_release import ROOT, digest, verify


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_task21(root=ROOT):
    archive = verify(root)
    task = root / "Experiment/SCOPE/Stage2/Task21"
    release = task / "artifacts/release"
    validation = read(release / "validation.json")
    calibration = read(release / "calibration.json")
    captures = read(task / "artifacts/capture/manifest.json")
    require(validation["passed"], "Task21 validation failed")
    require(digest(release / "adapter.pt") == validation["checkpoint_sha256"], "Adapter hash mismatch")
    for flag in ("backbone_parameters_unchanged", "backbone_frozen_and_no_grad", "adapter_frozen", "checkpoint_reload_equal"):
        require(validation[flag], f"Freeze/reload gate failed: {flag}")
    require(calibration["metrics"] == validation["metrics"] and calibration["gate"] == validation["gate"],
            "Calibration and validation disagree")
    require(len(calibration["trace"]) == 120 and [entry["step"] for entry in calibration["trace"]] == list(range(1, 121)),
            "Calibration step denominator mismatch")
    for split, count in (("calibration", 96), ("diagnostic", 32)):
        metrics = validation["metrics"][split]
        require(metrics["samples"] == count, "Calibration/diagnostic sample denominator mismatch")
        for label in ("glance", "initial", "calibrated", "deployed_bfloat16"):
            require(metrics[label]["noncovered_max_abs"] == metrics[label]["noncovered_mse"] == 0,
                    "Noncovered feature preservation failed")
    training_ids = validation["calibration_episode_ids"]
    diagnostic_ids = validation["diagnostic_episode_ids"]
    require(len(training_ids) == 6 and len(diagnostic_ids) == 2 and not set(training_ids) & set(diagnostic_ids),
            "Calibration/diagnostic episode overlap")
    require([record["source_episode_id"] for record in captures] == training_ids + diagnostic_ids,
            "Capture manifest episode mismatch")
    checks = validation["no_focus_checks"]
    require(len(checks) == 6 and {(check["episode"], check["tick"]) for check in checks} ==
            {(episode, tick) for episode in diagnostic_ids for tick in range(3)}, "No-Focus replay denominator mismatch")
    for check in checks:
        require(check["action_max_abs"] == check["feature_max_abs"] == 0 and check["grids_equal"] and check["stop_equal"],
                "No-Focus equivalence failed")
    timing = validation["timing"]
    require([record["focus_count"] for record in timing] == list(range(5)), "Focus count coverage mismatch")
    for record in timing:
        require(record["shape"] == [4, 16, 2560] and record["navigation_current_tokens"] == 64,
                "Fixed current token contract failed")
        require(record["runtime_noncovered_max_abs"] == 0 and record["vision_encoded_postmerge_tokens"] == 64 + 64 * record["focus_count"],
                "Runtime coverage or encoder token contract failed")
        require(len(record["encoding_and_fusion_ms"]) == len(record["vision_only_ms"]) == 5,
                "Timing sample denominator mismatch")
    return {"archive_integrity": archive["integrity"], "calibration_samples": 96, "diagnostic_samples": 32,
            "calibration_steps": 120, "no_focus_checks": 6, "current_tokens": 64,
            "checkpoint_sha256": validation["checkpoint_sha256"], "status": "PASS"}


if __name__ == "__main__":
    print(json.dumps(verify_task21(), indent=2))
