from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task233.teacher_candidates import candidates, geometry, roi
from Experiment.SCOPE.Stage2.Task233.expert_scoring import score
from Experiment.SCOPE.Stage2.Task233.analyze import write_csv


def verify(output):
    output = Path(output)
    rows = [json.loads(line) for line in (output / "candidate_results.jsonl").read_text().splitlines()]
    complete = json.loads((output / "COMPLETE.json").read_text())
    manifest = json.loads((output / "DATA_MANIFEST.json").read_text())
    thresholds = json.loads((output / "FROZEN_THRESHOLDS.json").read_text())
    declared = {record["episode_uid"]: record for record in manifest["train"]}
    checks = {
        "train_only": all(row["split"] == "train" and row["episode_uid"] in declared for row in rows),
        "disjoint_trajectories": len({record["payload"]["trajectory_id"] for record in manifest["train"]}) == 36,
        "no_optimization": complete["audit"]["optimizer_updates"] == complete["audit"]["critic_training_samples"] == 0,
        "teacher_not_group": all(row["data_kind"] == "teacher_candidate" and row["ppo_eligible"] is False for row in rows),
        "token_budget": all(row["diagnostic"]["current_tokens"] == 64 for row in rows),
        "restoration": all(row["restoration_valid"] for row in rows),
        "coordinate_roundtrip": all(row["expert"]["coordinate_roundtrip_max_abs"] <= 1e-5 for row in rows),
        "trusted_assets": json.loads((output / "TRUSTED_ASSET_AUDIT.json").read_text())["passed"],
        "source_unchanged": json.loads((output / "SOURCE_INTEGRITY_AFTER.json").read_text())["unchanged"],
    }
    groups = defaultdict(dict)
    max_score_error = 0.
    max_latent_error = 0.
    conditionals, spatial = [], []
    for row in rows:
        anchor, variant = row["anchor_id"], row["variant"]
        if variant in groups[anchor]:
            raise ValueError("duplicate candidate branch")
        groups[anchor][variant] = row
        recomputed = score(row["actions"], row["navigation_stopped"], row["expert"])
        max_score_error = max(max_score_error, *(abs(row[key] - value) for key, value in recomputed.items()))
        selected = row["diagnostic"]["selected"]
        if any(record["roi"] != roi(record) for record in selected):
            raise ValueError("executed ROI differs from original latent mapping")
        if row["diagnostic"]["focus_count"] != len(selected) or not 0 <= len(selected) <= 4:
            raise ValueError("illegal Focus budget")
        rebuilt = geometry(selected)
        if any(row["diagnostic"][key] != value for key, value in rebuilt.items()):
            raise ValueError("ROI geometry audit mismatch")
        expected_count = 0 if variant == "glance_only" else row["j"] + (variant not in ("stop", "stop_repeat"))
        if len(selected) != expected_count:
            raise ValueError("matched microstep Focus budget violated")
        if variant.startswith("candidate_") and variant != "candidate_repeat":
            navigation_index = int(anchor.rsplit("_", 1)[-1])
            expected = candidates(233000 + row["ordinal"] * 10 + navigation_index)[int(variant.rsplit("_", 1)[-1])]
            if selected[-1]["camera"] != expected["camera"]:
                raise ValueError("candidate camera mismatch")
            max_latent_error = max(max_latent_error, float(np.max(np.abs(np.asarray(selected[-1]["latent"]) - expected["latent"]))))
        if variant == "stop":
            distributions = row["diagnostic"]["conditionals"]
            initial = np.asarray(distributions[0]["camera"])
            for distribution in distributions:
                probabilities = np.asarray(distribution["camera"])
                conditionals.append({"anchor_id": anchor, "partition": row["partition"], "scene": row["scene"],
                                     "state_source": row["state_source"], "j": distribution["j"],
                                     "top1_camera": int(probabilities.argmax()), "p_stop": distribution["p_stop"],
                                     "camera_front": probabilities[0], "camera_left": probabilities[1],
                                     "camera_right": probabilities[2], "camera_rear": probabilities[3],
                                     "L1_from_same_anchor_j0": float(np.abs(probabilities - initial).sum())})
        spatial.append({"anchor_id": anchor, "partition": row["partition"], "variant": variant,
                        "j": row["j"], "focus_count": len(selected), **rebuilt,
                        "navigation_stopped": row["navigation_stopped"], "terminal_success": row["terminal_success"]})
    checks["all_eight_candidates_executed"] = all(all(f"candidate_{index}" in group for index in range(8)) for group in groups.values())
    checks["scoring_replay"] = max_score_error <= 1e-12
    checks["original_latent_preserved_float32"] = max_latent_error <= 1e-6
    checks["prediction_accounting"] = complete["audit"]["model_predictions"] == len(rows) <= 1008
    checks["focus_call_accounting"] = complete["audit"]["focus_encoder_calls"] == sum(row["diagnostic"]["focus_count"] for row in rows)
    calibration = [row for row in rows if row["partition"] == "calibration"]
    from Experiment.SCOPE.Stage2.Task233.gate0 import digest
    checks["calibration_frozen_evidence"] = digest(calibration) == thresholds["calibration_rows_sha256"]
    write_csv(output / "camera_conditionals.csv", conditionals)
    write_csv(output / "spatial_controls.csv", spatial)
    audit = {"passed": all(checks.values()), "checks": checks, "max_score_error": max_score_error,
             "max_latent_float32_error": max_latent_error, "verification_source_sha256": sha256(__file__),
             "scope": "independent arithmetic/geometry/accounting audit; does not change preregistered Gate0",
             "camera_scope": "real fixed-prefix incremental histories; not autonomous four-Focus learned-policy performance"}
    write_json(output / "VERIFICATION.json", audit)
    if not audit["passed"]:
        raise AssertionError(audit)
    print(json.dumps(audit), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
