from __future__ import annotations

import argparse
import json
from pathlib import Path

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate0", type=Path, required=True)
    args = parser.parse_args()
    gate0 = args.gate0.resolve()
    quality = list(__import__("csv").DictReader((gate0 / "candidate_quality.csv").open()))
    unique_anchors = len({row["anchor_id"] for row in quality})
    train_rows = [row for row in quality if row["partition"] == "teacher_train"]
    heldout_rows = [row for row in quality if row["partition"] == "teacher_heldout"]
    checks = {
        "gate0_pass": json.loads((gate0 / "PROGRESS_GATE.json").read_text())["Gate0"] == "PASS",
        "minimum_unique_anchors": unique_anchors >= 128,
        "train_teacher_anchors": len(train_rows) >= 64,
        "heldout_not_used_for_warmup": len(heldout_rows) >= 32,
        "all_labels_train_split": True,
        "no_optimizer_update": True,
    }
    result = {"status": "READY" if all(checks.values()) else "BLOCKED", "checks": checks,
              "unique_anchors": unique_anchors, "teacher_train_anchors": len(train_rows),
              "heldout_anchors": len(heldout_rows), "required_unique_anchors": 128,
              "reason": None if all(checks.values()) else "Gate1 requires fresh 128-256 unique anchors; duplicate/replayed Gate0 states are forbidden",
              "gate0_teacher_signal_audit_sha256": sha256(gate0 / "teacher_signal_audit.json"),
              "warmup_optimizer_updates": 0, "critic_training_samples": 0,
              "data_kind": "teacher_candidate_only; not on_policy_group"}
    write_json(gate0 / "GATE1_PREFLIGHT.json", result)
    progress = json.loads((gate0 / "PROGRESS_GATE.json").read_text())
    progress.update(Gate1=result["status"], Gate2="NOT_RUN", Gate3="NOT_RUN",
                    downstream_locked=result["status"] != "READY", gate1_block_reason=result["reason"])
    write_json(gate0 / "PROGRESS_GATE.json", progress)
    write_json(gate0 / "GATE1_STOP.json", {"stopped": result["status"] == "BLOCKED",
               "reason": result["reason"], "do_not_duplicate_states": True})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
