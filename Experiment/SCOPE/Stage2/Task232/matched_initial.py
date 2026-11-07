from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.audit import compare_nested
from Experiment.SCOPE.Stage2.Task231.diagnose import interval
from Experiment.SCOPE.Stage2.Task232.metrics import roi_metrics, write_csv
from Experiment.SCOPE.Stage2.data_guard import write_json


def initial_traces(run):
    episodes = json.loads((run / "evaluation_independent.json").read_text())
    return {row["episode_uid"]: torch.load(row["training"]["trace_paths"][0], map_location="cpu", weights_only=True) for row in episodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/baseline_independent")
    parser.add_argument("--a", type=Path, default=ROOT / "artifacts/branch_A512")
    parser.add_argument("--b", type=Path, default=ROOT / "artifacts/branch_B512")
    args = parser.parse_args()
    baseline = initial_traces(args.baseline)
    rows = []
    for branch, run in (("BASE", args.baseline), ("A", args.a), ("B", args.b)):
        traces = initial_traces(run)
        assert set(traces) == set(baseline) and len(traces) == 32
        for episode, trace in sorted(traces.items()):
            reference = baseline[episode]
            valid = True
            try:
                for key in ("body_pose", "context_grids", "frame_count"):
                    compare_nested(trace[key], reference[key], key)
                compare_nested(trace["decisions"][0]["state"], reference["decisions"][0]["state"], "pre_focus")
            except AssertionError:
                valid = False
            selected = [(decision["action"]["camera"], decision["roi"]) for decision in trace["decisions"] if not decision["action"]["stop"]]
            progress = trace["teacher_for_reward_only"]["distance_before"] - trace["teacher_for_reward_only"]["distance_after"]
            base_progress = reference["teacher_for_reward_only"]["distance_before"] - reference["teacher_for_reward_only"]["distance_after"]
            rows.append({"branch": branch, "episode_uid": episode, "scene": trace["scene_id"], "pre_focus_bitwise_equal": valid,
                         **roi_metrics(selected), "progress": progress, "progress_delta_vs_base": progress - base_progress,
                         "action_L2_vs_base": float((trace["navigation_actions"] - reference["navigation_actions"]).norm())})
    write_csv(ROOT / "matched_initial_states.csv", rows)
    results = {}
    for branch in ("BASE", "A", "B"):
        group = [row for row in rows if row["branch"] == branch and row["pre_focus_bitwise_equal"]]
        results[branch] = {"valid_states": len(group), "coverage": float(np.mean([row["coverage"] for row in group])),
                           "repeated_roi_fraction": float(np.mean([row["repeated_roi_fraction"] for row in group])),
                           "progress_delta": interval([row["progress_delta_vs_base"] for row in group])}
    write_json(ROOT / "matched_initial_states.json", {"results": results,
               "scope": "same 32 independent episode resets; exact detached initial state/pose/context grids equality; actual online sequential Focus encoding, not a frozen-intermediate-state proxy; only first chunk, not full-episode causality"})


if __name__ == "__main__":
    main()
