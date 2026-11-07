from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.audit import compare_nested
from Experiment.SCOPE.Stage2.data_guard import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--mode", choices=("restore", "forward"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "restore":
        first = torch.load(args.first, map_location="cpu", weights_only=False)
        second = torch.load(args.second, map_location="cpu", weights_only=False)
        for key in ("trainer", "cursor", "collected", "buffer_boundaries", "provenance", "episode_boundary"):
            compare_nested(first[key], second[key], key)
        checked = ["Actor", "Critic", "old snapshots", "optimizer", "RNG", "cursor", "buffer"]
    else:
        first = json.loads(args.first.read_text())
        second = json.loads(args.second.read_text())
        assert first["episode_uid"] == second["episode_uid"]
        first_paths, second_paths = first["training"]["trace_paths"], second["training"]["trace_paths"]
        assert len(first_paths) == len(second_paths)
        for first_path, second_path in zip(first_paths, second_paths):
            current = torch.load(first_path, map_location="cpu", weights_only=True)
            historical = torch.load(second_path, map_location="cpu", weights_only=True)
            for key in ("focus_count", "navigation_actions", "navigation_stop", "body_pose", "old_value"):
                compare_nested(current[key], historical[key], key)
            assert len(current["decisions"]) == len(historical["decisions"])
            for decision, reference in zip(current["decisions"], historical["decisions"]):
                for key in ("state", "action", "old_log_probability"):
                    compare_nested(decision[key], reference[key], key)
        checked = {"episode_uid": first["episode_uid"], "transitions": len(first_paths), "state_action_logprob_value": "bitwise_equal"}
    write_json(args.output, {"passed": True, "mode": args.mode, "checked": checked,
                             "first": str(args.first), "second": str(args.second)})


if __name__ == "__main__":
    main()
