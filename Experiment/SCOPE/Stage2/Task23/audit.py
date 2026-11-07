from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def compare_nested(first, second, name="root"):
    if isinstance(first, torch.Tensor):
        if not torch.equal(first.cpu(), second.cpu()):
            raise AssertionError(f"restored tensor differs: {name}")
    elif isinstance(first, np.ndarray):
        if not np.array_equal(first, second):
            raise AssertionError(f"restored numpy RNG differs: {name}")
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise AssertionError(f"restored keys differ: {name}")
        for key in first:
            compare_nested(first[key], second[key], f"{name}.{key}")
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError(f"restored sequence length differs: {name}")
        for index, (left, right) in enumerate(zip(first, second)):
            compare_nested(left, right, f"{name}[{index}]")
    elif first != second:
        raise AssertionError(f"restored scalar differs: {name}: {first} vs {second}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--restored", type=Path)
    sources.add_argument("--replay-before", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = torch.load(args.compare, map_location="cpu", weights_only=False)
    if args.replay_before is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        from Experiment.SCOPE.Stage2.Task22.actor import SpatialQueryActor
        from Experiment.SCOPE.Stage2.Task22.trajectory import load_episode_traces
        from Experiment.SCOPE.Stage2.Task23.learning import PPOTrainer

        second = torch.load(args.replay_before, map_location="cpu", weights_only=False)
        trainer = PPOTrainer(SpatialQueryActor(), second["trainer"]["config"], "cuda")
        trainer.restore(second["trainer"])
        transitions = []
        for boundary in second["buffer_boundaries"]:
            transitions.extend(trainer.prepare_episode(load_episode_traces(boundary)))
        update, _ = trainer.update(transitions)
        second["trainer"] = trainer.payload()
        second["buffer_boundaries"] = []
    else:
        second = torch.load(args.restored, map_location="cpu", weights_only=False)
    for key in ("trainer", "cursor", "collected", "buffer_boundaries", "provenance", "episode_boundary"):
        compare_nested(first[key], second[key], key)
    report = {"passed": True, "bitwise_equal": True, "actor_critic_optimizer_rng_cursor_equal": True,
              "first": str(args.compare), "restored": str(args.restored or args.replay_before),
              "fresh_process_replay": args.replay_before is not None,
              "updates": first["trainer"]["updates"], "optimizer_steps": first["trainer"]["optimizer_steps"],
              "trained_transitions": first["trainer"]["transitions_seen"]}
    if args.replay_before is not None:
        report["update"] = update
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
