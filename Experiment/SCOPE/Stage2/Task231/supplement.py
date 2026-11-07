from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import replay_decisions
from Experiment.SCOPE.Stage2.Task23.critic import NavigationCritic
from Experiment.SCOPE.Stage2.Task23.learning import sequence_objective
from Experiment.SCOPE.Stage2.Task231.diagnose import CHECKPOINTS, load, write_csv, write_json


def gradient_components(args):
    samples = list(csv.DictReader((args.offline / "head_gradients.csv").open()))
    config = json.loads((args.source / "config.json").read_text())
    policy_config = config | {"entropy_weights": {name: 0. for name in config["entropy_weights"]}}
    actor = SpatialQueryActor().to(args.device).eval()
    current_update = None
    rows = []
    for sample in samples:
        update = int(sample["update"])
        if update != current_update:
            actor.load_state_dict(load(args.source / f"runtime/snapshots/actor_{update:04d}.pt")["actor"])
            current_update = update
        decisions = replay_decisions(load(sample["trace_path"]), args.device)
        advantage = torch.tensor(float(sample["advantage"]), device=args.device)
        for component in ("policy", "entropy"):
            actor.zero_grad(set_to_none=True)
            if component == "policy":
                loss = sequence_objective(actor, decisions, advantage, policy_config)[0]
            else:
                loss = sequence_objective(actor, decisions, advantage.new_zeros(()), config)[0]
            loss.backward()
            row = {"update": update, "trace_path": sample["trace_path"], "focus_count": int(sample["focus_count"]),
                   "decision_count": len(decisions), "component": component, "advantage": float(advantage), "loss": float(loss.detach()),
                   "stop_output_bias_gradient": float(actor.stop_head[-1].bias.grad)}
            for name, module in (("stop", actor.stop_head), ("camera", actor.camera_head), ("position", actor.position_head)):
                row[f"{name}_gradient_norm"] = math.sqrt(sum(float(parameter.grad.square().sum()) for parameter in module.parameters() if parameter.grad is not None))
            row["log_std_gradient_norm"] = float(actor.log_std.grad.norm()) if actor.log_std.grad is not None else 0.
            rows.append(row)
    write_csv(args.output / "head_gradient_components.csv", rows)
    write_json(args.output / "gradient_components.json", {"samples": len(samples), "component_rows": len(rows),
        "meaning": "post-update retained-state gradient diagnostics; policy surrogate and entropy regularization separated; no optimizer steps",
        "signed_stop_bias": "negative loss gradient would increase STOP output bias under plain gradient descent; Adam and shared trunk prevent treating this as an exact historical update",
        "warning": "STOP-only samples can have camera/position gradients from entropy regularization, not from chosen camera/position action credit"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Task23/artifacts/ppo4096")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/supplement")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gradients-only", action="store_true")
    parser.add_argument("--offline", type=Path, default=ROOT / "artifacts/offline")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        parser.error("Task231-only output")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    if args.gradients_only:
        gradient_components(args)
        return
    critic = NavigationCritic().to(args.device).eval()
    rows, reports = [], []
    with torch.no_grad():
        for round_index, (update, transitions) in enumerate(CHECKPOINTS):
            payload = torch.load(args.source / f"runtime/checkpoints/update_{update:04d}.pt", map_location="cpu", weights_only=False)
            critic.load_state_dict(payload["trainer"]["critic"])
            current = []
            for result_path in sorted((args.source / "episodes").glob(f"val_{round_index:03d}_*/result.json")):
                episode = json.loads(result_path.read_text())
                traces = [load(path) for path in episode["training"]["trace_paths"]]
                carry = 0. if traces[-1]["terminated"] else float(critic(ActorState.unpack(traces[-1]["next_state"], args.device)))
                episode_rows = []
                for trace in reversed(traces):
                    carry = trace["navigation_reward"] + .99 * carry
                    prediction = float(critic(ActorState.unpack(trace["decisions"][0]["state"], args.device)))
                    episode_rows.append({"transitions": transitions, "episode_uid": episode["episode_uid"], "scene": episode["scene_id"],
                                         "trace_id": trace["transition_id"], "prediction": prediction,
                                         "discounted_return": carry, "episode_truncated": traces[-1]["truncated"]})
                current.extend(reversed(episode_rows))
            rows.extend(current)
            for label, subset in (("all_bootstrapped", current), ("terminal_episodes_only", [row for row in current if not row["episode_truncated"]])):
                targets = np.asarray([row["discounted_return"] for row in subset])
                predictions = np.asarray([row["prediction"] for row in subset])
                reports.append({"transitions": transitions, "subset": label, "states": len(subset),
                                "explained_variance": float(1 - np.var(targets - predictions) / max(np.var(targets), 1e-8)),
                                "mse": float(np.mean((targets - predictions) ** 2)), "target_mean": float(targets.mean()),
                                "prediction_mean": float(predictions.mean())})
    training = []
    for result_path in sorted((args.source / "episodes").glob("train_*/result.json")):
        episode = json.loads(result_path.read_text())
        for path in episode["training"]["trace_paths"]:
            trace = load(path)
            training.append({"version": trace["policy_version"], "focus": trace["focus_count"],
                             "p_stop": float(trace["decisions"][0]["old_distribution"]["stop_probability"])})
    phases = []
    for begin, end in ((0, 14), (14, 28), (28, 43), (43, 56)):
        subset = [row for row in training if begin <= row["version"] < end]
        counts = Counter(row["focus"] for row in subset)
        phases.append({"update_range": [begin, end], "transitions": len(subset), "focus_histogram": dict(counts),
                       "mean_focus": float(np.mean([row["focus"] for row in subset])),
                       "p_stop_quantiles": np.quantile([row["p_stop"] for row in subset], [.1, .5, .9]).tolist()})
    write_csv(args.output / "heldout_value_predictions.csv", rows)
    write_json(args.output / "supplement.json", {"heldout_critic": reports, "training_sampling": phases,
               "critic_caveat": "validation data never optimized; deterministic policy differs from training sampling. MC targets bootstrap truncations; terminal-only results avoid bootstrap circularity. States are correlated within episodes.",
               "cost_arithmetic": {"reward_saved_if_same_navigation_and_4_to_0_focus": .04,
                                   "extra_progress_meters_to_offset_four_focus_cost_in_unclipped_region": .6,
                                   "progress_meters_to_offset_four_focus_plus_step_cost": .75},
               "causal_caveat": "Fixed-trajectory cost arithmetic is not evidence of navigation improvement after removing the cost; that requires newly sampled training."})


if __name__ == "__main__":
    main()
