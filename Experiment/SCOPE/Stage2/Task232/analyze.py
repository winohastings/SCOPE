from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task22.actor import ActorState
from Experiment.SCOPE.Stage2.Task23.critic import NavigationCritic
from Experiment.SCOPE.Stage2.Task23.learning import navigation_reward
from Experiment.SCOPE.Stage2.Task232.metrics import roi_metrics, write_csv
from Experiment.SCOPE.Stage2.Task232.novelty import coverage_novelty
from Experiment.SCOPE.Stage2.data_guard import write_json, sha256


def critic_metrics(episodes, critic, cost, beta=0.):
    records = []
    with torch.no_grad():
        for episode in episodes:
            traces = [torch.load(path, map_location="cpu", weights_only=True) for path in episode["training"]["trace_paths"]]
            truncated = traces[-1]["truncated"]
            bootstrap = float(critic(ActorState.unpack(traces[-1]["next_state"], "cpu"))) if truncated else 0.
            running = {"branch": bootstrap, "original": bootstrap}
            for trace in reversed(traces):
                prediction = float(critic(ActorState.unpack(trace["decisions"][0]["state"], "cpu")))
                assert abs(prediction - trace["old_value"]) < 1e-4
                original = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                             terminated=trace["terminated"], truncated=trace["truncated"])
                reward = original + (.01 - cost) * trace["focus_count"] + coverage_novelty(trace["decisions"], beta)["reward"]
                assert abs(reward - trace["navigation_reward"]) < 1e-8
                for definition, current in (("branch", reward), ("original", original)):
                    running[definition] = current + .99 * running[definition]
                    records.append({"episode_uid": episode["episode_uid"], "scene": episode["scene_id"], "transition_id": trace["transition_id"],
                                    "definition": definition, "episode_truncated": truncated,
                                    "prediction": prediction, "return": running[definition]})
    summaries = {}
    for definition in ("branch", "original"):
        for subset in ("all_bootstrapped", "terminal_only"):
            group = [row for row in records if row["definition"] == definition and (subset == "all_bootstrapped" or not row["episode_truncated"])]
            prediction = np.asarray([row["prediction"] for row in group])
            returns = np.asarray([row["return"] for row in group])
            if not group:
                summaries[f"{definition}_{subset}"] = {"transitions": 0, "EV": None, "MSE": None, "prediction_mean": None, "return_mean": None}
                continue
            summaries[f"{definition}_{subset}"] = {"transitions": len(group), "EV": float(1 - np.var(returns - prediction) / max(np.var(returns), 1e-8)),
                                                   "MSE": float(np.mean((returns - prediction) ** 2)), "prediction_mean": float(prediction.mean()),
                                                   "return_mean": float(returns.mean())}
    return records, summaries


def behavior(episodes):
    rows = []
    camera_counts = Counter()
    camera_probabilities = defaultdict(list)
    for episode in episodes:
        for path in episode["training"]["trace_paths"]:
            trace = torch.load(path, map_location="cpu", weights_only=True)
            selected = [(decision["action"]["camera"], decision["roi"]) for decision in trace["decisions"] if not decision["action"]["stop"]]
            for decision in trace["decisions"]:
                if not decision["action"]["stop"]:
                    camera_counts[decision["action"]["camera"]] += 1
                    camera_probabilities[decision["focus_count_before"]].append(decision["old_distribution"]["camera_probabilities"])
            spatial = roi_metrics(selected)
            assert abs(spatial["coverage"] - float(trace["final_state"]["coverage"].mean())) < 1e-7
            rows.append({"episode_uid": episode["episode_uid"], "scene": episode["scene_id"], "trace_path": path,
                         "purpose": trace["purpose"], "policy_version": trace["policy_version"], **spatial,
                         "p_stop": float(trace["decisions"][0]["old_distribution"]["stop_probability"]),
                         "reward": trace["navigation_reward"], "model_ms": trace["elapsed_model_ms"], "perception_ms": trace["perception_ms"],
                         "collision": trace["teacher_for_reward_only"]["collision"], "terminated": trace["terminated"], "truncated": trace["truncated"]})
    counts = Counter(row["focus_count"] for row in rows)
    four = [row for row in rows if row["focus_count"] == 4]
    result = {"transitions": len(rows), "focus_distribution": {str(count): counts[count] for count in range(5)},
              "camera_histogram": {str(camera): camera_counts[camera] for camera in range(4)},
              "camera_probability_mean_by_focus_index": {str(index): np.mean(probabilities, axis=0).tolist() for index, probabilities in camera_probabilities.items()},
              "focus_mean": float(np.mean([row["focus_count"] for row in rows])), "zero_focus_fraction": counts[0] / len(rows),
              "coverage_mean": float(np.mean([row["coverage"] for row in rows])), "four_focus_samples": len(four),
              "four_focus_coverage": float(np.mean([row["coverage"] for row in four])) if four else None,
              "four_focus_repeat_fraction": float(np.mean([row["repeated_roi_fraction"] for row in four])) if four else None,
              "four_focus_unique_cameras": float(np.mean([row["unique_cameras"] for row in four])) if four else None,
              "model_seconds": sum(row["model_ms"] for row in rows) / 1000,
              "perception_seconds": sum(row["perception_ms"] for row in rows) / 1000}
    return rows, result


def energy(path):
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                groups[row["index"].strip()].append((datetime.datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp(),
                                                     float(row["power_draw"]), float(row["memory_used"])))
            except (ValueError, KeyError):
                continue
    return {device: {"samples": len(samples), "peak_memory_MiB": max(row[2] for row in samples),
                     "sampled_system_Wh": sum((second[0] - first[0]) * (first[1] + second[1]) / 2 / 3600 for first, second in zip(samples, samples[1:]))}
            for device, samples in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    args.run = args.run.resolve()
    if not args.run.is_relative_to(ROOT):
        parser.error("Task232-only runs")
    if not args.partial and not (args.run / "COMPLETE.json").exists():
        parser.error("run incomplete")
    torch.set_num_threads(2)
    config = json.loads((args.run / "config.json").read_text())
    episodes = json.loads((args.run / "episodes_summary.json").read_text())
    output = args.run / ("analysis_partial" if args.partial else "analysis")
    output.mkdir(exist_ok=True)
    checkpoints = sorted((args.run / "runtime/checkpoints").glob("added_*.pt"))
    checkpoint = checkpoints[-1]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    critic = NavigationCritic().eval()
    critic.load_state_dict(payload["trainer"]["critic"])
    subsets = {}
    training = [row for row in episodes if row["purpose"] == "train"]
    if training:
        rows, subsets["train"] = behavior(training)
        for row in rows:
            assert Path(row["trace_path"]).resolve().is_relative_to(args.run / "runtime/traces")
        write_csv(output / "train_behavior.csv", rows)
        subsets["train"]["reward_components"] = {key: sum(episode["training"]["reward_components"][key] for episode in training)
                                                for key in training[0]["training"]["reward_components"]}
        subsets["train"]["episodes"] = len(training)
        subsets["train"]["wall_seconds"] = sum(row["wall_seconds"] for row in training)
    for subset in ("historical", "independent"):
        path = args.run / f"evaluation_{subset}.json"
        if not path.exists():
            continue
        current = json.loads(path.read_text())
        rows, result = behavior(current)
        result.update(episodes=len(current), **{key: float(np.mean([row[key] for row in current])) for key in ("SR", "SPL", "nDTW", "oracle_success", "reward_sum")})
        predictions, diagnostics = critic_metrics(current, critic, config["focus_cost"], config.get("coverage_beta", 0.))
        result["critic"] = diagnostics
        write_csv(output / f"{subset}_critic.csv", predictions)
        write_csv(output / f"{subset}_behavior.csv", rows)
        subsets[subset] = result
    updates = [json.loads(path.read_text()) for path in sorted((args.run / "runtime/updates").glob("update_*.json"))]
    report = {"branch": config["task232_branch"], "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
              "added_trained_transitions": payload["trainer"]["transitions_seen"] - 1033, "updates": len(updates),
              "update_batch_sizes": [update["transitions"] for update in updates],
              "in_sample_critic_EV_by_update": [update["explained_variance"] for update in updates],
              "optimizer_hyperparameters": {key: config[key] for key in ("actor_learning_rate", "critic_learning_rate", "rollout_minimum", "focus_cost")},
              "new_optimizer_steps": payload["trainer"]["optimizer_steps"] - 146,
              "ratio_max_error": max((update["initial_ratio_max_error"] for update in updates), default=0),
              "update_seconds": sum(update["seconds"] for update in updates), "subsets": subsets,
              "system_gpu_samples": energy(args.run / "gpu.csv"),
              "energy_caveat": "whole GPU power during run; concurrent model/renderer/desktop workloads included, NOT exclusive per-branch energy",
              "critic_caveat": "deterministic held-out policy differs from training sampling; truncations bootstrap own Critic, terminal-only reported separately",
              "partial": args.partial}
    if (args.run / "COMPLETE.json").exists():
        completed = json.loads((args.run / "COMPLETE.json").read_text())
        report["total_run_wall_seconds_including_startup_probes_evaluation"] = completed["wall_seconds"]
        report["fresh_training_transitions_this_run"] = completed["new_training_transitions"]
    write_json(output / "summary.json", report)
    print(json.dumps({"branch": config["task232_branch"], "added_trained": report["added_trained_transitions"],
                      "subsets": {key: {name: value for name, value in subset.items() if name in ("SR", "SPL", "focus_mean", "coverage_mean", "four_focus_coverage")}
                                  for key, subset in subsets.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
