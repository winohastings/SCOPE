from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import html
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from Experiment.SCOPE.Stage2.Task23.learning import navigation_reward


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def svg_curve(path, rows, keys, title):
    width, height = 900, 360
    colors = ("#1565c0", "#c62828", "#2e7d32", "#8e24aa")
    values = [float(row[key]) for row in rows for key in keys]
    if not values:
        return
    lower, upper = min(values), max(values)
    span = max(upper - lower, 1e-8)
    xmax = max(float(row["transitions"]) for row in rows) or 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="35" y="25" font-size="18">{html.escape(title)}</text>',
             '<path d="M65 50V300H870" fill="none" stroke="black"/>',
             f'<text x="5" y="65" font-size="12">{upper:.4g}</text>',
             f'<text x="5" y="300" font-size="12">{lower:.4g}</text>',
             f'<text x="650" y="350" font-size="12">main transitions: {int(xmax)}</text>']
    for index, key in enumerate(keys):
        points = " ".join(f"{65 + 805 * row['transitions'] / xmax:.2f},{300 - 240 * (row[key] - lower) / span:.2f}" for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="2"/>')
        parts.append(f'<text x="{70 + index * 200}" y="325" fill="{colors[index]}" font-size="12">{html.escape(key)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--verify-traces", action="store_true")
    args = parser.parse_args()
    runtime = args.run / "runtime"
    updates = [json.loads(path.read_text()) for path in sorted((runtime / "updates").glob("update_*.json"))]
    episodes = json.loads((args.run / "episodes_summary.json").read_text())
    validation = json.loads((runtime / "validation_history.json").read_text()) if (runtime / "validation_history.json").exists() else []
    checkpoint = torch.load(runtime / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    report_dir = args.run / "summary"
    report_dir.mkdir(exist_ok=True)
    rows = []
    for update in updates:
        minibatches = update["minibatches"]
        rows.append({"transitions": update["total_transitions"], "update": update["update"],
                     "actor_loss": float(np.mean([row["actor_loss"] for row in minibatches])),
                     "critic_loss": float(np.mean([row["critic_loss"] for row in minibatches])),
                     "KL": float(np.mean([row["approximate_kl"] for row in minibatches])),
                     "clip_fraction": float(np.mean([row["clip_fraction"] for row in minibatches])),
                     "STOP_entropy": float(np.mean([row["entropy"]["stop"] for row in minibatches])),
                     "camera_entropy": float(np.mean([row["entropy"]["camera"] for row in minibatches])),
                     "position_entropy": float(np.mean([row["entropy"]["position_latent"] for row in minibatches])),
                     "actor_change": update["actor_parameter_l2_change"], "critic_change": update["critic_parameter_l2_change"],
                     "initial_ratio_error": update["initial_ratio_max_error"], "explained_variance": update["explained_variance"],
                     "update_seconds": update["seconds"], "kl_early_stop": update["kl_early_stop"]})
    if rows:
        with (report_dir / "training.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        svg_curve(report_dir / "loss.svg", rows, ["actor_loss", "critic_loss"], "PPO losses")
        svg_curve(report_dir / "entropy.svg", rows, ["STOP_entropy", "camera_entropy", "position_entropy"], "Policy entropies (nats)")
        svg_curve(report_dir / "kl.svg", rows, ["KL", "clip_fraction"], "PPO KL and clipping")
    if validation:
        svg_curve(report_dir / "val_seen.svg", validation, ["SR", "SPL"], "Fixed 32-episode val_seen subset")
    validation_focus = []
    validation_rows = [row for row in episodes if row["purpose"] == "val_seen" and not row.get("failure")]
    offset = 0
    for record in validation:
        selected = validation_rows[offset:offset + record["episodes"]]
        offset += record["episodes"]
        histogram = Counter()
        for row in selected:
            histogram.update({int(key): value for key, value in row["training"]["episode_summary"]["focus_histogram"].items()})
        total = sum(histogram.values())
        validation_focus.append({"round": record["round"], "transitions": record["transitions"],
                                 "navigation_transitions": total, "focus_histogram": dict(histogram),
                                 "zero_focus_fraction": histogram[0] / max(1, total),
                                 "transition_weighted_focus_mean": sum(key * value for key, value in histogram.items()) / max(1, total)})
    grouped = {}
    for split in ("train", "val_seen"):
        selected = [row for row in episodes if row["purpose"] == split and row.get("failure") is None
                    and row.get("training") is not None]
        if not selected:
            continue
        focus, cameras = Counter(), Counter()
        for row in selected:
            focus.update({int(key): value for key, value in row["training"]["episode_summary"]["focus_histogram"].items()})
            cameras.update({int(key): value for key, value in row["training"]["episode_summary"]["camera_histogram"].items()})
        grouped[split] = {"episodes": len(selected), "transitions": sum(row["training"]["transitions"] for row in selected),
                          "SR": float(np.mean([row["SR"] for row in selected])), "SPL": float(np.mean([row["SPL"] for row in selected])),
                          "mean_episode_reward": float(np.mean([row["reward_sum"] for row in selected])),
                          "focus_histogram": dict(focus), "camera_histogram": dict(cameras),
                          "focus_mean": sum(count * total for count, total in focus.items()) / sum(focus.values()),
                          "wall_seconds": sum(row["wall_seconds"] for row in selected),
                          "model_seconds": sum(row["training"]["episode_summary"]["model_ms"] for row in selected) / 1000}
    energies, max_memory, last_sample = Counter(), Counter(), {}
    with (args.run / "gpu.csv").open() as stream:
        for sample in csv.DictReader(stream):
            timestamp = datetime.strptime(sample["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            gpu = sample["index"].strip()
            power = float(sample["power_draw"])
            max_memory[gpu] = max(max_memory[gpu], float(sample["memory_used"]))
            if gpu in last_sample:
                previous_time, previous_power = last_sample[gpu]
                energies[gpu] += (timestamp - previous_time) * (power + previous_power) / 2 / 3600
            last_sample[gpu] = timestamp, power
    trace_audit = {"checked": False}
    if args.verify_traces:
        counts = Counter()
        identities = set()
        components = Counter()
        train_focus_by_version = {}
        boundaries = Counter()
        for path in sorted((runtime / "traces").glob("*.pt")):
            trace = torch.load(path, weights_only=True)
            if trace["transition_id"] in identities or trace["current_visual_tokens"] != 64 or trace["feedback"] is None:
                raise AssertionError("invalid, duplicate, or incomplete transition")
            identities.add(trace["transition_id"])
            counts[trace["purpose"]] += 1
            reward = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                       terminated=trace["terminated"], truncated=trace["truncated"])
            if abs(reward - trace["navigation_reward"]) > 1e-9:
                raise AssertionError("saved navigation reward differs from formula")
            if trace["purpose"] == "train":
                train_focus_by_version.setdefault(trace["policy_version"], Counter())[trace["focus_count"]] += 1
                boundaries["terminated"] += int(trace["terminated"])
                boundaries["truncated"] += int(trace["truncated"])
                if trace["truncated"] and (trace["next_state"] is None or not math.isfinite(trace["old_next_value"])):
                    raise AssertionError("truncation bootstrap missing or nonfinite")
                teacher = trace["teacher_for_reward_only"]
                components["progress_sum"] += .2 * float(np.clip((teacher["distance_before"] - teacher["distance_after"]) / 3, -1, 1))
                components["focus_cost_sum"] -= .01 * trace["focus_count"]
                components["step_cost_sum"] -= .01
                components["terminal_bonus_sum"] += (2. if teacher["success"] else -.5) if trace["terminated"] else 0.
        trace_audit = {"checked": True, "transitions": dict(counts), "reward_components_train": dict(components),
                       "train_focus_by_policy_version": {str(key): dict(value) for key, value in sorted(train_focus_by_version.items())},
                       "train_boundaries": dict(boundaries)}
    complete = json.loads((args.run / "COMPLETE.json").read_text()) if (args.run / "COMPLETE.json").exists() else None
    findings = []
    if validation and (validation[-1]["SR"], validation[-1]["SPL"]) <= (validation[0]["SR"], validation[0]["SPL"]):
        findings.append("final fixed-subset SR/SPL did not improve over initialization; do not claim navigation benefit")
    if grouped.get("train", {}).get("focus_histogram", {}).get(0, 0) / max(1, grouped.get("train", {}).get("transitions", 0)) > .9:
        findings.append("more than 90% training states used zero Focus; inspect early-STOP collapse")
    if validation_focus and validation_focus[-1]["zero_focus_fraction"] > .9:
        findings.append("final deterministic validation uses zero Focus in over 90% of states: early-STOP collapse; do not claim useful adaptive gaze")
    if not complete:
        findings.append("formal budget is incomplete; latest checkpoint is resumable")
    if trace_audit["checked"]:
        for split in ("train", "val_seen"):
            if trace_audit["transitions"].get(split, 0) != grouped.get(split, {}).get("transitions", 0):
                raise AssertionError("trace count differs from completed episode summaries")
        if grouped.get("train", {}).get("transitions", 0) != checkpoint["collected"]:
            raise AssertionError("single-run audit requires all original segments when resuming")
    summary = {"complete": complete is not None, "status": complete, "updates": len(updates),
               "stage": complete.get("stage") if complete else None,
               "failed_episodes": [row for row in episodes if row.get("failure")],
               "optimizer_steps": checkpoint["trainer"]["optimizer_steps"], "trained_transitions": checkpoint["trainer"]["transitions_seen"],
               "collected_transitions": checkpoint["collected"], "best_val_seen": checkpoint["best"],
               "initial_validation": validation[0] if validation else None, "final_validation": validation[-1] if validation else None,
               "validation_history": validation, "validation_focus": validation_focus, "episodes": grouped, "trace_audit": trace_audit,
               "energy_wh_by_gpu": dict(energies), "peak_memory_mib_by_gpu": dict(max_memory),
               "max_initial_ratio_error": max((row["initial_ratio_error"] for row in rows), default=0),
               "kl_early_stopped_updates": sum(row["kl_early_stop"] for row in rows),
               "all_actor_critic_updates_nonzero": all(row["actor_change"] > 0 and row["critic_change"] > 0 for row in rows),
               "findings": findings, "limitations": ["single training seed", "32 fixed val_seen episodes, not full 778 split",
                                                      "no val_unseen evaluation", "4096 initial budget is not a convergence guarantee"]}
    write_json(report_dir / "SUMMARY.json", summary)
    print(json.dumps({key: summary[key] for key in ("complete", "trained_transitions", "updates", "max_initial_ratio_error", "findings")}, indent=2))


if __name__ == "__main__":
    main()
