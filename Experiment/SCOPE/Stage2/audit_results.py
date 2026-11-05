from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from data_guard import ROOT, STAGE1, protected_snapshot, write_json


def read(path):
    return json.loads(Path(path).read_text())


def review_stage1(output):
    with (STAGE1 / "RESULTS_PER_EPISODE.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    with (STAGE1 / "RESULTS_PER_STEP.csv").open() as stream:
        steps = list(csv.DictReader(stream))
    groups = {}
    for group in ("G0", "G1", "G2", "G3"):
        selected_episodes = [row for row in episodes if row["group"] == group]
        selected_steps = [row for row in steps if row["group"] == group]
        if len(selected_episodes) != 64 or len({row["episode_id"] for row in selected_episodes}) != 64:
            raise AssertionError("Stage1 denominator mismatch")
        groups[group] = {"episodes": len(selected_episodes), "steps": len(selected_steps),
                         "failures": sum(bool(row["failure"]) for row in selected_episodes),
                         "SR": statistics.mean(float(row["SR"]) for row in selected_episodes),
                         "SPL": statistics.mean(float(row["SPL"]) for row in selected_episodes),
                         "nDTW": statistics.mean(float(row["nDTW"]) for row in selected_episodes),
                         "current_tokens": sorted({int(row["current_visual_tokens"]) for row in selected_steps}),
                         "checkpoint_hashes": sorted({row["checkpoint_sha256"] for row in selected_episodes})}
    write_json(output, {"all_episode_rows": len(episodes), "all_step_rows": len(steps), "groups": groups})


def audit_navigation(directory):
    reports = {}
    for name in ("dev_before", "dev_after", "dev_restored"):
        records = read(directory / name / "results.json")
        reports[name] = {"episodes": len(records), "failures": sum(bool(row["failure"]) for row in records),
                         "SR": statistics.mean(row["success"] for row in records),
                         "nDTW": statistics.mean(row["nDTW"] for row in records), "records": records}
    after, restored = reports["dev_after"]["records"], reports["dev_restored"]["records"]
    keys = ("success", "oracle_success", "final_distance", "path_length", "steps", "termination_reason", "nDTW")
    reports["restored_metrics_exact"] = all(all(left[key] == right[key] for key in keys) for left, right in zip(after, restored, strict=True))
    return reports


def audit_diagnostics(directory):
    records = read(directory / "diagnostics_train/diagnostics.json") + read(directory / "diagnostics_dev/diagnostics.json")
    states = defaultdict(dict)
    for row in records:
        key = (row["purpose"], row["episode"], row["state_index"], row["history"])
        states[key][row["variant"]] = row
    deltas = defaultdict(list)
    native_errors = []
    pairs = []
    for key, variants in states.items():
        if len({row["image_sha256"] for row in variants.values()}) != 1:
            raise AssertionError("diagnostic pixel state differs")
        if max(row["distance_before"] for row in variants.values()) - min(row["distance_before"] for row in variants.values()) > 1e-6:
            raise AssertionError("diagnostic starting distance differs")
        glance = variants["glance"]["progress"]
        for name, row in variants.items():
            delta = row["progress"] - glance
            deltas[name].append(delta)
            pairs.append({"purpose": key[0], "episode": key[1], "state": key[2], "history": key[3],
                          "variant": name, "glance_progress": glance, "progress": row["progress"], "delta": delta})
        cached = np.asarray(variants["fixed"]["actions"])
        native = np.asarray(variants["fixed_native_scatter"]["actions"])
        native_errors.append(float(np.abs(cached - native).max()))
    if max(native_errors) != 0:
        raise AssertionError("matched-history cached/native scatter actions differ")
    summary = {name: {"count": len(values), "mean_delta_m": statistics.mean(values),
                      "positive": sum(value > 1e-4 for value in values),
                      "negative": sum(value < -1e-4 for value in values), "min": min(values), "max": max(values)}
               for name, values in deltas.items()}
    annotations = [variants["fixed_spatial"]["progress"] - variants["fixed"]["progress"] for variants in states.values()]
    cross_annotations = [variants["cross_spatial"]["progress"] - variants["cross_camera"]["progress"] for variants in states.values()]
    detail = [variants["fixed"]["progress"] - variants["fixed_low_detail"]["progress"] for variants in states.values()]
    history_interactions = []
    for key, variants in states.items():
        other = key[:3] + (False,)
        if key[3] and other in states:
            full_delta = variants["fixed"]["progress"] - variants["glance"]["progress"]
            no_history = states[other]["fixed"]["progress"] - states[other]["glance"]["progress"]
            history_interactions.append({"purpose": key[0], "episode": key[1], "state": key[2],
                                         "focus_delta_with_history": full_delta, "focus_delta_without_history": no_history,
                                         "interaction": full_delta - no_history})
    reencode = [error for row in records for error in row["metadata"]["reencode_errors"]]
    return {"comparisons": len(records), "state_history_pairs": len(states), "summary": summary,
            "pairs": pairs, "annotation_delta": annotations, "cross_annotation_delta": cross_annotations,
            "full_detail_minus_low_detail": detail, "history_interactions": history_interactions,
            "cached_native_action_max_abs": max(native_errors),
            "cached_native_feature_max_abs": max(error["max_abs"] for error in reencode),
            "cached_native_feature_relative_l2_max": max(error["relative_l2"] for error in reencode),
            "old_batched_feature_relative_l2_max": max(error["old_batched_relative_l2"] for error in reencode),
            "fixed_state_pixel_identity_passed": True}


def audit_training(directory, runtime):
    results = {}
    total_transitions, total_episode_seconds, total_phase_seconds = 0, 0.0, 0.0
    commands = {row["label"]: row for row in read(directory / "commands.json")}
    for regime in ("A", "B", "C"):
        rows = read(directory / f"train_{regime}/results.json")
        transitions = sum(row["training"]["transitions"] for row in rows)
        episode_seconds = sum(row["wall_seconds"] for row in rows)
        phase_seconds = commands[f"train_{regime}"]["wall_seconds"]
        statistics_rows = [item for row in rows for item in row["training"]["statistics"]]
        results[regime] = {"episodes": len(rows), "transitions": transitions,
                           "updates": sum(row["training"]["updated"] for row in rows),
                           "episode_seconds": episode_seconds, "phase_seconds": phase_seconds,
                           "transitions_per_second_episode_inclusive": transitions / episode_seconds,
                           "transitions_per_second_phase_inclusive": transitions / phase_seconds,
                           "ppo_update_seconds": sum(row["training"]["update_seconds"] for row in rows),
                           "actor_l2_changes": [row["training"]["actor_parameter_l2_change"] for row in rows],
                           "actor_grad_max": max(row["actor_grad_norm"] for row in statistics_rows),
                           "critic_grad_max": max(row["critic_grad_norm"] for row in statistics_rows),
                           "actor_state_gru_grad_max": max(row["actor_state_gru_grad_norm"] for row in statistics_rows),
                           "critic_state_gru_grad_max": max(row["critic_state_gru_grad_norm"] for row in statistics_rows),
                           "backbone_gradients_none": all(row["training"]["backbone_gradients_none"] for row in rows),
                           "first_epoch_ratios": [row["training"]["statistics"][0]["ratio_mean"] for row in rows]}
        total_transitions += transitions
        total_episode_seconds += episode_seconds
        total_phase_seconds += phase_seconds
    with (runtime / "trajectories.jsonl").open() as stream:
        trajectories = [json.loads(line) for line in stream]
    train = [row for row in trajectories if row["purpose"] == "train"]
    all_normal = [row for row in trajectories if not row["diagnostic_reencode"]]
    if any(row["bank"]["encoder_images"] != 4 + len(row["selected_roi_sequence"]) or
           row["reused_current"] != row["bank"]["encoder_images"] for row in all_normal):
        raise AssertionError("current visual cache reuse failed")
    if any(len(row["focus_summaries"]) != len(row["selected_roi_sequence"]) for row in trajectories):
        raise AssertionError("trajectory Focus summary mismatch")
    results["total"] = {"transitions": total_transitions, "episode_seconds": total_episode_seconds,
                        "phase_seconds": total_phase_seconds, "normal_navigation_steps": len(trajectories),
                        "focus_count_histogram_train": dict(Counter(len(row["selected_roi_sequence"]) for row in train)),
                        "reward_min": min(row["reward"] for row in train), "reward_max": max(row["reward"] for row in train),
                        "measured_filter_contact_chunks": sum(row["reward_teacher"]["collision"] for row in train),
                        "actual_feedback_state_updates": len(trajectories),
                        "current_feature_reuse_passed": True}
    rate = results["C"]["transitions_per_second_episode_inclusive"]
    conservative = results["C"]["transitions_per_second_phase_inclusive"]
    budget = {"reference_not_authorization": "199680 transitions per seed, 3 seeds, 2 GPUs",
              "per_seed_navigation_hours_optimistic": 199680 / rate / 3600,
              "per_seed_navigation_hours_phase_inclusive": 199680 / conservative / 3600,
              "three_seed_two_gpu_hours_optimistic": 199680 / rate / 3600 * 6,
              "three_seed_two_gpu_hours_phase_inclusive": 199680 / conservative / 3600 * 6,
              "excludes": ["long histories", "longer PPO epochs", "multi-environment scaling changes", "periodic dev evaluation",
                           "multi-seed variation", "failures and restarts", "full-training steady-state measurement"]}
    return results, budget


def audit_gpu(directory):
    samples = defaultdict(list)
    with (directory / "gpu_samples.csv").open() as stream:
        for row in csv.DictReader(stream):
            samples[int(row[" index"])].append(float(row[" memory.used [MiB]"].strip().split()[0]))
    return {str(device): {"samples": len(values), "peak_memory_mib": max(values), "sampling_ms": 50}
            for device, values in samples.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    complete = read(args.output / "COMPLETE.json")
    if not complete["passed"]:
        raise RuntimeError("real pilot has not passed")
    runtime = Path(complete["runtime"])
    review_stage1(ROOT / "artifacts/stage1_review_audit.json")
    training, budget = audit_training(args.output, runtime)
    report = {"navigation": audit_navigation(args.output), "diagnostics": audit_diagnostics(args.output),
              "training": training, "budget": budget, "gpu": audit_gpu(args.output),
              "pilot": complete, "learning_sanity": read(args.output / "learning_sanity.json")}
    original = ROOT / "artifacts/protected_before.json"
    earliest = read(original if original.exists() else args.output / "protected_before.json")
    current = protected_snapshot()
    changed = [path for path, digest in earliest.items() if current.get(path) != digest]
    report["protected_history"] = {"passed": not changed, "count": len(earliest), "changed": changed}
    write_json(args.output / "AUDIT.json", report)
    if changed:
        raise AssertionError("historical artifact changed")
    print(json.dumps({"training": training, "budget": budget, "diagnostic_summary": report["diagnostics"]["summary"],
                      "navigation": {key: value for key, value in report["navigation"].items() if key == "restored_metrics_exact"},
                      "gpu": report["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
