from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def write_csv(path, rows):
    if not rows:
        raise ValueError("refuse empty evidence table")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(records, key, level=.95, replicates=10000):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["episode_uid"]].append(row[key])
    values = np.asarray([np.mean(items) for items in grouped.values()])
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("insufficient independent finite episodes")
    generator = np.random.default_rng(233)
    draws = values[generator.integers(len(values), size=(replicates, len(values)))].mean(axis=1)
    tail = (1 - level) / 2
    return {"mean": float(values.mean()), "ci_low": float(np.quantile(draws, tail)),
            "ci_high": float(np.quantile(draws, 1 - tail)), "episodes": len(values), "anchors": len(records)}


def rank_correlation(first, second):
    def ranks(values):
        values = np.asarray(values)
        return np.asarray([np.count_nonzero(values < value) + (np.count_nonzero(values == value) - 1) / 2 for value in values])
    first_rank, second_rank = ranks(first), ranks(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def summarize_anchor(records, thresholds):
    variants = {row["variant"]: row for row in records}
    if len(variants) != len(records):
        raise ValueError("duplicate branches cannot inflate sample size")
    candidates = [variants[f"candidate_{index}"] for index in range(8)]
    baseline = variants["stop"]
    ranked = sorted(candidates, key=lambda row: row["score_gt"], reverse=True)
    teacher, runner_up = ranked[:2]
    fixed = variants[f"candidate_{baseline['j'] % 4}"]
    random_progress = float(np.mean([variants[f"random_{index}"]["progress"] for index in range(3)]))
    oracle = max(candidates, key=lambda row: row["progress"])
    margin = teacher["score_gt"] - runner_up["score_gt"]
    result = {key: baseline[key] for key in ("anchor_id", "episode_uid", "scene", "partition", "state_source", "j")}
    result.update(teacher_variant=teacher["variant"], gt_margin=margin,
                  gt_range=max(row["score_gt"] for row in candidates) - min(row["score_gt"] for row in candidates),
                  environment_range=max(row["progress"] for row in candidates) - min(row["progress"] for row in candidates),
                  rank_correlation=rank_correlation([row["score_gt"] for row in candidates], [row["progress"] for row in candidates]),
                  teacher_minus_stop=teacher["progress"] - baseline["progress"],
                  teacher_minus_fixed=teacher["progress"] - fixed["progress"],
                  teacher_minus_random=teacher["progress"] - random_progress,
                  oracle_minus_stop=oracle["progress"] - baseline["progress"],
                  oracle_minus_fixed=oracle["progress"] - fixed["progress"],
                  oracle_minus_random=oracle["progress"] - random_progress,
                  learned_minus_stop=variants["learned"]["progress"] - baseline["progress"],
                  learned_minus_fixed=variants["learned"]["progress"] - fixed["progress"],
                  learned_minus_random=variants["learned"]["progress"] - random_progress,
                  teacher_camera=teacher["diagnostic"]["selected"][-1]["camera"],
                  teacher_union_coverage=teacher["diagnostic"]["union_coverage"],
                  learned_union_coverage=variants["learned"]["diagnostic"]["union_coverage"],
                  fixed_union_coverage=fixed["diagnostic"]["union_coverage"],
                  teacher_stop=teacher["navigation_stopped"], expert_stop=baseline["expert"]["stop"])
    result["confident"] = margin >= thresholds["min_gt_margin"] and result["teacher_minus_stop"] >= thresholds["min_progress_m"]
    return result


def gate_decision(anchors, thresholds, integrity):
    evaluation = [row for row in anchors if row["partition"] != "calibration"]
    heldout = [row for row in anchors if row["partition"] == "teacher_heldout"]
    fraction = float(np.mean([row["confident"] for row in heldout])) if heldout else 0.
    checks = {"evaluation_count": len(evaluation) >= thresholds["minimum_evaluation_anchors"],
              "heldout_count": len(heldout) >= thresholds["minimum_heldout_anchors"],
              "integrity": integrity, "credible_label_fraction": fraction >= thresholds["minimum_confident_fraction"]}
    effects = []
    for control in ("stop", "fixed", "random"):
        if len({row["episode_uid"] for row in heldout}) < 2:
            checks["teacher_beats_" + control] = False
            continue
        result = paired_interval(heldout, "teacher_minus_" + control, thresholds["paired_ci_level"], thresholds["bootstrap_episode_replicates"])
        effects.append({"comparison": "teacher_minus_" + control, **result})
        checks["teacher_beats_" + control] = result["ci_low"] > 0 and result["mean"] >= thresholds["min_progress_m"]
    return {"status": "PASS" if all(checks.values()) else "INCONCLUSIVE" if integrity else "FAIL",
            "checks": checks, "heldout_credible_fraction": fraction, "gate_effects": effects}


def analyze(output):
    output = Path(output)
    complete = json.loads((output / "COMPLETE.json").read_text())
    thresholds = json.loads((output / "FROZEN_THRESHOLDS.json").read_text())
    rows = [json.loads(line) for line in (output / "candidate_results.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row["split"] != "train" or row["data_kind"] != "teacher_candidate" or row["ppo_eligible"]:
            raise PermissionError("teacher/on-policy/split boundary violated")
        groups[row["anchor_id"]].append(row)
    anchors = [summarize_anchor(records, thresholds) for records in groups.values()]
    integrity = (all(row["restoration_valid"] and row["diagnostic"]["current_tokens"] == 64 for row in rows)
                 and all(complete["audit"]["checks"].values())
                 and json.loads((output / "SOURCE_INTEGRITY_AFTER.json").read_text())["unchanged"])
    decision = gate_decision(anchors, thresholds, integrity)
    write_csv(output / "candidate_quality.csv", anchors)
    effects = []
    for partition in ("teacher_train", "teacher_heldout"):
        subset = [row for row in anchors if row["partition"] == partition]
        for source in ("teacher", "oracle", "learned"):
            for control in ("stop", "fixed", "random"):
                key = source + "_minus_" + control
                effects.append({"partition": partition, "comparison": key, "level": .95, **paired_interval(subset, key)})
    write_csv(output / "paired_effects.csv", effects)
    pairs = []
    for records in groups.values():
        baseline = next(row for row in records if row["variant"] == "stop")
        for row in records:
            pairs.append({"anchor_id": row["anchor_id"], "partition": row["partition"], "variant": row["variant"],
                          "j": row["j"], "score_gt": row["score_gt"], "progress": row["progress"],
                          "gt_gain_vs_microstep_stop": row["score_gt"] - baseline["score_gt"],
                          "progress_gain_vs_microstep_stop": row["progress"] - baseline["progress"],
                          "focus_count": row["diagnostic"]["focus_count"], "terminal": row["terminated"],
                          "truncated": row["truncated"], "native_steps": row["native_steps"]})
    write_csv(output / "candidate_pairs.csv", pairs)
    strata = []
    for field in ("scene", "state_source", "j"):
        for value in sorted({row[field] for row in anchors if row["partition"] == "teacher_heldout"}):
            subset = [row for row in anchors if row["partition"] == "teacher_heldout" and row[field] == value]
            strata.append({"field": field, "value": value, "anchors": len(subset),
                           "teacher_minus_stop": float(np.mean([row["teacher_minus_stop"] for row in subset])),
                           "teacher_minus_random": float(np.mean([row["teacher_minus_random"] for row in subset])),
                           "confident_fraction": float(np.mean([row["confident"] for row in subset]))})
    write_csv(output / "strata.csv", strata)
    valid_correlations = [row["rank_correlation"] for row in anchors if row["partition"] == "teacher_heldout" and row["rank_correlation"] is not None]
    audit = {**decision, "anchors": len(anchors), "branches": len(rows), "thresholds": thresholds,
             "heldout_mean_rank_correlation": float(np.mean(valid_correlations)) if valid_correlations else None,
             "rank_correlation_nonconstant_anchors": len(valid_correlations), "complete": complete,
             "result_sha256": sha256(output / "candidate_results.jsonl"), "optimizer_updates": 0,
             "SR_SPL": "NOT_RUN: one-chunk counterfactuals are not complete navigation episodes",
             "heldout_critic": "NOT_RUN: no Critic used or trained in Gate0"}
    write_json(output / "teacher_signal_audit.json", audit)
    write_json(output / "restore_exactness.json", {"branches": len(rows), "all_matched": all(row["restoration_valid"] for row in rows),
               "comparison": "environment+history+context; real perception prefix state except separately labeled glance-only"})
    write_json(output / "PROGRESS_GATE.json", {"Gate0": decision["status"], "Gate1": "NOT_RUN", "Gate2": "NOT_RUN", "Gate3": "NOT_RUN",
               "downstream_locked": decision["status"] != "PASS", "failed_checks": [key for key, passed in decision["checks"].items() if not passed],
               "teacher_signal_audit_sha256": sha256(output / "teacher_signal_audit.json"), "optimizer_updates": 0})
    lines = ["# Task233 Gate0 实测报告", "", f"Gate0：{decision['status']}。Gate1/2/3：NOT_RUN。", "",
             f"实际完成{len(anchors)} anchors / {len(rows)}候选分支；包括独立校准8 anchors。", "",
             f"held-out可信标签比例：{decision['heldout_credible_fraction']:.2%}；要求至少25%。", "",
             "| held-out GT教师配对progress差（米） | 均值 | 判门98.333%CI |", "| --- | --- | --- |"]
    for effect in decision["gate_effects"]:
        lines.append(f"| {effect['comparison']} | {effect['mean']:.6f} | [{effect['ci_low']:.6f}, {effect['ci_high']:.6f}] |")
    lines += ["", "未通过项：" + ", ".join(key for key, passed in decision["checks"].items() if not passed), "",
              "本次只做真实1-chunk反事实评价，不能报告整集SR/SPL提升。训练0步；未创建optimizer、Critic buffer或Group数据。",
              "若GT代理未过门，即使oracle-best可获益也不授权warm-up；只能说明当前教师协议不足，不证明冻结导航无可学价值。",
              "原始候选、动作、当前位姿专家路径、真实ROI/latent、恢复身份及成本见candidate_results.jsonl。",
              "成本见COMPLETE.json和runtime/scope.jsonl；所有预测包含4当前Glance图的编码及实际Focus调用，历史编码另按telemetry核计。",
              "完整learned/fixed/random/no-Focus导航验收和held-out Critic留在锁定的下游门禁，不用历史指标代填。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decision), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output)


if __name__ == "__main__":
    main()
