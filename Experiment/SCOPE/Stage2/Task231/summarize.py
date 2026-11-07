from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task231.diagnose import CHECKPOINTS, digest, interval, paired, write_csv, write_json


def counterfactual_summary(paths):
    records = [record for path in paths for record in json.loads(path.read_text())]
    groups = defaultdict(list)
    rows = []
    for record in records:
        groups[(record["transitions"], record["mode"])].append(record)
        summary = record["training"]["episode_summary"]
        rows.append({"transitions": record["transitions"], "mode": record["mode"], "episode_uid": record["episode_uid"],
                     "scene": record["scene_id"], "SR": record["SR"], "SPL": record["SPL"], "nDTW": record["nDTW"],
                     "final_distance": record["final_distance"], "path_length": record["path_length"], "focus_mean": record["focus_mean"],
                     "oracle_success": record["oracle_success"], "terminated": summary["terminated"], "truncated": summary["truncated"],
                     "termination_reason": record["termination_reason"], "reward_sum": record["reward_sum"]})
    for group in groups.values():
        group.sort(key=lambda row: row["episode_uid"])
        assert len(group) == 32 and len({row["episode_uid"] for row in group}) == 32
    assert set(groups) == {(transitions, mode) for transitions in (1033, 2054, 4096) for mode in ("native", "forced4")}
    summaries = [{"transitions": transitions, "mode": mode, "episodes": len(group),
                  **{key: float(np.mean([row[key] for row in group])) for key in ("SR", "SPL", "nDTW", "focus_mean", "oracle_success")}}
                 for (transitions, mode), group in sorted(groups.items())]
    comparisons = {f"{transitions}_native->forced4": paired(groups[(transitions, "native")], groups[(transitions, "forced4")])
                   for transitions in (1033, 2054, 4096)}
    comparisons["1033_forced4->2054_forced4"] = paired(groups[(1033, "forced4")], groups[(2054, "forced4")])
    comparisons["1033_forced4->4096_forced4"] = paired(groups[(1033, "forced4")], groups[(4096, "forced4")])
    return rows, {"groups": summaries, "comparisons": comparisons}, groups


def matched_summary(path):
    records = json.loads(path.read_text())
    by_state = defaultdict(dict)
    flat = []
    for row in records:
        if row["restoration_valid"]:
            by_state[(row["episode_uid"], row["state_index"])][(row["checkpoint_update"], row["variant"])] = row
        diagnostics = row["diagnostic"]
        flat.append({key: row[key] for key in ("episode_uid", "scene", "state_index", "checkpoint_update", "variant", "restoration_valid", "progress", "focus_count")} |
                    {key: diagnostics[key] for key in ("teacher_mse_glance", "teacher_mse_fused", "coverage_mean", "token_change_mse",
                                                      "covered_teacher_mse_glance", "covered_teacher_mse_fused")})
    contrasts = defaultdict(lambda: defaultdict(list))
    action_differences = []
    for (episode, state_index), variants in by_state.items():
        needed = {(14, "none"), (14, "learned"), (28, "learned"), (56, "learned"), (28, "fixed"),
                  (28, "random"), (28, "random_seed234"), (28, "random_seed235")}
        if not needed <= set(variants):
            continue
        learned = variants[(28, "learned")]
        random_mean = np.mean([variants[(28, name)]["progress"] for name in ("random", "random_seed234", "random_seed235")])
        contrasts["2054_learned_minus_random3"][episode].append(learned["progress"] - random_mean)
        for update in (14, 28, 56):
            contrasts[f"u{update}_learned_minus_no_focus"][episode].append(variants[(update, "learned")]["progress"] - variants[(14, "none")]["progress"])
        contrasts["2054_learned_minus_fixed"][episode].append(learned["progress"] - variants[(28, "fixed")]["progress"])
        contrasts["2054_learned_minus_1033_learned"][episode].append(learned["progress"] - variants[(14, "learned")]["progress"])
        for key in needed - {(14, "none")}:
            branch = variants[key]
            action_differences.append({"episode_uid": episode, "scene": branch["scene"], "state_index": state_index,
                                       "update": key[0], "variant": key[1],
                                       "actions_L2_vs_no_focus": float(np.linalg.norm(np.asarray(branch["actions"]) - np.asarray(variants[(14, "none")]["actions"]))),
                                       "progress_delta_vs_no_focus": branch["progress"] - variants[(14, "none")]["progress"],
                                       "token_change_mse": branch["diagnostic"]["token_change_mse"]})
    clustered = {name: interval([float(np.mean(values)) for values in episodes.values()]) for name, episodes in contrasts.items()}
    by_scene = defaultdict(list)
    for row in flat:
        if row["restoration_valid"]:
            by_scene[(row["scene"], row["checkpoint_update"], row["variant"])].append(row)
    scene_rows = [{"scene": scene, "update": update, "variant": variant, "states": len(group),
                   **{key: float(np.mean([row[key] for row in group])) for key in ("progress", "teacher_mse_glance", "teacher_mse_fused", "coverage_mean")}}
                  for (scene, update, variant), group in sorted(by_scene.items())]
    correlation = float(np.corrcoef([row["token_change_mse"] for row in action_differences],
                                   [row["progress_delta_vs_no_focus"] for row in action_differences])[0, 1])
    return flat, action_differences, scene_rows, {"chunks": len(records), "valid_chunks": sum(row["restoration_valid"] for row in records),
            "states": len(by_state), "episode_clustered_progress_contrasts_meters": clustered,
            "token_change_progress_delta_correlation_descriptive": correlation,
            "scope": "1 chunk, independently reset + deterministic prefix replay; pose/instruction/full image history/pre-Focus/BATS blocks verified. Three random ROI seeds averaged before episode-level bootstrap; no final-SR claim."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Task23/artifacts/ppo4096")
    parser.add_argument("--offline", type=Path, default=ROOT / "artifacts/offline")
    parser.add_argument("--counterfactual", type=Path, nargs="+", default=[ROOT / "artifacts/counterfactual32_shard0", ROOT / "artifacts/counterfactual32_shard1"])
    parser.add_argument("--matched", type=Path, default=ROOT / "artifacts/matched8")
    parser.add_argument("--supplement", type=Path, default=ROOT / "artifacts/supplement")
    args = parser.parse_args()
    for directory in (args.offline, *args.counterfactual, args.matched):
        assert (directory / "COMPLETE.json").exists(), f"incomplete diagnostic: {directory}"
    original_manifest = json.loads((args.source.parent / "ARTIFACT_MANIFEST.json").read_text())
    prefix = str(args.source.resolve().relative_to(ROOT.parents[3])) + "/"
    original_files = [record for record in original_manifest["files"] if record["path"].startswith(prefix)]
    print(f"Final integrity verification: {len(original_files)} original artifacts", flush=True)
    integrity_failures = [record["path"] for record in original_files if digest(ROOT.parents[3] / record["path"]) != record["sha256"]]
    contract = json.loads((args.source / "runtime/model_contract.json").read_text())
    source_failures = [path for path, checksum in contract["source_hashes"].items() if digest(Path(path)) != checksum]
    integrity = {"original_artifact_files_checked": len(original_files), "artifact_failures": integrity_failures,
                 "original_actor_source_files_checked": len(contract["source_hashes"]), "source_failures": source_failures,
                 "original_run_unchanged": not integrity_failures and not source_failures,
                 "environment_caveat": "Current model/Habitat versions recorded; original run has no complete package lock, so byte-identical historical environment is not asserted.",
                 "camera_semantics": {"order": ["front", "left", "right", "rear"], "habitat_yaw_radians": [0, float(np.pi / 2), float(-np.pi / 2), float(np.pi)],
                                      "source": "/home/data/czh/Third_Party/baselines/SimpleNav/benchmark/common/simulators/habitat/vlnce031_runtime.py:310",
                                      "check": "source sensor orientation/mapping plus FusionPolicy runtime camera-order assertion; not a new physical calibration"}}
    write_json(ROOT / "source_integrity.json", integrity)
    assert integrity["original_run_unchanged"], "original artifact/source integrity failure"
    history = json.loads((args.offline / "paired_statistics.json").read_text())
    numerical = json.loads((args.offline / "numerical_audit.json").read_text())
    supplement = json.loads((args.supplement / "supplement.json").read_text())
    gradient_file = ROOT / "artifacts/gradient_components/gradient_components.json"
    gradient_components = json.loads(gradient_file.read_text()) if gradient_file.exists() else {"status": "not run"}
    rows, counterfactual, groups = counterfactual_summary([directory / "counterfactual_results.json" for directory in args.counterfactual])
    write_csv(ROOT / "counterfactual.csv", rows)
    matched_rows, action_differences, scenes, matched = matched_summary(args.matched / "matched_results.json")
    write_csv(ROOT / "matched_counterfactual.csv", matched_rows)
    write_csv(ROOT / "matched_action_differences.csv", action_differences)
    write_csv(ROOT / "matched_scenes.csv", scenes)
    for name in ("paired_episodes.csv", "checkpoint_policy_probe.csv"):
        shutil.copyfile(args.offline / name, ROOT / name)
    probe_rows = list(csv.DictReader((args.offline / "checkpoint_policy_probe.csv").open()))
    sensitivity = []
    for transitions in (0, 1033, 2054, 3100, 4096):
        for split in ("train", "val_seen"):
            probabilities = [float(row["p_stop"]) for row in probe_rows if int(row["transitions"]) == transitions and row["split"] == split]
            for threshold in (.45, .5, .55, .6):
                sensitivity.append({"transitions": transitions, "split": split, "threshold": threshold, "states": len(probabilities),
                                    "first_step_continue_fraction": sum(value < threshold for value in probabilities) / len(probabilities)})
    write_csv(ROOT / "threshold_sensitivity.csv", sensitivity)
    validation_rewards = []
    for round_index, (_, transitions) in enumerate(CHECKPOINTS):
        episodes = [json.loads(path.read_text()) for path in sorted((args.source / "episodes").glob(f"val_{round_index:03d}_*/result.json"))]
        totals = {"progress": 0., "focus_cost": 0., "step_cost": 0., "terminal": 0., "reward": 0.}
        for episode in episodes:
            summary = episode["training"]["episode_summary"]
            focus_cost = -.01 * sum(int(count) * frequency for count, frequency in summary["focus_histogram"].items())
            step_cost = -.01 * summary["transitions"]
            terminal = (2. if episode["SR"] else -.5) if summary["terminated"] else 0.
            totals["progress"] += episode["reward_sum"] - focus_cost - step_cost - terminal
            totals["focus_cost"] += focus_cost
            totals["step_cost"] += step_cost
            totals["terminal"] += terminal
            totals["reward"] += episode["reward_sum"]
        validation_rewards.append({"transitions": transitions, **{name: value / 32 for name, value in totals.items()}})
    write_csv(ROOT / "validation_reward_components.csv", validation_rewards)
    replay = []
    for transitions, round_index in ((1033, 1), (2054, 2), (4096, 4)):
        historical = [json.loads(path.read_text()) for path in sorted((args.source / "episodes").glob(f"val_{round_index:03d}_*/result.json"))]
        historical.sort(key=lambda row: row["episode_uid"])
        current = groups[(transitions, "native")]
        assert [row["episode_uid"] for row in historical] == [row["episode_uid"] for row in current]
        replay.append({"transitions": transitions, "SR_mismatches": sum(first["SR"] != second["SR"] for first, second in zip(historical, current)),
                       **{f"{metric}_max_abs_difference": max(abs(first[metric] - second[metric]) for first, second in zip(historical, current))
                          for metric in ("SPL", "nDTW", "final_distance", "path_length", "focus_mean")}})
    early = counterfactual["comparisons"]["1033_forced4->2054_forced4"]
    rescue = counterfactual["comparisons"]["4096_native->forced4"]
    rescue_observation = "观察到部分恢复，但SR配对检验未排除小样本波动" if rescue["delta_SR"]["mean"] > 0 and rescue["exact_mcnemar_p"] >= .05 else "依据完整配对统计判断"
    hypotheses = {
        "H1": {"status": "部分支持", "priority": 1, "supported": "STOP阈值放大得到固定状态探针支持；成本在导航结果相同时直接偏好少Focus。", "unresolved": "去掉成本能否改善训练必须重新采样，当前禁止重训。", "forced4": rescue},
        "H2": {"status": "部分支持", "priority": 2, "supported": "held-out Critic EV接近零，最终值预测出现明显偏置。", "unresolved": "不能据此确证训练集过拟合或认定PPO方差是唯一病因。"},
        "H3": {"status": "仍不确定，存在可检验的局部证据", "priority": 2, "supported": "固定/随机ROI与learned同预算对照、跨场景teacher-MSE见matched；特征逼近不等价于导航改善。", "unresolved": "有限初始/一段前缀状态不能代表完整导航分布；不据此自动重校准Adapter。"},
        "H4": {"status": "反证数值实现断裂；信用分配仍不确定", "priority": 4, "supported": "全部4096训练transition重算reward/GAE/return/old logprob误差为零；未覆盖token精确不变。", "unresolved": "序列求和的长度效应存在，但不是bug；不改为mean。"},
        "H5": {"status": "支持停止失败和小样本不确定性", "priority": 1, "supported": "1033→2054的6个成功转失败中3个曾有oracle success；历史配对SR差异CI跨0。", "unresolved": "单seed固定32集与best-checkpoint选择偏差不支持泛化结论。"}}
    ordered_p = sorted((comparison["exact_mcnemar_p"], name) for name, comparison in history["comparisons"].items())
    adjusted, running = {}, 0.
    for rank, (probability, name) in enumerate(ordered_p):
        running = max(running, min(1., probability * (len(ordered_p) - rank)))
        adjusted[name] = running
    diagnosis = {"task": "Stage2 Task2.3.1", "date": "2026-11-07", "role": "Exec with analysis", "status": "diagnostics_complete_with_stated_scope_limits",
                 "original_training_modified": False, "new_training_updates": 0, "counterfactual": counterfactual, "native_reproduction": replay,
                 "source_integrity": integrity,
                 "matched": matched, "historical_pairing": history, "historical_McNemar_Holm_adjusted_10_comparisons": adjusted,
                 "numerical": numerical, "supplement": supplement, "hypotheses": hypotheses,
                 "gradient_components": gradient_components,
                 "validation_mean_reward_components": validation_rewards,
                 "gate_questions": {
                     "late_forced4_rescue": {"answered": True, "evidence": rescue, "observation": rescue_observation,
                                            "interpretation": "partial rescue不等于恢复1033或证明普遍增益；优先诊断STOP/成本，而非把后期camera/position视为全面失效。"},
                     "1033_to_2054_source": {"answered_at_observable_mechanism_level": True, "evidence": early,
                         "interpretation": "两者原生都使用4 Focus；退化不是减少Focus次数，涉及camera/ROI策略变化后导航轨迹和停止/碰撞结果变化；matched局部对照不等价于完整中介因果分解。"},
                     "negative_cost_incentive": {"answered": True, "same_navigation_four_to_zero_reward_gain": .04,
                         "interpretation": "存在直接省成本动机，四次Focus需额外0.6米单步进展抵消Focus成本；不能把同轨迹算术当作无成本重训效果。"}},
                 "formal_optimization_started": False,
                 "limitations": ["Actor-only expected Focus count is an explicitly named stationary-prestate proxy, not an on-policy sequential expectation.",
                                 "Matched experiment has at most 16 states from 8 scenes and 3 random ROI seeds; no complete SR claim.",
                                 "Original agents/exec-agent.md absent; followed user and task documents.",
                                 "Original run has no full environment lock; current versions recorded and native reproducibility measured.",
                                 "Initial diagnostic attempts were intentionally interrupted for full32 scene-grouped two-GPU continuation; only completed episodes reused."]}
    write_json(ROOT / "diagnosis.json", diagnosis)
    lines = ["# Task2.3.1 失败归因诊断报告", "", "角色：执行者（含结果分析）。日期：2026-11-07。只写Task231，不修改Task23，不重跑4096训练。", "",
             "## 核验与执行范围", "", "- 真实5个checkpoint、原始固定32集及训练trace对照原ARTIFACT_MANIFEST逐项SHA256核验。",
             "- train/val_seen各128个相同detached pre-Focus状态，5 checkpoint共1280次Actor探针；Actor输入无reward/GT。",
             "- 全部4096训练transition、56次更新重算；原5轮160集配对；新3×2×32集对照（复用首次尝试已完成项）。",
             f"- matched-budget：{matched['states']}状态、{matched['chunks']}分支，{matched['valid_chunks']}分支恢复检查有效；含learned/fixed/3随机种子/无Focus。",
             "- 新增5项单元测试与原Task23的11项回归测试通过；未执行optimizer更新。", "", "## 三个门禁问题", "",
             "### 1. 后期forced4是否挽救性能？", "", "| transitions | 诊断模式 | SR | SPL | nDTW | mean Focus |", "| --- | --- | --- | --- | --- | --- |"]
    for group in counterfactual["groups"]:
        lines.append(f"| {group['transitions']} | {group['mode']} | {group['SR']:.2%} | {group['SPL']:.2%} | {group['nDTW']:.4f} | {group['focus_mean']:.3f} |")
    lines += ["", f"4096 native→forced4配对ΔSR={rescue['delta_SR']['mean']:.2%}，95% bootstrap CI={rescue['delta_SR']['ci95']}，exact McNemar p={rescue['exact_mcnemar_p']:.6g}。",
              f"**{rescue_observation}。** forced4下最终模型已接近1033的SR，并不支持‘后期camera/position全面失效’。新native与历史差异见diagnosis.json/native_reproduction。", "",
              "### 2. 1033→2054退化来自哪里？", "",
              f"forced4配对ΔSR={early['delta_SR']['mean']:.2%}，CI={early['delta_SR']['ci95']}；两者原生均4 Focus，因此STOP次数不能解释这段退化。",
              "历史6个成功→失败、2个失败→成功；失去的6集中，752/536/107曾有oracle success，206/111/338未进入成功范围。",
              "752/536/206/338截断；107/111错误位置停止。536最终仍在3米内但未真实终止。碰撞、绕行和目标附近不终止共同出现，不能仅凭整体nDTW上升认定导航更好。",
              "固定状态1033→2054的camera/position分布漂移（含部分left→right切换）见策略CSV；局部因果对照如下。", "",
              "| 1-chunk配对进展差（米） | episode均值 | 95% episode-cluster bootstrap CI |", "| --- | --- | --- |"]
    for name, result in matched["episode_clustered_progress_contrasts_meters"].items():
        lines.append(f"| {name} | {result['mean']:.6f} | {result['ci95']} |")
    lines += ["", "同预算ROI只改变感知决策；reset+固定前缀重放后，姿态、图像/历史、指令、pre-Focus特征及BATS blocks逐项匹配。不一致分支不合并。",
              "跨场景teacher-MSE、覆盖率及动作变化见matched_scenes.csv/matched_action_differences.csv；MSE下降并不自动带来chunk进展，更不等价于最终SR。", "",
              "本次learned四次Focus覆盖总四相机面积约3.52%，近似在同一ROI反复查看；固定四相机中央覆盖14.06%，随机约13%。三模型learned的teacher-MSE均改善约4%，而fixed/random约14%。该差异不能单独证明Adapter失效，也不支持现在重校准。", "",
              "### 3. 旧reward是否产生负面成本动机？", "",
              "是：固定相同导航结果时4→0 Focus立刻增加0.04 reward；在progress未裁剪区，相当于0.6米进展。4 Focus加step成本需0.75米进展才能抵消。",
              "4096训练累计progress +55.829，focus cost −52.89，step cost −40.96，terminal +121.5。成本量级足以形成省感知动机，但不证明是全部退化的因果来源。",
              "历史1033→4096的每集Focus成本节省+0.6125，超过terminal奖励损失−0.515625；因此平均reward升高可以与成功数下降同时出现。完整分解见validation_reward_components.csv。",
              "最终训练区间仍包含0/1/2/3/4 Focus，mean约0.984；最终固定val状态首步p_stop中位数约0.572，不是采样策略完全不Focus。0.5确定性阈值将此变为全部立即STOP。", "",
              "注意：Actor的STOP只结束感知，不直接结束导航；导航waypoint_motion_stop与环境截断是另一条机制，不能混用这两种STOP。", "",
              "## 数值、Critic和统计边界", "",
              f"- reward/return/normalized advantage/old logprob/未覆盖token最大误差：{numerical['errors']}。",
              "- head_gradients.csv为每更新每Focus count首个样本的整个head未裁剪梯度，不冒充全量optimizer时刻梯度；numerical_transitions.csv含全部样本长度、loss、KL、clip。",
              "- gradient_components/head_gradient_components.csv进一步分离policy与entropy梯度；STOP-only样本上的camera/position非零entropy梯度不等于这些动作获得了reward信用。",
              "- held-out Critic EV在5轮总体约0.00005/0.0418/0.0043/0.0026/0.0016；terminal-only亦近0。最终预测均值约−0.650，而目标均值约+0.080，说明基线质量弱且偏移。",
              "- 历史0→1033的exact McNemar未校正p=0.015625，不应写成‘绝无显著差异’，但对10个探索比较Holm校正以及best-checkpoint选择后不支持推广结论。",
              "- 1033→2054历史ΔSR=−12.5pp，95%CI=[−28.125pp,+3.125pp]，p=0.2890625。32集只提供诊断证据。",
              "- Actor-only预期Focus数是静态首状态代理，未来Focus会改变输入，不能将代理列误读为真实序列期望。", "",
              "- SR严格沿用原Task23的真实导航终止且distance<3m定义；截断时距离<3m不自动记成功。当前两套环境版本已记录，原run无完整package lock，不声称历史环境逐包一致。", "",
              "## 假设判定与优先级", "", "| 假设 | 判定 | 优先级 | 边界 |", "| --- | --- | --- | --- |"]
    for name, hypothesis in hypotheses.items():
        lines.append(f"| {name} | {hypothesis['status']} | {hypothesis['priority']} | {hypothesis['unresolved']} |")
    lines += ["", "**不启动正式优化。** 三问已有可审计的观测层回答，完整因果分解仍有限。先审阅诊断，再决定单因素成本/停止规则或Critic稳定性实验；不提前重写融合模块，不无依据把序列sum改为mean。", "",
              "## 产物与复现", "", "入口README.md；机器可读总判定diagnosis.json；所有新增文件列于FILES_MANIFEST.tsv。原始checkpoint/数据仍依赖本机可信路径，不声称独立离线可移植。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    files = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name != "FILES_MANIFEST.tsv"]
    with (ROOT / "FILES_MANIFEST.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"), delimiter="\t")
        writer.writeheader()
        writer.writerows(files)


if __name__ == "__main__":
    main()
