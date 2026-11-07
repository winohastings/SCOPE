from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task231.diagnose import paired
from Experiment.SCOPE.Stage2.Task232.metrics import write_csv
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def load(path):
    return json.loads(Path(path).read_text())


def extension_eligibility(base, branch, probe, audits):
    independent = branch["subsets"]["independent"]
    reference = base["subsets"]["independent"]
    coverage = independent["four_focus_coverage"]
    repeat = independent["four_focus_repeat_fraction"]
    checks = {
        "fixed_zero_focus_fraction_le_010": probe["heads"]["val_seen"]["zero_focus_first_step_fraction"] <= .10,
        "independent_zero_focus_fraction_le_010": independent["zero_focus_fraction"] <= .10,
        "four_focus_coverage_gain_ge_25pct": coverage is not None and coverage >= 1.25 * reference["four_focus_coverage"],
        "repeated_roi_fraction_drop_ge_015": repeat is not None and repeat <= reference["four_focus_repeat_fraction"] - .15,
        "independent_SR_loss_le_one_episode": independent["SR"] >= reference["SR"] - 1 / 32,
        "independent_SPL_loss_le_003": independent["SPL"] >= reference["SPL"] - .03,
        "historical_SR_loss_le_two_episodes": branch["subsets"]["historical"]["SR"] >= .5625 - 2 / 32,
        "terminal_only_EV_loss_le_005": independent["critic"]["branch_terminal_only"]["EV"] >= reference["critic"]["branch_terminal_only"]["EV"] - .05,
        "all_engineering_gates": audits,
    }
    return {"eligible": all(checks.values()), "checks": checks, "failed": [name for name, passed in checks.items() if not passed]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/baseline_independent")
    parser.add_argument("--a", type=Path, default=ROOT / "artifacts/branch_A512")
    parser.add_argument("--b", type=Path, default=ROOT / "artifacts/branch_B512")
    args = parser.parse_args()
    base = load(args.baseline / "analysis/summary.json")
    reports = {"A": load(args.a / "analysis/summary.json"), "B": load(args.b / "analysis/summary.json")}
    runs = {"A": args.a, "B": args.b}
    decisions, comparisons, summary_rows, progress = {}, {}, [], []
    source = ROOT.parent / "Task23/artifacts/ppo4096"
    historical = [load(path) for path in sorted((source / "episodes").glob("val_001_*/result.json"))]
    original_new = load(args.baseline / "evaluation_independent.json")
    reference_sets = {"historical": sorted(historical, key=lambda row: row["episode_uid"]),
                      "independent": sorted(original_new, key=lambda row: row["episode_uid"])}
    for branch, report in reports.items():
        run = runs[branch]
        probe = load(run / "runtime/probes/added_0512.json")
        complete = load(run / "COMPLETE.json")
        audits = load(run / "RESTORE_GATE.json")["passed"] and all(complete["freeze_audit"].values()) and report["ratio_max_error"] < 1e-4
        assert report["subsets"]["train"]["transitions"] == 512 and report["added_trained_transitions"] == 512
        decisions[branch] = extension_eligibility(base, report, probe, audits)
        comparisons[branch] = {}
        for subset in ("historical", "independent"):
            current = sorted(load(run / f"evaluation_{subset}.json"), key=lambda row: row["episode_uid"])
            comparisons[branch][subset] = paired(reference_sets[subset], current)
            result = report["subsets"][subset]
            summary_rows.append({"branch": branch, "subset": subset, "new_train_transitions": 512,
                                 **{key: result[key] for key in ("SR", "SPL", "nDTW", "focus_mean", "zero_focus_fraction", "coverage_mean", "four_focus_coverage", "four_focus_repeat_fraction")},
                                 "critic_EV": result["critic"]["branch_all_bootstrapped"]["EV"],
                                 "critic_terminal_only_EV": result["critic"]["branch_terminal_only"]["EV"],
                                 "optimizer_steps": report["new_optimizer_steps"], "update_seconds": report["update_seconds"],
                                 "train_wall_seconds": report["subsets"]["train"]["wall_seconds"],
                                 "train_model_seconds": report["subsets"]["train"]["model_seconds"]})
        for count in (0, 256, 512):
            fixed = load(run / "runtime/probes" / f"added_{count:04d}.json")
            progress.append({"branch": branch, "added_transitions": count,
                             "p_stop_median": fixed["heads"]["val_seen"]["p_stop_P10_P50_P90"][1],
                             "zero_focus_fraction": fixed["heads"]["val_seen"]["zero_focus_first_step_fraction"],
                             "camera_entropy": fixed["heads"]["val_seen"]["camera_entropy"],
                             "camera_top1_fractions": json.dumps(fixed["heads"]["val_seen"]["camera_top1_fractions"], sort_keys=True),
                             "position_KL_from_fork": fixed["heads"]["val_seen"]["KL_from_fork"]["position"],
                             "conditional_four_coverage": fixed["geometry"]["forced4_coverage"],
                             "conditional_four_repeat": fixed["geometry"]["forced4_repeated_roi_fraction"]})
    write_csv(ROOT / "branch_comparison.csv", summary_rows)
    write_csv(ROOT / "early_probes.csv", progress)
    write_json(ROOT / "extension_eligibility.json", decisions)
    write_json(ROOT / "paired_evaluation.json", comparisons)
    complementary = decisions["A"]["checks"]["four_focus_coverage_gain_ge_25pct"] and decisions["A"]["checks"]["repeated_roi_fraction_drop_ge_015"] and reports["B"]["subsets"]["independent"]["critic"]["branch_terminal_only"]["EV"] > base["subsets"]["independent"]["critic"]["branch_terminal_only"]["EV"] + .02
    proposal = {"train_A_plus_B_now": False, "complementary_evidence": complementary,
                "rationale": "Only propose a separately labeled combination if A spatial improvement and B held-out baseline quality are both supported; no combination training authorized by this automated report.",
                "next": "review a separate A+B protocol" if complementary else "do not combine unsupported fixes; consider an isolated lightweight repetition-constraint diagnostic"}
    decision = {"task": "Task2.3.2", "date": "2026-11-07", "stage": "512-transition screening", "fork": load(ROOT / "frozen/FORK_AUDIT.json"),
                "baseline": base, "branches": reports, "extension_eligibility": decisions, "paired_evaluation": comparisons,
                "A_plus_B": proposal, "qualified_for_extension": [branch for branch, record in decisions.items() if record["eligible"]],
                "budgets": {"unchanged_gate_fresh": 64, "A_fresh": 512, "B_fresh": 512, "restore_replays_are_not_new_data": True},
                "heldout_scope": "independent episode AND trajectory from historical32; not necessarily scene-disjoint and now a development set, not an untouched final test",
                "control_limitation": "U has 64 fresh transitions for engineering consistency, not a budget-matched 512 unchanged-training arm; comparisons to the 1033 baseline do not isolate repair benefit over ordinary 512-step continuation",
                "spatial_probe_scope": "fixed historical intermediate-state geometry is conditional; extension gate uses real held-out four-Focus union coverage",
                "retraining_adapter": False, "GA_PPO": False, "original_modules_changed": False}
    write_json(ROOT / "repair_decision.json", decision)
    lines = ["# Task2.3.2 定向短续训报告", "", "角色：执行者（含分析）；2026-11-07。全部新内容仅在Task232，保留原视觉融合、Actor及冻结导航模型。", "",
             "## 执行与门禁", "", "- 从update_0014完整checkpoint分叉：1033 transitions、update14、下一train ordinal70；保留Actor/Critic/old、optimizer动量与RNG。",
             "- U门禁新增64条真实transition；首episode13条状态/动作/logprob/value与历史逐位一致。U/A/B各自首更新restore重放逐位一致。",
             "- A成本0.0025；B成本0.01、rollout minimum128、Actor LR3e-5/Critic LR1e-4；其余算法参数不变。每分支512新增on-policy数据，不复用历史buffer。",
             "- 256/512真实预算边界probe；精确预算截断使用真实后继bootstrap。B的实际batch可能因episode边界及最小batch门禁大于128。",
             "- 独立32集按seed232固定，排除历史32的episode及trajectory；原32保留为历史诊断。仅用新集配对效果和空间/Critic门槛共同决策，不取单次最高SR。", "",
             "## 独立集基线与结果", "",
             f"BASE1033：SR={base['subsets']['independent']['SR']:.3%}，SPL={base['subsets']['independent']['SPL']:.3%}，N=4覆盖={base['subsets']['independent']['four_focus_coverage']:.6f}，重复比例={base['subsets']['independent']['four_focus_repeat_fraction']:.6f}。", "",
             "| 分支 | 集合 | SR | SPL | mean Focus | N=4联合覆盖 | N=4重复比例 | Critic EV / terminal-only |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary_rows:
        lines.append(f"| {row['branch']} | {row['subset']} | {row['SR']:.3%} | {row['SPL']:.3%} | {row['focus_mean']:.3f} | {row['four_focus_coverage']} | {row['four_focus_repeat_fraction']} | {row['critic_EV']:.5f} / {row['critic_terminal_only_EV']:.5f} |")
    lines += ["", "覆盖按四相机总面积归一化；重复定义为同相机且与历史ROI IoU≥0.8。真实N=4覆盖与固定参考状态几何代理严格区分。", "",
              "## 配对证据与早期趋势", ""]
    for branch in ("A", "B"):
        paired_result = comparisons[branch]["independent"]
        lines.append(f"- {branch}独立集ΔSR={paired_result['delta_SR']['mean']:.3%}，95% episode-bootstrap CI={paired_result['delta_SR']['ci95']}，McNemar p={paired_result['exact_mcnemar_p']:.6g}。")
    lines += ["", "| 分支 | 新增步数 | 固定val p_stop中位数 | 首步零Focus比例 | camera entropy | Position KL相对fork | 条件N4覆盖代理 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in progress:
        lines.append(f"| {row['branch']} | {row['added_transitions']} | {row['p_stop_median']:.5f} | {row['zero_focus_fraction']:.3%} | {row['camera_entropy']:.5f} | {row['position_KL_from_fork']:.6f} | {row['conditional_four_coverage']:.6f} |")
    lines += ["", "Position KL为相同pre-Focus状态上、按fork相机概率加权的latent位置分布KL，不是像素位移。各相机位置mean/sigma逐状态保存在runtime/probes的CSV；camera top1占比见early_probes.csv。低KL只说明较少漂移，不代表覆盖改善。"]
    reference = base["subsets"]["independent"]
    lines += ["", "## 结果解释", "",
              f"独立基线Critic EV(all/terminal-only)={reference['critic']['branch_all_bootstrapped']['EV']:.6f}/{reference['critic']['branch_terminal_only']['EV']:.6f}。"]
    for branch, report in reports.items():
        result = report["subsets"]["independent"]
        gain = result["four_focus_coverage"] / reference["four_focus_coverage"] - 1
        lines.append(f"- {branch}真实四Focus覆盖相对基线变化={gain:.4%}；独立集零Focus比例={result['zero_focus_fraction']:.3%}；原reward定义Critic EV(all/terminal-only)={result['critic']['original_all_bootstrapped']['EV']:.6f}/{result['critic']['original_terminal_only']['EV']:.6f}。")
    lines += ["- A降低了每次感知的直接惩罚，但未观察到空间探索改善；固定状态STOP概率仍上升，不能把降低成本本身当作学到了有效注视。",
              "- B的STOP/Position漂移较小，但独立集terminal-only EV未改善；历史集较好的EV不能替代独立集证据，也不足以支撑A+B互补假设。",
              "- 两支当前确定性均为4 Focus，只能说明这512步尚未出现零Focus塌缩。相机/ROI重复与覆盖停滞仍存在，不能外推长期不塌缩。",
              "- 采样训练的四Focus覆盖约0.10，而确定性评估约0.035；前者包含随机探索，不是部署策略空间覆盖已改善的证据。",
              "- 独立SR配对区间均跨0：本轮可据冻结门槛拒绝扩训，但不能据此宣称A/B的真实效果已被统计证明为负。"]
    lines += ["", "## 1024扩展与A+B决策", ""]
    for branch, record in decisions.items():
        lines.append(f"- {branch}扩展资格：{record['eligible']}；未通过：{', '.join(record['failed']) or '无'}。")
    lines.append(f"- A+B互补证据：{complementary}。不自动训练组合；{proposal['next']}。")
    if not any(record["eligible"] for record in decisions.values()):
        lines += ["", "**本轮没有达到预先冻结的1024扩展门槛，不宣称找到Task2.3.3可直接扩训的稳定配置。**",
                  "若SR保持或偶有上升但重复ROI/联合覆盖没有改善，不能把它等同有效空间注视学习。保留1033作为对照，不据单次小验证选择A或B。",
                  "A/B结果支持下一步单独设计轻量重复约束诊断，而非直接上复杂ROI奖励或改Adapter。建议先纯评估‘同相机IoU≥0.8时抑制再次选择’并审计动作分布；若进入训练，必须正确记录约束后的条件概率，不能沿用未经修正的旧logprob。此建议未执行，不计为已获改进。"]
    lines += ["", "## 预算、成本与限制", "", "| 分支 | 新增train | optimizer steps | train wall秒 | train model秒 | 更新日志秒（每逻辑更新计一次） |", "| --- | --- | --- | --- | --- | --- |"]
    for branch, report in reports.items():
        lines.append(f"| {branch} | 512 | {report['new_optimizer_steps']} | {report['subsets']['train']['wall_seconds']:.2f} | {report['subsets']['train']['model_seconds']:.2f} | {report['update_seconds']:.2f} |")
    for branch, report in reports.items():
        lines.append(f"\n{branch}实际更新batch：{report['update_batch_sizes']}；新训练Focus分布：{report['subsets']['train']['focus_distribution']}；reward分量：{report['subsets']['train']['reward_components']}。")
        lines.append(f"{branch}完整run wall（含启动、probe和评估）={report['total_run_wall_seconds_including_startup_probes_evaluation']:.2f}秒；表中train wall为episode计时之和，不含模型启动/全部管理开销。")
    lines += ["", "- 新训练总计1088=64门禁+512A+512B；恢复重复计算不算新增环境样本，原1033预训练预算另列。评估transition见各run analysis。",
              "- GPU csv记录整卡采样功耗和显存；并发模型/renderer共享设备，不能把整卡能耗当某分支独占能耗。训练、评估wall和模型/感知时间分别保留。",
              "- Critic EV：独立集真实rollout，branch和原reward两套定义；终止集单报以排除truncation bootstrap循环。确定性评估与采样训练存在policy差异。",
              "- A沿用旧reward训练的Critic及optimizer动量，存在新目标适应期；B是工程包，不能归因于LR或batch单因素。",
              "- 单seed、短续训、32集不证明长期稳定或泛化；未扩展不等于该方向在更大预算下一定无效。",
              "- U只做64步工程一致性门禁，没有同预算512步不变配置训练组；A/B相对1033基线的变化，不能单独证明修复优于正常512步续训。新旧评估集难度不同，也不能仅比较两套SR就宣称过拟合。",
              "- 固定probe读取历史detached状态仅用于观测，不进入optimizer；覆盖门槛使用新独立集真实轨迹。",
              "- 原4096阈值敏感性另见threshold_summary.json；阈值修改不是策略学习成功。", "",
              "## 交付", "", "repair_decision.json、extension_eligibility.json、branch_comparison.csv、paired_evaluation.json、early_probes.csv；可复现脚本与完整原始运行见README及FILES_MANIFEST.tsv。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
