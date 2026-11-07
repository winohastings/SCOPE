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
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, gae, navigation_reward
from Experiment.SCOPE.Stage2.Task231.diagnose import interval, paired
from Experiment.SCOPE.Stage2.Task232.metrics import roi_metrics, write_csv
from Experiment.SCOPE.Stage2.Task232.novelty import coverage_novelty
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def training_reward_audit(run):
    config = read(run / "config.json")
    errors = {"reward": 0., "components": 0., "returns": 0., "advantages": 0.}
    total = 0
    seen = set()
    details = []
    for path in sorted((run / "runtime/updates").glob("update_*.json")):
        update = read(path)
        transitions = []
        for boundary_path in update["boundary_paths"]:
            assert Path(boundary_path).is_relative_to(run / "runtime")
            boundary = torch.load(boundary_path, map_location="cpu", weights_only=True)
            for trace_path in boundary["report"]["trace_paths"]:
                assert trace_path not in seen
                seen.add(trace_path)
                assert Path(trace_path).is_relative_to(run / "runtime/traces")
                trace = torch.load(trace_path, map_location="cpu", weights_only=True)
                assert trace["purpose"] == "train" and trace["mode"] == "training-sampling"
                assert trace["policy_version"] == update["old_policy_version"]
                novelty = coverage_novelty(trace["decisions"], config["coverage_beta"])
                assert novelty == trace["task232_coverage_novelty"]
                original = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                             terminated=trace["terminated"], truncated=trace["truncated"])
                reward = original + novelty["reward"]
                errors["reward"] = max(errors["reward"], abs(reward - trace["navigation_reward"]))
                errors["components"] = max(errors["components"], abs(sum(trace["task232_reward_components"].values()) - reward))
                transitions.append(NavigationTransition(trace, trace["old_value"], trace["old_next_value"], reward,
                                                        trace["transition_id"].rsplit(":", 1)[0]))
                for index, focus in enumerate(novelty["focus"]):
                    details.append({"transition_id": trace["transition_id"], "focus_index": index, **focus})
        assert len(transitions) == update["transitions"]
        advantages, returns = gae(transitions, config["gamma"], config["gae_lambda"])
        if config["normalize_advantage"]:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        targets = torch.load(run / f"runtime/updates/targets_{update['update']:04d}.pt", map_location="cpu", weights_only=True)
        errors["returns"] = max(errors["returns"], float((returns - targets["returns"]).abs().max()))
        errors["advantages"] = max(errors["advantages"], float((advantages - targets["advantages"]).abs().max()))
        total += len(transitions)
    assert total == 512 and max(errors.values()) < 1e-6
    write_csv(run / "analysis/coverage_reward_per_focus.csv", details)
    result = {"passed": True, "fresh_transitions_updated_once": total, "max_abs_errors": errors,
              "scope": "independent reconstruction from fresh traces, compared against actual saved PPO GAE targets; recovery replay is not double-counted"}
    write_json(run / "analysis/C_REWARD_AUDIT.json", result)
    return result


def matched_first(runs):
    traces = {}
    for name, run in runs.items():
        episodes = read(run / "evaluation_independent.json")
        traces[name] = {episode["episode_uid"]: torch.load(episode["training"]["trace_paths"][0], map_location="cpu", weights_only=True) for episode in episodes}
    reference = traces["C0"]
    rows = []
    for name, group in traces.items():
        assert set(group) == set(reference) and len(group) == 32
        for episode, trace in sorted(group.items()):
            base = reference[episode]
            for key in ("body_pose", "context_grids", "frame_count"):
                compare_nested(trace[key], base[key], key)
            compare_nested(trace["decisions"][0]["state"], base["decisions"][0]["state"], "pre_focus")
            selected = [(decision["action"]["camera"], decision["roi"]) for decision in trace["decisions"] if not decision["action"]["stop"]]
            progress = trace["teacher_for_reward_only"]["distance_before"] - trace["teacher_for_reward_only"]["distance_after"]
            base_progress = base["teacher_for_reward_only"]["distance_before"] - base["teacher_for_reward_only"]["distance_after"]
            rows.append({"branch": name, "episode_uid": episode, "same_initial_state": True, **roi_metrics(selected),
                         "progress_delta_vs_C0": progress - base_progress,
                         "actions_L2_vs_C0": float((trace["navigation_actions"] - base["navigation_actions"]).norm())})
    write_csv(ROOT / "C_matched_initial.csv", rows)
    return {name: {"states": 32, "progress_delta_vs_C0": interval([row["progress_delta_vs_C0"] for row in rows if row["branch"] == name]),
                   "coverage": float(np.mean([row["coverage"] for row in rows if row["branch"] == name])),
                   "repeat": float(np.mean([row["repeated_roi_fraction"] for row in rows if row["branch"] == name]))} for name in runs}


def exploration_gate(reference, candidate):
    coverage = candidate["four_focus_coverage"]
    repeat = candidate["four_focus_repeat_fraction"]
    return {"N4_coverage_gain_ge_25pct": coverage is not None and reference["four_focus_coverage"] is not None and coverage >= 1.25 * reference["four_focus_coverage"],
            "N4_repeat_drop_ge_015": repeat is not None and reference["four_focus_repeat_fraction"] is not None and repeat <= reference["four_focus_repeat_fraction"] - .15}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=Path, default=ROOT / "artifacts/branch_C512")
    parser.add_argument("--control", type=Path, default=ROOT / "artifacts/branch_C0512")
    args = parser.parse_args()
    torch.set_num_threads(2)
    runs = {"C0": args.control.resolve(), "C": args.c.resolve()}
    frozen = read(ROOT / "frozen/C_FORK_AUDIT.json")
    assert sha256(ROOT / "C_PROTOCOL.md") == frozen["protocol_sha256"]
    assert all(sha256(ROOT / name) == expected for name, expected in frozen["source_hashes"].items())
    reports, audits, comparisons, table, probes = {}, {}, {}, [], []
    for name, run in runs.items():
        assert run.is_relative_to(ROOT)
        complete = read(run / "COMPLETE.json")
        assert complete["new_training_transitions"] == 512 and complete["status"]["trained"] == 1545
        assert complete["status"]["buffer_transitions"] == 0
        assert all(complete["freeze_audit"].values()) and read(run / "RESTORE_GATE.json")["passed"]
        assert sha256(run / "config.json") == frozen["branches"][name]["config_sha256"]
        reports[name] = read(run / "analysis/summary.json")
        assert reports[name]["ratio_max_error"] < 1e-4 and not reports[name]["partial"]
        audits[name] = training_reward_audit(run)
        audits[name]["process_exit_code"] = int((run / "PROCESS_EXIT_CODE.txt").read_text())
        audits[name]["opengl_cleanup_error"] = "GL::Context::current(): no current context" in (ROOT / f"branch_{name}512.log").read_text()
        for subset in ("independent", "historical"):
            episodes = read(run / f"evaluation_{subset}.json")
            assert len(episodes) == len({row["episode_uid"] for row in episodes}) == 32
            assert all(row["failure"] is None for row in episodes)
            result = reports[name]["subsets"][subset]
            table.append({"branch": name, "subset": subset, **{key: result[key] for key in ("SR", "SPL", "focus_mean", "zero_focus_fraction", "four_focus_coverage", "four_focus_repeat_fraction")},
                          "shaped_terminal_EV": result["critic"]["branch_terminal_only"]["EV"],
                          "original_terminal_EV": result["critic"]["original_terminal_only"]["EV"]})
        for count in (0, 256, 512):
            probe = read(run / f"runtime/probes/added_{count:04d}.json")
            heads = probe["heads"]["val_seen"]
            probes.append({"branch": name, "added": count, "stop_median": heads["p_stop_P10_P50_P90"][1],
                           "zero_focus": heads["zero_focus_first_step_fraction"], "position_KL": heads["KL_from_fork"]["position"],
                           "camera_top1": heads["camera_top1_fractions"], "camera_entropy": heads["camera_entropy"],
                           "coverage_proxy": probe["geometry"]["forced4_coverage"]})
    assert read(runs["C0"] / "UNCHANGED_FORWARD_GATE.json")["passed"]
    base = ROOT / "artifacts/baseline_independent"
    for subset in ("independent", "historical"):
        control = sorted(read(runs["C0"] / f"evaluation_{subset}.json"), key=lambda row: row["episode_uid"])
        candidate = sorted(read(runs["C"] / f"evaluation_{subset}.json"), key=lambda row: row["episode_uid"])
        comparisons[f"C_minus_C0_{subset}"] = paired(control, candidate)
    reference = sorted(read(base / "evaluation_independent.json"), key=lambda row: row["episode_uid"])
    for name, run in runs.items():
        comparisons[f"{name}_minus_BASE_independent"] = paired(reference, sorted(read(run / "evaluation_independent.json"), key=lambda row: row["episode_uid"]))
    matched = matched_first({"BASE": base, **runs})
    checks = exploration_gate(reports["C0"]["subsets"]["independent"], reports["C"]["subsets"]["independent"])
    result = {"task": "Task2.3.2 new C coverage novelty", "protocol": "C_PROTOCOL.md", "beta": .01,
              "new_training": {"C": 512, "C0": 512}, "reports": reports, "audits": audits,
              "paired": comparisons, "same_initial_state": matched, "spatial_checks": checks,
              "exploration_screen_passed": all(checks.values()), "automatic_extension": False,
              "Task233_configuration_approved": False, "feature_JS_or_KL_reward_used": False,
              "adapter_retrained": False, "gate_note": "Exploration screen is distinct from navigation benefit; beta fixed before run; no automatic extra budget."}
    result["mechanism_attribution"] = read(ROOT / "C_MECHANISM.json")
    write_json(ROOT / "C_DECISION.json", result)
    write_csv(ROOT / "C_comparison.csv", table)
    write_csv(ROOT / "C_early_probes.csv", probes)
    baseline = read(base / "analysis/summary.json")["subsets"]["independent"]
    lines = ["# Task2.3.2 新C：新增覆盖奖励实验", "", "用户新增授权的独立探索分支，不是原任务文档的Adapter修复C；旧C_REVIEW保留历史语义。",
             "", "## 配置与预算", "", "C仅增加beta=0.01的新增覆盖奖励，C0为beta=0等预算对照；两者均从1033完整checkpoint分叉，各新增512条真实on-policy数据，0/256/512探针。原奖励、Actor、Adapter、PPO学习率和结构不变。",
             "奖励按四相机独立像素平面计算，每导航transition清空覆盖；逐Focus奖励之和进入原导航级GAE。没有直接KL/JS、没有新增微步advantage。空间覆盖是探索代理，不冒称真实信息增益。",
             "", "## 实测结果", "", f"1033基线独立集：SR={baseline['SR']:.3%}，SPL={baseline['SPL']:.3%}，N4覆盖={baseline['four_focus_coverage']:.6f}。",
             "", "| 分支 | 集合 | SR | SPL | mean Focus | 零Focus比例 | N4覆盖 | N4重复 | shaped/original terminal EV |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        lines.append(f"| {row['branch']} | {row['subset']} | {row['SR']:.3%} | {row['SPL']:.3%} | {row['focus_mean']:.3f} | {row['zero_focus_fraction']:.3%} | {row['four_focus_coverage']} | {row['four_focus_repeat_fraction']} | {row['shaped_terminal_EV']:.6f}/{row['original_terminal_EV']:.6f} |")
    lines += ["", "重复比例以首次Focus之后的决策为分母，同相机IoU≥0.8视为重复；覆盖按四相机总面积归一化。Critic全样本与terminal-only、MSE详见各run analysis。",
              "", "## 早期固定状态探针", "", "| 分支 | 新增步数 | STOP中位数 | 首步零Focus | camera entropy | position KL | 条件覆盖代理 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in probes:
        lines.append(f"| {row['branch']} | {row['added']} | {row['stop_median']:.6f} | {row['zero_focus']:.3%} | {row['camera_entropy']:.6f} | {row['position_KL']:.6f} | {row['coverage_proxy']:.6f} |")
    lines += ["", "条件覆盖基于冻结参考中间状态链，不代替真实rollout；各相机位置mean/sigma与top1分布见probe CSV和C_early_probes.csv。",
              "", "## 相同初态与配对导航证据", ""]
    for name, report in reports.items():
        independent = report["subsets"]["independent"]
        lines.append(f"- {name}独立集真实camera直方图={independent['camera_histogram']}；逐Focus序号0..3的平均相机概率={independent['camera_probability_mean_by_focus_index']}。这些是各自真实历史下的条件分布，不是固定其他变量的coverage消融。")
    for name, record in matched.items():
        delta = record["progress_delta_vs_C0"]
        lines.append(f"- {name}：32初态逐位匹配；首chunk覆盖={record['coverage']:.6f}，重复={record['repeat']:.6f}，进展差vsC0={delta['mean']:.6f}米，95%CI={delta['ci95']}。")
    for name, comparison in comparisons.items():
        delta = comparison["delta_SR"]
        lines.append(f"- {name}：ΔSR={delta['mean']:.3%}，95%episode-bootstrap CI={delta['ci95']}，McNemar p={comparison['exact_mcnemar_p']:.6g}。")
    lines += ["", "## 判读与后续", "", f"预设探索筛查：{checks}；通过={result['exploration_screen_passed']}。"]
    lines.append("达到空间筛查仅支持覆盖奖励改变了探索行为，不等于导航收益；本次不自动扩训。" if result["exploration_screen_passed"] else "本次未达到预设空间改善门槛；奖励实现正确也不代表确定性空间注视已经学好。短预算否定结果不证明覆盖奖励在其他系数/训练长度下一定无效；不事后调beta掩盖结果。")
    lines += ["采样训练空间探索与确定性部署行为分别报告，不将前者改善替代后者。冻结的视觉表征仍可能限制新视角的导航价值；本实验不检验或修复Adapter。",
              "同一导航级advantage分配给全部Focus选择保持原算法，不能声称已实现逐Focus精细信用分配。独立32集已成为开发诊断集，非最终未触碰测试集；单seed、小样本不证明泛化。",
              "", "## 成本、奖励与审计", ""]
    for name, report in reports.items():
        train = report["subsets"]["train"]
        lines += [f"- {name}：新增512；更新batch={report['update_batch_sizes']}；optimizer steps={report['new_optimizer_steps']}；train wall={train['wall_seconds']:.2f}s；model={train['model_seconds']:.2f}s；更新计时={report['update_seconds']:.2f}s；完整run={report['total_run_wall_seconds_including_startup_probes_evaluation']:.2f}s。",
                  f"- {name}采样Focus分布0..4={train['focus_distribution']}；采样N4覆盖={train['four_focus_coverage']}，重复={train['four_focus_repeat_fraction']}；奖励分量={train['reward_components']}。",
                  f"- {name}新轨迹奖励/GAE重算误差={audits[name]['max_abs_errors']}；进程退出码={audits[name]['process_exit_code']}。"]
    lines += ["两分支冻结/首更新恢复/ratio审计通过；C0首episode与历史前向逐位一致。C/C0均在全部结果保存后出现OpenGL context清理错误，退出码134；已核验每支512训练与64集无failure评估，不把COMPLETE误称零退出码。",
              "新增1024训练样本，加此前1088，总Task232训练新增2112；未4096重训。GPU日志为共享整卡采样，不作为分支独占能耗。",
              "", "## 交付", "", "C_DECISION.json、C_comparison.csv、C_early_probes.csv、C_matched_initial.csv；各run保留原始trace/checkpoint、逐Focus新增面积奖励和C_REWARD_AUDIT.json。"]
    lines += ["", "## 视角集中原因分析", "", "以下区分实现、奖励量级、梯度竞争与信用分配，不把未验证因素当作定论。完整结构化数据见C_MECHANISM.json，逐transition分解见C_reward_attribution.csv。", "",
              (ROOT / "C_MECHANISM.md").read_text().replace("# C视角集中：奖励与优化机制诊断", "### 机制证据").replace("\n## ", "\n### ")]
    current = reports["C"]["subsets"]["independent"]
    control = reports["C0"]["subsets"]["independent"]
    dominant = max(current["camera_histogram"], key=current["camera_histogram"].get)
    current_prob = current["camera_probability_mean_by_focus_index"]
    control_prob = control["camera_probability_mean_by_focus_index"]
    lines += ["", "### 实测归因结论", "",
              f"C主相机{dominant}在逐Focus序号上的平均概率={[probabilities[int(dominant)] for probabilities in current_prob.values()]}；C0为{[probabilities[int(dominant)] for probabilities in control_prob.values()]}。C并未随重复查看明显降低主相机概率，且比本次C0更集中；这是可观察的条件行为，不是对内部表征原因的证明。",
              "实现层面：本轮查验范围内没有发现覆盖计算或奖励未进入更新的问题；512条轨迹GAE目标重算和恢复一致性提供直接证据，不宣称排除所有潜在bug。",
              "奖励设计层面：几何函数确实鼓励新增区域，但首次Focus不区分相机，同相机新增ROI也受奖；约半数奖励不能直接区分首视角，且奖励与Focus次数高度相关，目标与‘换视角’不是同一件事。",
              "优化层面：共享导航advantage与较弱探索信号存在局限，camera输出bias梯度显示局部竞争；不能将此夸大为整个Actor的单一失败原因。无KL早停，不支持‘更新被门禁完全阻断’解释。",
              "尚不能确定：增大beta是否足够、逐Focus信用分配是否有效、coverage编码是否被忽略，需要后续独立消融。本次不追加调参、不承诺导航收益。"]
    (ROOT / "C_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"spatial_checks": checks, "exploration_screen_passed": result["exploration_screen_passed"], "paired_C_C0": comparisons["C_minus_C0_independent"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
