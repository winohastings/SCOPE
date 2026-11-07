from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, gae, navigation_reward
from Experiment.SCOPE.Stage2.Task232.metrics import write_csv
from Experiment.SCOPE.Stage2.Task232.novelty import coverage_novelty
from Experiment.SCOPE.Stage2.data_guard import write_json


def safe_correlation(first, second):
    if np.std(first) < 1e-10 or np.std(second) < 1e-10:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def validate_gradient_formula(run):
    actor = SpatialQueryActor().eval()
    actor.load_state_dict(torch.load(run / "runtime/snapshots/actor_0014.pt", map_location="cpu", weights_only=False)["actor"])
    report = json.loads((run / "runtime/updates/update_0015.json").read_text())
    errors = []
    for boundary_path in report["boundary_paths"]:
        boundary = torch.load(boundary_path, map_location="cpu", weights_only=True)
        for trace_path in boundary["report"]["trace_paths"]:
            trace = torch.load(trace_path, map_location="cpu", weights_only=True)
            for decision in trace["decisions"]:
                if decision["action"]["stop"]:
                    continue
                distribution = actor(ActorState.unpack(decision["state"], "cpu"))
                loss = -distribution.camera.log_prob(torch.tensor(decision["action"]["camera"]))
                gradient = torch.autograd.grad(loss, actor.camera_head[-1].bias)[0]
                expected = torch.as_tensor(decision["old_distribution"]["camera_probabilities"]).clone()
                expected[decision["action"]["camera"]] -= 1
                errors.append(float((gradient - expected).abs().max()))
            if len(errors) >= 16:
                break
        if len(errors) >= 16:
            break
    assert max(errors) < 1e-6
    return {"passed": True, "decisions": len(errors), "max_error": max(errors),
            "scope": "actual update14 camera output bias autograd vs stored probability score; no optimizer update"}


def attribute(run):
    config = json.loads((run / "config.json").read_text())
    updates, rows = [], []
    camera_count = np.zeros(4, dtype=int)
    first_camera_count = np.zeros(4, dtype=int)
    pure_first, pure_later, pure_later_full = [], [], []
    beta = .01
    for path in sorted((run / "runtime/updates").glob("update_*.json")):
        update = json.loads(path.read_text())
        transitions, delta_transitions, score_rows, local_rows = [], [], [], []
        for boundary_path in update["boundary_paths"]:
            boundary = torch.load(boundary_path, map_location="cpu", weights_only=True)
            for trace_path in boundary["report"]["trace_paths"]:
                trace = torch.load(trace_path, map_location="cpu", weights_only=True)
                reward = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                           terminated=trace["terminated"], truncated=trace["truncated"])
                novelty = coverage_novelty(trace["decisions"], beta)
                altered = copy.deepcopy(trace["decisions"])
                next_camera = 0
                for decision in altered:
                    if not decision["action"]["stop"]:
                        decision["action"]["camera"] = next_camera
                        next_camera += 1
                diverse = coverage_novelty(altered, beta)["reward"]
                score = np.zeros(4)
                selected = []
                for decision in trace["decisions"]:
                    if decision["action"]["stop"]:
                        continue
                    camera = decision["action"]["camera"]
                    camera_count[camera] += 1
                    selected.append(camera)
                    score -= np.asarray(decision["old_distribution"]["camera_probabilities"])
                    score[camera] += 1
                if selected:
                    first_camera_count[selected[0]] += 1
                score_rows.append(score)
                first_reward = novelty["focus"][0]["reward"] if novelty["focus"] else 0.
                later = [row["reward"] for row in novelty["focus"][1:]]
                pure_first.append(first_reward)
                pure_later.extend(later)
                pure_later_full.extend(row["novelty"] >= .95 for row in novelty["focus"][1:])
                key = trace["transition_id"].rsplit(":", 1)[0]
                transitions.append(NavigationTransition(trace, trace["old_value"], trace["old_next_value"], reward, key))
                delta_transitions.append(NavigationTransition(trace, 0., 0., novelty["reward"], key))
                local_rows.append({"branch": config["task232_branch"], "update": update["update"], "transition_id": trace["transition_id"],
                                   "focus_count": trace["focus_count"], "unique_cameras": len(set(selected)),
                                   "original_reward": reward, "bonus_at_beta_001": novelty["reward"],
                                   "first_focus_bonus": first_reward, "later_focus_bonus": sum(later),
                                   "counterfactual_four_distinct_camera_bonus": diverse,
                                   "counterfactual_gain_same_count_same_roi": diverse - novelty["reward"],
                                   "progress_reward": trace["task232_reward_components"]["progress"],
                                   "terminal_reward": trace["task232_reward_components"]["terminal"]})
        base_adv, _ = gae(transitions, config["gamma"], config["gae_lambda"])
        bonus_adv, _ = gae(delta_transitions, config["gamma"], config["gae_lambda"])
        shaped_adv = base_adv + bonus_adv
        scale = float(shaped_adv.std(unbiased=False).clamp_min(1e-8))
        base_centered = (base_adv - base_adv.mean()).numpy() / scale
        bonus_centered = (bonus_adv - bonus_adv.mean()).numpy() / scale
        score_array = np.asarray(score_rows)
        base_gradient = -(base_centered[:, None] * score_array).mean(axis=0)
        bonus_gradient = -(bonus_centered[:, None] * score_array).mean(axis=0)
        denominator = np.linalg.norm(base_gradient) * np.linalg.norm(bonus_gradient)
        for index, row in enumerate(local_rows):
            row.update(original_advantage=float(base_adv[index]), bonus_advantage=float(bonus_adv[index]),
                       centered_bonus_contribution=float(bonus_centered[index]),
                       shared_shaped_advantage=float(base_centered[index] + bonus_centered[index]))
        rows.extend(local_rows)
        updates.append({"branch": config["task232_branch"], "update": update["update"], "transitions": len(transitions),
                        "base_adv_std": float(base_adv.std(unbiased=False)), "bonus_adv_std": float(bonus_adv.std(unbiased=False)),
                        "base_camera_output_bias_loss_gradient": base_gradient.tolist(),
                        "bonus_camera_output_bias_loss_gradient": bonus_gradient.tolist(),
                        "base_bonus_camera_gradient_cosine": float(np.dot(base_gradient, bonus_gradient) / denominator) if denominator else None,
                        "bonus_base_camera_gradient_norm_ratio": float(np.linalg.norm(bonus_gradient) / max(np.linalg.norm(base_gradient), 1e-10)),
                        "kl_early_stop": update["kl_early_stop"],
                        "mean_logged_clip_fraction": float(np.mean([batch["clip_fraction"] for batch in update["minibatches"]]))})
    assert len(rows) == 512
    total_bonus = sum(row["bonus_at_beta_001"] for row in rows)
    result = {"branch": config["task232_branch"], "transitions": len(rows), "camera_counts": camera_count.tolist(),
              "first_camera_counts": first_camera_count.tolist(), "geometric_bonus_at_beta_001": total_bonus,
              "actual_beta": config["coverage_beta"], "first_focus_share_of_bonus": sum(pure_first) / total_bonus,
              "later_focus_novelty_ge_095_fraction": float(np.mean(pure_later_full)),
              "sum_absolute_original_reward": sum(abs(row["original_reward"]) for row in rows),
              "sum_absolute_progress_reward": sum(abs(row["progress_reward"]) for row in rows),
              "sum_absolute_terminal_reward": sum(abs(row["terminal_reward"]) for row in rows),
              "bonus_focus_count_correlation": safe_correlation([row["bonus_at_beta_001"] for row in rows], [row["focus_count"] for row in rows]),
              "counterfactual_mean_bonus_gain_same_count_distinct_cameras": float(np.mean([row["counterfactual_gain_same_count_same_roi"] for row in rows])),
              "updates": updates,
              "gradient_scope": "Exact initial-ratio-one unclipped surrogate gradient for camera_head final output bias only, summed over chosen decisions and averaged by navigation states; base and bonus centered with common shaped-advantage std. Excludes entropy, shared trunk, later minibatches, clipping and Adam history. C0 bonus is counterfactual diagnostic, not actual training.",
              "counterfactual_scope": "geometry-only relabeling to distinct cameras, same number/coordinates; not a navigation intervention or off-policy training"}
    return result, rows


def main():
    torch.set_num_threads(2)
    formula = validate_gradient_formula(ROOT / "artifacts/branch_C512")
    write_json(ROOT / "C_GRADIENT_FORMULA_AUDIT.json", formula)
    results, rows = {}, []
    for branch in ("C", "C0"):
        result, current = attribute(ROOT / f"artifacts/branch_{branch}512")
        results[branch] = result
        rows.extend(current)
    synthetic = []
    for name, cameras in (("same_roi_same_camera", [1, 1, 1, 1]), ("same_roi_four_cameras", [0, 1, 2, 3])):
        decisions = [{"action": {"stop": False, "camera": camera}, "roi": [160, 160, 352, 352]} for camera in cameras]
        bonus = coverage_novelty(decisions, .01)["reward"]
        synthetic.append({"case": name, "bonus": bonus, "cost_plus_bonus": bonus - .04})
    write_json(ROOT / "C_MECHANISM.json", {"branches": results, "synthetic_geometry_checks": synthetic, "gradient_formula_audit": formula,
               "not_established": ["beta is causally too small", "coverage encoder ignores history", "larger beta would improve navigation", "micro-step credit would fix the issue"]})
    write_csv(ROOT / "C_reward_attribution.csv", rows)
    lines = ["# C视角集中：奖励与优化机制诊断", "", "本分析不进行训练；实际空间结果见C_REPORT。几何反事实、固定Critic GAE分解和输出bias梯度用于定位，不等同重新训练因果实验。",
             "", "## 1. 奖励计算与接入", "", "相同ROI同相机四次bonus=0.01，四相机各一次bonus=0.04；扣除原Focus成本后分别为-0.03和0。奖励确实偏好新增区域，没有把不同相机相同坐标错误地视为重叠。实际512条轨迹的奖励/GAE/normalized advantage重算见C_REWARD_AUDIT。",
             "", "## 2. 奖励比重与样本区分度", ""]
    for branch, record in results.items():
        lines.append(f"- {branch}：实际beta={record['actual_beta']}；按beta=.01计几何bonus={record['geometric_bonus_at_beta_001']:.6f}，原reward绝对值总和={record['sum_absolute_original_reward']:.6f}，progress绝对值总和={record['sum_absolute_progress_reward']:.6f}，terminal绝对值总和={record['sum_absolute_terminal_reward']:.6f}；首次Focus占bonus={record['first_focus_share_of_bonus']:.3%}；后续Focus新颖性≥.95比例={record['later_focus_novelty_ge_095_fraction']:.3%}；bonus与Focus次数相关={record['bonus_focus_count_correlation']}。")
    lines += ["首次Focus无论选哪台相机都是全新区域，不能提供首相机偏好的相对信号。空间奖励不直接奖励相机多样性，同相机不重叠ROI同样可以得满分；不能将它等同于‘必须换相机’奖励。",
              "beta=.01只让完全新增ROI的直接感知净成本为0，不提供额外正净感知收益；仍有导航step成本。单次最大0.01对比成功终止2.0，奖励累计和GAE才决定更新，不能仅凭这个比例就认定beta必然太小。",
              "", "## 3. 信用分配与更新方向", "", "| 分支 | update | base A std | bonus A std | 相机bias梯度cosine | bonus/base梯度norm | clip均值 | KL早停 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for branch, record in results.items():
        for update in record["updates"]:
            lines.append(f"| {branch} | {update['update']} | {update['base_adv_std']:.6f} | {update['bonus_adv_std']:.6f} | {update['base_bonus_camera_gradient_cosine']} | {update['bonus_base_camera_gradient_norm_ratio']:.6f} | {update['mean_logged_clip_fraction']:.6f} | {update['kl_early_stop']} |")
    current = results["C"]["updates"]
    opposing = sum(update["base_bonus_camera_gradient_cosine"] is not None and update["base_bonus_camera_gradient_cosine"] < 0 for update in current)
    promotes_left = sum(update["bonus_camera_output_bias_loss_gradient"][1] < 0 for update in current)
    ratios = [update["bonus_base_camera_gradient_norm_ratio"] for update in current]
    lines += [f"", f"C的{len(current)}次更新中，{opposing}次在camera输出bias投影上bonus与原奖励梯度相反；bonus梯度norm仅为原奖励的{min(ratios):.2%}–{max(ratios):.2%}。这支持局部竞争且探索信号较弱，但不是全部参数的因果归因。",
              f"其中{promotes_left}/{len(current)}次bonus自身的camera-1 loss bias梯度为负，按简单梯度下降反而会抬高既有主视角bias。说明‘奖励了新增覆盖’不保证每批聚合梯度都降低重复主视角；共享序列信用、后续回报和样本相关性可以改变方向。",
              f"KL早停次数={sum(update['kl_early_stop'] for update in current)}；没有证据称优化器被KL门禁完全阻断。"]
    lines += ["相机bias的四维梯度见JSON，正loss梯度在简单梯度下降下会降低该相机bias；此处不是Adam真实参数位移，不能忽略共享trunk、熵和后续minibatch。C0的bonus梯度仅为反事实，不曾用于其训练。",
              "所有Focus仍共享导航级advantage：某一步的新覆盖奖励会连带给同一序列的首相机和重复动作信用；归一化后部分有正bonus的序列仍可得到负中心化贡献。这是保留原算法的信用分配限制，不是奖励未计算。",
              "若两个梯度cosine为负，可说在该输出bias投影上存在竞争；不能据此认定全部Actor梯度或整个训练都受抑制。clip/KL日志用于排查更新限制，不把非零clipping自动视作故障。",
              "", "## 4. 确定性解码与未识别因素", "", "训练相机采样具有多视角并不保证argmax换相机；固定probe概率/熵/位置KL与真实确定性N4覆盖必须同时看。输出仍偏同相机，也可能是短预算尚未学到历史覆盖条件化，不能凭输入中存在coverage就认定已使用。",
              "本实验只测试beta=.01、512步、单seed，未做beta消融、冻结导航reward消融或逐Focus信用分配对照，不能唯一归因‘比重太小’、‘导航回报抑制’或‘Actor无法利用coverage’。如需后续验证，应单独冻结一个因素，不直接上JS/KL或叠加多项奖励。"]
    (ROOT / "C_MECHANISM.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({branch: {key: value for key, value in record.items() if key in ("geometric_bonus_at_beta_001", "first_focus_share_of_bonus", "later_focus_novelty_ge_095_fraction", "camera_counts")} for branch, record in results.items()}))


if __name__ == "__main__":
    main()
