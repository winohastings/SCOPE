from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task231.summarize import matched_summary
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def main():
    source = ROOT.parent / "Task231"
    paths = [source / name for name in ("diagnosis.json", "REPORT.md", "artifacts/matched8/matched_results.json", "artifacts/matched8/COMPLETE.json")]
    with (source / "FILES_MANIFEST.tsv").open() as stream:
        manifest = {row["path"]: row for row in csv.DictReader(stream, delimiter="\t")}
    hashes = {}
    for path in paths:
        relative = str(path.relative_to(source))
        hashes[relative] = sha256(path)
        assert hashes[relative] == manifest[relative]["sha256"], relative
    flat, _, _, summary = matched_summary(paths[2])
    original = json.loads(paths[0].read_text())["matched"]
    contrasts = summary["episode_clustered_progress_contrasts_meters"]
    assert contrasts == original["episode_clustered_progress_contrasts_meters"]
    assert summary["chunks"] == summary["valid_chunks"] == 128
    assert summary["states"] == 16
    glance = [row for row in flat if row["variant"] == "none"]
    assert len(glance) == 16
    assert all(row["focus_count"] == 0 and row["token_change_mse"] == 0 and row["teacher_mse_glance"] == row["teacher_mse_fused"] for row in glance)
    assert all(row["focus_count"] == 4 for row in flat if row["variant"] != "none")
    negative = [name for name, result in contrasts.items() if result["ci95"][1] < 0]
    result = {"task": "Task2.3.2 C", "source_sha256": hashes, "evidence": summary,
              "glance_unchanged_existing_states": len(glance), "negative_interval_contrasts": negative,
              "status": "requires_targeted_review" if negative else "not_triggered_insufficient_H3_evidence",
              "repair_executed": False, "new_training_transitions": 0, "new_environment_evaluations": 0,
              "interpretation": "Existing-data review, not a new intervention or proof that Adapter is harmless. All reported paired intervals span zero; failed A/B does not establish H3. Original user constraints retain fusion and forbid Adapter retraining.",
              "gate_scope": "Retrospective conservative evidence review, not a preregistered hypothesis test. A negative interval alone would still not identify Adapter as the cause."}
    write_json(ROOT / "C_REVIEW.json", result)
    lines = ["# Task2.3.2 C 条件分支审查", "", "C的Adapter/视觉融合修复没有执行。此前A/B报告未明确交代C状态，本次补齐条件审查。",
             "", "## 原计划与边界", "", "原计划仅在H3有直接负面证据时启动C，不是要求无条件依次训练A/B/C。用户补充要求保留视觉融合并禁止重训Adapter；不能把A/B未改善等同于Adapter有害。",
             "", "## 本次实际执行", "", "重新核验Task231证据SHA，从128条原始分支记录重算episode-cluster bootstrap；16状态全部恢复有效，所有Focus分支预算均为4。没有新增环境采样或optimizer更新。",
             "", "| 对比 | 进展差均值（米） | 95%区间 |", "| --- | --- | --- |"]
    for name, contrast in contrasts.items():
        lines.append(f"| {name} | {contrast['mean']:.6f} | {contrast['ci95']} |")
    lines += ["", f"复核结果：{result['status']}。16个既有无Focus状态的token变化为0，融合前后teacher-MSE相同；这不是未执行修复的Glance验收，也不是全分布无退化证明。",
              "", "## 处置", "", "本轮C记为条件未触发，而不是修复成功或实验完成。没有充分证据选择Adapter、gate或融合修复。保持原模型不变，不开展无依据重校准。",
              "A/B确定性重复ROI是策略空间行为证据，不单独归因Adapter；轻量重复约束属于独立后续诊断，不冒称原计划C。",
              "若后续更广泛matched-budget实验得到稳定负面结果，再独立冻结C配置并验证一步导航效果与Glance无退化；须先明确是否解除当前视觉融合/Adapter限制。",
              "Task232受控筛选及C适用性审查已结束，但尚无合格Task233扩训配置。"]
    (ROOT / "C_REVIEW.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "states": summary["states"], "branches": summary["chunks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
