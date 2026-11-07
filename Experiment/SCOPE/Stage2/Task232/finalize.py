from __future__ import annotations

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


def main():
    threshold_run = ROOT / "artifacts/thresholds"
    assert (threshold_run / "COMPLETE.json").exists()
    rows = json.loads((threshold_run / "threshold_results.json").read_text())
    source = ROOT.parent / "Task23/artifacts/ppo4096"
    indices = sorted({row["manifest_index"] for row in rows})
    originals = [json.loads((source / f"episodes/val_004_{index:03d}/result.json").read_text()) | {"stop_threshold": .5, "manifest_index": index} for index in indices]
    all_rows = rows + originals
    summaries, contrasts = [], {}
    reference = sorted(originals, key=lambda row: row["episode_uid"])
    for threshold in (.45, .5, .6):
        group = sorted([row for row in all_rows if row["stop_threshold"] == threshold], key=lambda row: row["episode_uid"])
        assert len(group) == 12
        summaries.append({"threshold": threshold, "episodes": len(group),
                          **{key: float(np.mean([row[key] for row in group])) for key in ("SR", "SPL", "nDTW", "focus_mean")}})
        contrasts[str(threshold)] = paired(reference, group)
    write_json(ROOT / "threshold_summary.json", {"checkpoint": "original4096/update0056", "groups": summaries, "paired_vs_050": contrasts,
               "scope": "fixed original historical quick12; .50 historical values already independently reproduced by Task231; evaluation-only, not learning evidence"})
    write_csv(ROOT / "threshold_episodes.csv", [{"threshold": row["stop_threshold"], "episode_uid": row["episode_uid"],
                                                 **{key: row[key] for key in ("SR", "SPL", "nDTW", "focus_mean")}} for row in all_rows])
    manifest = json.loads((source.parent / "ARTIFACT_MANIFEST.json").read_text())
    prefix = str(source.relative_to(ROOT.parents[3])) + "/"
    records = [row for row in manifest["files"] if row["path"].startswith(prefix)]
    failures = [row["path"] for row in records if sha256(ROOT.parents[3] / row["path"]) != row["sha256"]]
    assert not failures, failures
    frozen = json.loads((ROOT / "frozen/FORK_AUDIT.json").read_text())
    assert sha256(ROOT.parent / "Task231/diagnosis.json") == frozen["diagnosis_sha256"]
    assert sha256(ROOT.parent / "Task231/REPORT.md") == frozen["diagnosis_report_sha256"]
    actor_contract = json.loads((source / "runtime/model_contract.json").read_text())
    actor_source_failures = [path for path, expected in actor_contract["source_hashes"].items() if sha256(path) != expected]
    assert not actor_source_failures, actor_source_failures
    write_json(ROOT / "SOURCE_INTEGRITY.json", {"original_Task23_files_checked": len(records), "failures": failures,
                                               "Task231_diagnosis_report_unchanged": True,
                                               "actor_source_files_checked": len(actor_contract["source_hashes"]), "actor_source_failures": actor_source_failures})
    report = ROOT / "REPORT.md"
    text = report.read_text()
    marker = "\n## 原模型STOP阈值纯评估\n"
    text = text.split(marker)[0]
    execution_health = {}
    for name in ("branch_A512", "branch_B512", "thresholds"):
        run = ROOT / "artifacts" / name
        complete = json.loads((run / "COMPLETE.json").read_text())
        expected = {"threshold_results.json": 24} if name == "thresholds" else {"evaluation_independent.json": 32, "evaluation_historical.json": 32}
        counts = {}
        for filename, count in expected.items():
            episodes = json.loads((run / filename).read_text())
            assert len(episodes) == count
            assert all(episode.get("failure") is None for episode in episodes)
            assert len({(episode["episode_uid"], episode.get("stop_threshold")) for episode in episodes}) == count
            counts[filename] = len(episodes)
        assert complete["new_training_transitions"] == (0 if name == "thresholds" else 512)
        log = (ROOT / f"{name}.log").read_text()
        execution_health[name] = {"complete_artifact_present": True, "evaluation_counts": counts,
                                  "new_training_transitions": complete["new_training_transitions"],
                                  "opengl_context_error_in_log": "GL::Context::current(): no current context" in log,
                                  "note": "COMPLETE is written before cleanup; a cleanup failure does not imply a clean process exit. Evaluation count and failure fields independently checked."}
    write_json(ROOT / "EXECUTION_HEALTH.json", execution_health)
    text += marker + "\n| STOP阈值 | 12集SR | SPL | mean Focus |\n| --- | --- | --- | --- |\n"
    for row in summaries:
        text += f"| {row['threshold']} | {row['SR']:.3%} | {row['SPL']:.3%} | {row['focus_mean']:.3f} |\n"
    text += "\n该12集结果仅说明阈值敏感性，未改变模型参数，不当作策略学习改善。原Task23全部归档产物与Task231诊断证据最终完整性复核通过。\n"
    text += "\n### 进程退出与产物完整性\n\n"
    for name, health in execution_health.items():
        text += f"- {name}：评估记录数={health['evaluation_counts']}，新增训练={health['new_training_transitions']}；日志OpenGL context清理异常={health['opengl_context_error_in_log']}。\n"
    text += "\nCOMPLETE在清理前落盘，不等同进程零退出码。本次执行器观测到A/B均在全部结果保存后退出码134，日志末尾为OpenGL context错误；阈值评估进程退出码0。未重跑训练，也未掩盖此异常。所有预期评估记录、无episode failure及预算另行核验，见EXECUTION_HEALTH.json。\n"
    matched_path = ROOT / "matched_initial_states.json"
    if matched_path.exists():
        matched = json.loads(matched_path.read_text())
        text += "\n## 独立集相同初态真实Focus对照\n\n"
        text += "此补充不是固定参考中间状态代理：直接配对独立集相同reset初态，逐位核验pre-Focus tensor/姿态/context grid，再比较实际顺序编码Focus的结果。仅解释首chunk，不外推整集因果。\n\n"
        text += "| 分支 | 有效同态数 | 实际覆盖 | 重复比例 | 首chunk进展差均值/95%CI（米） |\n| --- | --- | --- | --- | --- |\n"
        for branch, result in matched["results"].items():
            text += f"| {branch} | {result['valid_states']} | {result['coverage']:.6f} | {result['repeated_roi_fraction']:.6f} | {result['progress_delta']['mean']:.6f} / {result['progress_delta']['ci95']} |\n"
    decision_path = ROOT / "repair_decision.json"
    decision = json.loads(decision_path.read_text())
    decision["threshold_diagnostic"] = json.loads((ROOT / "threshold_summary.json").read_text())
    decision["source_integrity"] = json.loads((ROOT / "SOURCE_INTEGRITY.json").read_text())
    decision["execution_health"] = execution_health
    if matched_path.exists():
        decision["matched_initial_states"] = json.loads(matched_path.read_text())
    c_review = ROOT / "C_REVIEW.json"
    if c_review.exists():
        decision.pop("C", None)
        decision["original_document_adapter_C_review"] = json.loads(c_review.read_text())
        text += "\n## 原文Adapter C条件分支审查\n\n"
        text += "C修复未执行：现有H3证据不满足直接负面证据前提。已从128条原始分支记录重算配对统计并核验16个既有Glance状态，详见C_REVIEW.md/C_REVIEW.json。该项是适用性审查完成，不是Adapter修复完成；A/B未改善不能代替H3证据。\n"
    novelty_decision = ROOT / "C_DECISION.json"
    if novelty_decision.exists():
        novelty = json.loads(novelty_decision.read_text())
        decision["new_C_coverage_novelty"] = novelty
        decision["total_fresh_training_including_new_C_control"] = 2112
        text += "\n## 用户新增C：覆盖奖励探索（已执行）\n\n"
        text += "该C不同于原文Adapter修复。新增覆盖奖励beta=0.01与beta=0的C0等预算对照均已完成512新on-policy transitions，详细结果见C_REPORT.md、C_DECISION.json。未加入KL/JS，未改Actor/Adapter，未组合A/B。\n"
        text += f"空间探索筛查通过={novelty['exploration_screen_passed']}；探索变化不等同导航收益，不自动扩训。Task232累计新增训练2112=原1088+C512+C0共512。\n"
    write_json(decision_path, decision)
    report.write_text(text)
    files = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(ROOT.rglob("*")) if path.is_file() and path.name != "FILES_MANIFEST.tsv"]
    with (ROOT / "FILES_MANIFEST.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"), delimiter="\t")
        writer.writeheader()
        writer.writerows(files)
    print(json.dumps({"thresholds": summaries, "files": len(files), "original_files_unchanged": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
