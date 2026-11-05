from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import datetime
from pathlib import Path

import numpy as np

from compile_configs import ROOT, sha256, validate_freeze, write_json
from policies import MODES, TOKENS, select_pair

METRICS = ("SR", "OSR", "NE", "SPL", "nDTW", "path_length", "steps_taken")
COSTS = ("vision_current_ms", "vision_history_ms", "vision_combined_ms", "selector_ms", "llm_ms", "dit_b_ms",
         "preprocess_build_wall_ms", "model_wall_ms", "rpc_wall_ms", "render_observation_wall_ms",
         "sim_step_combined_ms", "policy_cycle_wall_ms", "current_visual_tokens", "history_visual_tokens",
         "peak_gpu_memory_bytes", "peak_reserved_bytes", "estimated_current_vision_gflops",
         "estimated_history_vision_gflops")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def vision_flops(grid, config):
    patches = math.prod(grid)
    width = config["hidden_size"]
    merge = config["spatial_merge_size"] ** 2
    patch_projection = 2 * patches * config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2 * width
    projections = 8 * patches * width ** 2
    mlp = 4 * patches * width * config["intermediate_size"]
    attention = 4 * grid[0] * (grid[1] * grid[2]) ** 2 * width
    merger = 2 * (patches // merge) * ((merge * width) ** 2 + merge * width * config["out_hidden_size"])
    return (patch_projection + config["depth"] * (projections + mlp + attention) + merger) / 1e9


def augment_costs(record, config):
    estimates = {"current": 0.0, "history": 0.0}
    mixed = False
    for call in record.get("encoder_calls", []):
        kinds = call.get("online_kinds") or [call["kind"]] * len(call["grids"])
        if len(kinds) != len(call["grids"]):
            raise ValueError("encoder image/grid attribution mismatch")
        mixed = mixed or len(set(kinds)) > 1
        for kind, grid in zip(kinds, call["grids"]):
            estimates[kind] += vision_flops(grid, config)
    record["mixed_current_history_vision_call"] = mixed
    record["estimated_current_vision_gflops"] = estimates["current"]
    record["estimated_history_vision_gflops"] = estimates["history"]
    record["flops_status"] = "shape-based estimate, incomplete elementwise coverage, not measured FLOPs"


def bar_plot(path, title, values, unit):
    maximum = max(values.values()) or 1.0
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350" viewBox="0 0 760 350">',
                '<rect width="760" height="350" fill="white"/>',
                f'<text x="30" y="35" font-size="20">{html.escape(title)}</text>']
    for index, (group, value) in enumerate(values.items()):
        coordinate = 110 + index * 160
        height = 210 * value / maximum
        elements.extend([f'<rect x="{coordinate}" y="{285-height}" width="85" height="{height}" fill="{("#4c78a8", "#72b7b2", "#f58518", "#e45756")[index]}"/>',
                         f'<text x="{coordinate}" y="310" font-size="16">{group}</text>',
                         f'<text x="{coordinate}" y="{275-height}" font-size="14">{value:.3f}</text>'])
    elements.extend([f'<text x="30" y="335" font-size="12">{html.escape(unit)}; single seed, development set, descriptive only</text>', '</svg>'])
    path.write_text("\n".join(elements))


def table(rows, fields):
    def format_value(value):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return "| " + " | ".join(fields) + " |\n|" + "---|" * len(fields) + "\n" + "\n".join(
        "| " + " | ".join(format_value(row.get(field, "NA")) for field in fields) + " |" for row in rows)


def gpu_accounting():
    samples = {0: [], 1: []}
    with (ROOT / "gpu_samples.csv").open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if len(row) != 7:
                continue
            timestamp = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            device = int(row[1].strip())
            memory = float(row[3].split()[0])
            utilization = float(row[4].split()[0])
            samples[device].append((timestamp, memory, utilization))
    devices = []
    for device, values in samples.items():
        if not values:
            continue
        baseline = values[0][1]
        occupied_seconds, utilization_seconds = 0.0, 0.0
        intervals = []
        for previous, current in zip(values, values[1:]):
            duration = current[0] - previous[0]
            if duration < 0:
                raise ValueError("non-monotonic GPU sampling clock")
            intervals.append(duration)
            occupied_seconds += duration * int(previous[1] > baseline + 64)
            utilization_seconds += duration * previous[2] / 100
        devices.append({"gpu": device, "baseline_mib": baseline, "peak_device_mib": max(row[1] for row in values),
                        "sample_count": len(values), "interval_p50_ms": float(np.percentile(intervals, 50)) * 1000,
                        "interval_p90_ms": float(np.percentile(intervals, 90)) * 1000,
                        "occupied_device_hours_estimate": occupied_seconds / 3600,
                        "utilization_weighted_device_hours_estimate": utilization_seconds / 3600})
    write_json(ROOT / "GPU_ACCOUNTING.json", {"devices": devices,
               "definition": "sample-based device-level estimates; occupancy > initial baseline +64MiB, utilization includes desktop; not process-exact kernel time"})
    return devices


def aggregate(allow_incomplete=False):
    manifest = validate_freeze()
    episodes = json.loads((ROOT / "DEV_EPISODES.json").read_text())
    architecture = json.loads((ROOT / "VISION_ARCHITECTURE.json").read_text())["config"]
    episode_rows, step_rows, aggregate_rows, checks = [], [], [], []
    for group, mode in MODES.items():
        run = ROOT / "runs" / group
        paths = sorted(run.glob("episode_*.json"))
        if not allow_incomplete and len(paths) != 64:
            raise ValueError(f"{group}: expected 64 episode artifacts, got {len(paths)}")
        records = read_jsonl(run / "steps.jsonl")
        for path in sorted(run.glob("steps_shard_*.jsonl")):
            records.extend(read_jsonl(path))
        records.sort(key=lambda record: (record["ordinal"], record["step"]))
        group_episodes = []
        for ordinal, episode in enumerate(episodes):
            path = run / f"episode_{ordinal:03d}.json"
            if not path.exists():
                if allow_incomplete:
                    continue
                raise FileNotFoundError(path)
            result = json.loads(path.read_text())
            if (result["source_episode_id"], result["ordinal"], result["session_seed"]) != (episode["source_episode_id"], ordinal, 42 + ordinal):
                raise ValueError("episode denominator/order/seed mismatch")
            eval_path = run / "logs" / episode["scene_id"] / "scope_feasibility_dev" / episode["source_episode_id"] / "eval_info.json"
            metrics = json.loads(eval_path.read_text())["metrics"] if eval_path.exists() else {}
            row = {"group": group, "ordinal": ordinal, "episode_id": result["source_episode_id"],
                   "scene_id": result["scene_id"], "run_seed": 42, "session_seed": 42 + ordinal,
                   "failure": result.get("failure"), "failure_type": result.get("failure_type"),
                   "termination_reason": result.get("termination_reason"),
                   "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                   "episodes_sha256": manifest["episodes_sha256"],
                   **{metric: metrics.get(metric) for metric in METRICS},
                   **{key: result.get(key) for key in ("episode_wall_inclusive_ms", "reset_scene_combined_ms",
                                                      "initial_render_ms", "artifact_io_ms", "prediction_count")}}
            group_episodes.append(row)
            episode_rows.append(row)
        for record in records:
            ordinal = record["ordinal"]
            original_cycle = run / "logs" / episodes[ordinal]["scene_id"] / "scope_feasibility_dev" / record["episode_id"] / f"scope_{record['step']:05d}.json"
            if original_cycle.exists():
                record["model_rpc_to_action_return_wall_ms"] = record["policy_cycle_wall_ms"]
                record["policy_cycle_wall_ms"] = json.loads(original_cycle.read_text())["total_ms"]
                record["cycle_timing_source"] = "existing benchmark worker: before prepare through waypoint return"
            else:
                record["cycle_timing_source"] = "RPC-to-return fallback; original scope absent"
            if record["episode_id"] != episodes[ordinal]["source_episode_id"] or record["session_seed"] != 42 + ordinal:
                raise ValueError("step episode/seed mismatch")
            pair, digest = select_pair(mode, seed=42 + ordinal, episode_id=record["episode_id"], step=record["step"])
            if record["current_visual_tokens"] != TOKENS[group] or record["selection_trace"] != pair:
                raise ValueError("token/ROI policy mismatch")
            if record["history_visual_tokens"] != sum(block["tokens"] for block in record["blocks"] if block["is_history"]):
                raise ValueError("history token mismatch")
            if record["history_contract"] != "task1_compatible_256_pool4_BATS":
                raise ValueError("history protocol mismatch")
            augment_costs(record, architecture)
            step_rows.append(record)
        if sum(row["prediction_count"] or 0 for row in group_episodes) != len(records):
            raise ValueError("missing prediction/STOP cost logs")
        for row in group_episodes:
            episode_records = [record for record in records if record["ordinal"] == row["ordinal"]]
            for key in ("model_wall_ms", "rpc_wall_ms", "render_observation_wall_ms", "policy_cycle_wall_ms",
                        "estimated_current_vision_gflops", "estimated_history_vision_gflops"):
                row[key + "_total"] = sum(record.get(key, 0.0) or 0.0 for record in episode_records)
            row["render_with_reset_observation_ms_total"] = row["render_observation_wall_ms_total"] + (row["initial_render_ms"] or 0.0)
        failures = sum(bool(row["failure"]) for row in group_episodes)
        complete = len(paths) == 64 and not failures
        summary = {"group": group, "planned": 64, "recorded": len(paths), "failures": failures,
                   "status": "complete" if complete else "incomplete", "prediction_count": len(records),
                   "current_visual_tokens": TOKENS[group], "current_token_reduction_vs_G0": 1 - TOKENS[group] / 256}
        for metric in METRICS:
            values = [row[metric] for row in group_episodes if row[metric] is not None and not row["failure"]]
            summary[metric + "_valid_mean"] = statistics.mean(values) if values else None
            if metric in ("SR", "OSR", "SPL", "nDTW"):
                summary[metric] = sum(values) / 64 if len(paths) == 64 else None
        for field in COSTS:
            values = [record[field] for record in records if isinstance(record.get(field), (int, float))]
            summary[field + "_n"] = len(values)
            summary[field + "_mean"] = statistics.mean(values) if values else None
            summary[field + "_max"] = max(values) if values else None
            for quantile, probability in (("p50", 50), ("p90", 90)):
                summary[field + "_" + quantile] = float(np.percentile(values, probability)) if values else None
        for key in ("model_wall_ms", "rpc_wall_ms", "render_observation_wall_ms", "policy_cycle_wall_ms"):
            summary[key + "_all_episodes_total_s"] = sum(row[key + "_total"] for row in group_episodes) / 1000
        summary["episode_wall_total_s"] = sum(row["episode_wall_inclusive_ms"] or 0 for row in group_episodes) / 1000
        aggregate_rows.append(summary)
        checks.append({"group": group, "count": len(paths), "failures": failures,
                       "token_roi_seed_history_stop_logs_validated": True, "complete": complete})
    write_csv(ROOT / "RESULTS_PER_EPISODE.csv", episode_rows)
    write_csv(ROOT / "RESULTS_PER_STEP.csv", step_rows)
    write_csv(ROOT / "RESULTS_AGGREGATED.csv", aggregate_rows)
    write_json(ROOT / "CONSISTENCY_CHECKS.json", {"groups": checks,
               "all_complete": all(row["complete"] for row in checks),
               "mixed_vision_calls": sum(row["mixed_current_history_vision_call"] for row in step_rows),
               "step_records": len(step_rows), "episode_records": len(episode_rows)})
    pilot = read_jsonl(ROOT / "pilot_steps.jsonl")
    if (ROOT / "PILOT.json").exists():
        pilot = json.loads((ROOT / "PILOT.json").read_text())
    pilot_rows = []
    for group in MODES:
        timed = [row for row in pilot if row["group"] == group and row["measurement_phase"] == "timed"]
        summary = {"group": group, "timed": len(timed),
                   "warmup": sum(row["group"] == group and row["measurement_phase"] == "warmup" for row in pilot)}
        for field in ("vision_current_ms", "vision_history_ms", "llm_ms", "dit_b_ms", "model_wall_ms", "rpc_wall_ms"):
            for label, probability in (("p50", 50), ("p90", 90)):
                summary[field + "_" + label] = float(np.percentile([row[field] for row in timed], probability)) if timed else None
        pilot_rows.append(summary)
    write_csv(ROOT / "PILOT_AGGREGATED.csv", pilot_rows)
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    if all(row["SR"] is not None for row in aggregate_rows):
        for metric in ("SR", "SPL", "nDTW"):
            bar_plot(plots / f"navigation_{metric}.svg", f"R2R development {metric}", {row["group"]: row[metric] for row in aggregate_rows}, metric)
        for field, title in (("model_wall_ms_p50", "Online model latency p50"),
                             ("policy_cycle_wall_ms_p50", "Online navigation cycle latency p50"),
                             ("current_visual_tokens", "Current visual tokens")):
            bar_plot(plots / f"{field}.svg", title, {row["group"]: row[field] for row in aggregate_rows}, "tokens" if field == "current_visual_tokens" else "ms")
    navigation = "# 导航指标\n\n单固定seed42，R2R隔离开发集64集/组；非val_unseen全量，非显著性研究。\n\n"
    navigation += table(aggregate_rows, ("group", "planned", "recorded", "failures", "SR", "OSR", "SPL", "nDTW", "NE_valid_mean", "prediction_count"))
    navigation += "\n\nSR/OSR/SPL/nDTW以64为分母；存在基础设施未解决失败时表格不具正式比较效力。NE仅有效轨迹均值。\n"
    (ROOT / "NAVIGATION_METRICS.md").write_text(navigation)
    cost = "# 计算成本\n\n## 真实闭环测量\n\nCUDA events为设备流时间；CPU enqueue不冒充同步墙钟。模型墙钟与闭环墙钟直接测量。\n\n"
    cost += table(aggregate_rows, ("group", "model_wall_ms_p50", "model_wall_ms_p90", "policy_cycle_wall_ms_p50", "policy_cycle_wall_ms_p90",
                                   "vision_combined_ms_p50", "vision_combined_ms_p90", "history_visual_tokens_p50"))
    cost += "\n\n" + table(aggregate_rows, ("group", "peak_gpu_memory_bytes_max", "peak_reserved_bytes_max", "render_observation_wall_ms_p50", "render_observation_wall_ms_p90"))
    cost += "\n\npolicy_cycle优先读取既有benchmark worker从prepare前到waypoint返回的直接墙钟；RPC到返回另列，不能混用。\n"
    profile_rows = []
    for group in MODES:
        records = read_jsonl(ROOT / "profile" / group / "steps.jsonl")
        for record in records:
            augment_costs(record, architecture)
        summary = {"group": group, "samples": len(records)}
        for key in ("profile_vision_current_ms", "profile_vision_history_ms", "estimated_current_vision_gflops", "estimated_history_vision_gflops"):
            values = [row[key] for row in records]
            summary[key + "_p50"] = float(np.percentile(values, 50)) if values else None
            summary[key + "_p90"] = float(np.percentile(values, 90)) if values else None
        profile_rows.append(summary)
    write_csv(ROOT / "PROFILE_AGGREGATED.csv", profile_rows)
    cost += "\n\n## 同状态独立拆分重放（非闭环计时）\n\n" + table(profile_rows, ("group", "samples", "profile_vision_current_ms_p50", "profile_vision_current_ms_p90", "profile_vision_history_ms_p50", "profile_vision_history_ms_p90"))
    cost += "\n\n真实在线执行为当前/历史混合batch，保持不变；动作生成后独立拆分重放以测量两个部分，二者不可相加冒充原混合batch成本。四组完整episode逐步动作等价性见COST_VALIDATION.json。\n"
    if (ROOT / "RENDER_PILOT.json").exists():
        render = json.loads((ROOT / "RENDER_PILOT.json").read_text())
        cost += f"\n独立512四相机固定状态渲染：20次预热、200次timed，观测接口墙钟p50={render['p50_ms']:.3f}ms、p90={render['p90_ms']:.3f}ms，含读回，不是纯GPU光栅kernel。原始数据见RENDER_PILOT.json。\n"
    cost += "\n\n## 固定开发观测试点（history=0）\n\n" + table(pilot_rows, ("group", "warmup", "timed", "model_wall_ms_p50", "model_wall_ms_p90", "rpc_wall_ms_p50"))
    cost += "\n\n同一dev首帧、相同指令/pose、每次重置session；每组至少20次独立预热并检查≤5%漂移，100次timed。"
    cost += "不是任务2三历史层、三进程完整microbenchmark，不能推广到长历史稳态。预热不删除正式episode首步。\n"
    cost += "\n## FLOPs与显存边界\n\n每次真实visual.forward的grid、输入shape和current/history来源保存在逐步CSV；"
    cost += "用VISION_ARCHITECTURE.json实际24层模块尺寸估计patch Conv3d、QKV/输出Linear、两层MLP、attention及merger。"
    cost += "一次乘加=2 FLOPs；attention按每图/每时间帧的4×N²D，非跨图全注意力。"
    cost += "未计norm、softmax、激活、RoPE、索引/拷贝，状态incomplete；这些是明确形状估计，不是完整实测FLOPs。"
    cost += "不按合并token比例推断算力；历史context pool4不代表只编码4个patch。\n"
    cost += "\nGPU驱动以请求50ms间隔采样，可能漏峰；gpu_samples.csv含设备总占用（包括桌面）。"
    cost += "正式评估两环境worker共享单模型，RPC/闭环含排队及并发竞争，不能称单请求独占延迟；独立profile/固定观测试点无并发排队。四组使用相同并发设置。"
    cost += "torch allocated/reserved为模型进程峰值，不含Habitat。模拟器reset/scene合并报告，物理非渲染量为差值估计，"
    cost += "不把组件之和当直接测量的闭环总耗时；日志IO与reset在逐episode CSV另列。\n"
    mixed = sum(row["mixed_current_history_vision_call"] for row in step_rows)
    cost += f"\n检测到{mixed}次current/history混合视觉调用；若非零，相应分项设备时间不能解释为纯当前/纯历史，需要独立profile。\n"
    cost += "在线纯current/history分项的缺失值保留为空；各项CSV附_n，不能把只在history=0时测得的纯current分位数当全轨迹成本。\n"
    devices = gpu_accounting()
    cost += "\n\n" + table(devices, ("gpu", "peak_device_mib", "interval_p50_ms", "interval_p90_ms", "occupied_device_hours_estimate", "utilization_weighted_device_hours_estimate"))
    (ROOT / "COMPUTATIONAL_COST.md").write_text(cost)
    failure_rows = [row for row in episode_rows if row["failure"]]
    failure = "# 失败分析\n\n正式评估失败数：" + str(len(failure_rows)) + "。导航SR=0不等于执行基础设施失败。\n\n"
    failure += "正式运行前保留两次实现接线失败：v1.0清单漏gt_path_length；v1.1初次日志包装引用不存在的write_step。"
    failure += "均在首次模型预测前阻塞，已修复并重跑；未替换任何episode或改变分母。"
    failure += "诊断原始日志位于attempts/。外部回归首次忘设PYTHONPATH导致import失败，补齐后5项通过，原日志保留。\n"
    if failure_rows:
        failure += table(failure_rows, ("group", "episode_id", "failure_type", "failure"))
    (ROOT / "FAILURE_ANALYSIS.md").write_text(failure)
    status = json.loads((ROOT / "EXECUTION_STATUS.json").read_text()) if (ROOT / "EXECUTION_STATUS.json").exists() else {}
    complete = all(row["complete"] for row in checks)
    report = "# 第一阶段可行性报告\n\n状态：" + ("评估完成" if complete else "未完成，禁止正式结论") + "。不授权自动进入第二阶段。\n\n"
    report += navigation.split("\n\n", 1)[1] + "\n"
    if complete:
        by_group = {row["group"]: row for row in aggregate_rows}
        report += f"\n## 研究问题的描述性回答\n\n1. 仅Glance：G1 SR={by_group['G1']['SR']:.4f}、nDTW={by_group['G1']['nDTW']:.4f}；G0分别{by_group['G0']['SR']:.4f}/{by_group['G0']['nDTW']:.4f}。是否可接受需结合任务容忍度，单seed不作非劣性证明。\n"
        report += f"2. Focus效用：G2−G1 nDTW={by_group['G2']['nDTW']-by_group['G1']['nDTW']:+.4f}，G3−G1={by_group['G3']['nDTW']-by_group['G1']['nDTW']:+.4f}；只属于受控预算下的初步观察。\n"
        report += f"3. Fixed与Random：G2−G3 SR={by_group['G2']['SR']-by_group['G3']['SR']:+.4f}，nDTW={by_group['G2']['nDTW']-by_group['G3']['nDTW']:+.4f}，不能外推到学习型策略。\n"
        report += "4. 当前视觉token：G1相对G0减少75%，G2/G3减少25%；历史预算不变，实际历史token另计。\n"
        report += "5. 实际模型/端到端延迟：见COMPUTATIONAL_COST.md的p50/p90及PILOT_AGGREGATED.csv；不同闭环轨迹/历史长度是混杂因素，固定首帧试点单列，不以token减少替代实测加速。\n"
        for group in ("G1", "G2", "G3"):
            model_change = 100 * (by_group[group]["model_wall_ms_p50"] / by_group["G0"]["model_wall_ms_p50"] - 1)
            cycle_change = 100 * (by_group[group]["policy_cycle_wall_ms_p50"] / by_group["G0"]["policy_cycle_wall_ms_p50"] - 1)
            report += f"   {group}在线模型step p50相对G0变化{model_change:+.2f}%，闭环step p50变化{cycle_change:+.2f}%；不是配对固定长历史的因果加速结论。\n"
        report += "6. 四组完整episode和逐步token/ROI/seed/历史规则检查通过，说明当前采集执行链可用；不证明特征条件学习/训练的稳定性。\n"
        report += "7. 第二阶段起点：先补当前特征bank与顺序编码复用、真实instruction/Glance/已选Focus接口及训练/开发隔离，再独立审查Actor/Critic/PPO方案；完善多历史层成本profile与多seed开发验证。\n"
    prior_statuses = [json.loads(path.read_text()) for path in (ROOT / "attempts").rglob("EXECUTION_STATUS.json")]
    prior_wall = sum(row.get("wall_seconds", 0) for row in prior_statuses)
    total_gpu_hours = status.get("two_gpu_reservation_hours", 0) + sum(row.get("two_gpu_reservation_hours", 0) for row in prior_statuses)
    all_statuses = prior_statuses + [status]
    wall_span = max(row.get("finished_epoch", 0) for row in all_statuses) - min(row.get("started_epoch", float("inf")) for row in all_statuses)
    write_json(ROOT / "RESOURCE_ACCOUNTING.json", {"final_execution_wall_seconds": status.get("wall_seconds", 0),
               "prior_execution_wall_seconds": prior_wall, "total_two_gpu_reservation_hours": total_gpu_hours,
               "first_start_to_final_finish_seconds": wall_span,
               "prior_attempt_count": len(prior_statuses), "definition": "two provisioned devices × measured execution wall; not utilization integral"})
    report += f"\n## 实际资源\n\n最终执行挂钟{status.get('wall_seconds', 0)/3600:.4f}小时，之前执行/失败尝试{prior_wall/3600:.4f}小时；合计两卡预约口径{total_gpu_hours:.4f} GPU小时；"
    report += "这不是利用率积分或纯kernel时间。正式episode耗时另列聚合CSV；预先失败尝试的实际资源另见attempts/，不得忽略。"
    report += f"从首次GPU执行启动到最终执行结束的挂钟跨度{wall_span/3600:.4f}小时（含修复间隙）；最终进程设备占用/利用率加权采样估计另见GPU_ACCOUNTING.json。"
    report += "旧10093/PID1825348已按用户补充授权停止；模型/模拟器分别GPU0/GPU1，GPU0桌面进程保留。\n"
    report += "\n## 限制\n\n固定64集来自原train开发scene，主干可能已见该来源；单seed、7开发scene、无预注册可用性阈值。"
    report += "不声称统计显著、val_unseen泛化、完整FLOPs或学习型状态条件策略有效。\n"
    (ROOT / "STAGE1_FEASIBILITY_REPORT.md").write_text(report)
    return aggregate_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    aggregate(args.allow_incomplete)
