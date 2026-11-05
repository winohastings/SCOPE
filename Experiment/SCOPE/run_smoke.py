from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path

SIMPLENAV = Path("/home/data/czh/Third_Party/baselines/SimpleNav")
HABITAT_PYTHON = "/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python"
MODES = ("original", "glance_only", "fixed_2", "random_2", "learned_2", "learned_adaptive")


def aggregate(root, tasks=("r2r", "rxr")):
    rows, verification = [], []
    for task in tasks:
        reference = None
        for mode in MODES:
            run = root / "runs" / f"{task}_{mode}"
            if not (run / "summary.json").exists():
                continue
            results = [json.loads(path.read_text()) for path in sorted(run.rglob("eval_info.json"))]
            records = [json.loads(path.read_text()) for path in sorted(run.rglob("scope_*.json"))]
            episodes = sorted((result["source_episode_id"], result["scene_id"]) for result in results)
            if reference is None:
                reference = episodes
            if episodes != reference or len(episodes) != 2:
                raise AssertionError(f"Unmatched episodes in {task}/{mode}: {episodes}")
            if any(result.get("failure") for result in results):
                raise AssertionError(f"Failed rollout in {task}/{mode}")
            if len(records) != sum(result["steps"] for result in results):
                raise AssertionError(f"Missing instrumentation in {task}/{mode}")
            expected = {"original": 256, "glance_only": 64, "fixed_2": 192, "random_2": 192, "learned_2": 192}
            for record in records:
                if record["current_visual_tokens"] != expected.get(mode, 64 + 64 * record["focus_count"]):
                    raise AssertionError("Token budget mismatch")
                if not 0 <= record["focus_count"] <= 2:
                    raise AssertionError("Focus budget exceeded")
            row = {"task": task, "policy_mode": mode, "episodes": len(results), "prediction_steps": len(records),
                   "scope": "bounded smoke, not benchmark performance",
                   "selector": records[0]["selector"] if records else "none"}
            for metric in ("SR", "OSR", "NE", "SPL", "nDTW", "path_length", "steps_taken"):
                row[metric] = statistics.mean(result["metrics"][metric] for result in results)
            for key in ("current_visual_tokens", "total_visual_tokens", "focus_count", "vision_ms", "llm_ms",
                        "dit_b_ms", "server_total_ms", "habitat_render_ms", "total_ms", "peak_gpu_memory_bytes"):
                values = [record[key] for record in records]
                row[key + "_mean"] = statistics.mean(values)
                row[key + "_max"] = max(values)
            rows.append(row)
            verification.append({"task": task, "policy_mode": mode, "episodes": episodes,
                                 "steps": len(records), "passed": True})
    if rows:
        with (root / "aggregate.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (root / "smoke_verification.json").write_text(json.dumps(verification, indent=2))
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--task", choices=("r2r", "rxr", "both"), default="both")
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    tasks = ("r2r", "rxr") if args.task == "both" else (args.task,)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        environment = {**os.environ, "PYTHONPATH": str(SIMPLENAV), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
        for task in tasks:
            for mode in MODES:
                config = "config_standard_local.yaml" if task == "r2r" else "config_standard_local_50.yaml"
                command = [HABITAT_PYTHON, str(SIMPLENAV / f"benchmark/vlnce/{task}/eval.py"),
                           "--config", str(SIMPLENAV / f"benchmark/vlnce/{task}/{config}")]
                for override in ("benchmark.max_samples=2", f"benchmark.max_steps={args.max_steps}",
                                 "env.kwargs.image_size=512", "model.uri=ws://127.0.0.1:10094",
                                 f"policy_mode={mode}", f"output.root={root / 'runs'}", f"output.run_name={task}_{mode}"):
                    command.extend(("--override", override))
                (root / f"{task}_{mode}_command.json").write_text(json.dumps(command, indent=2))
                with (root / f"{task}_{mode}.log").open("w") as stream:
                    subprocess.run(command, env=environment, cwd=SIMPLENAV, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
    count = aggregate(root, tasks=tasks)
    expected_count = len(tasks) * len(MODES)
    if count != expected_count:
        raise AssertionError(f"Expected {expected_count} task/mode rows, got {count}")
    print(f"Verified {count} task/mode groups, {count * 2} rollouts")


if __name__ == "__main__":
    main()
