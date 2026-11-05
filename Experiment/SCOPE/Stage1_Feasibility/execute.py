from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from compile_configs import ROOT, SIMPLENAV, sha256, validate_freeze, write_json

MODEL_PYTHON = SIMPLENAV / ".venv/bin/python"
HABITAT_PYTHON = Path("/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python")


def command_output(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"returncode": result.returncode, "output": result.stdout}


def main():
    validate_freeze()
    started = time.time()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                   "PYTHONPATH": str(SIMPLENAV), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    write_json(ROOT / "RUN_ENVIRONMENT.json", {"started": datetime.datetime.now().astimezone().isoformat(),
               "platform": platform.platform(), "python": sys.version,
               "gpu_before": command_output(["nvidia-smi"]),
               "software_model": command_output([str(MODEL_PYTHON), "-m", "pip", "freeze"]),
               "software_habitat": command_output([str(HABITAT_PYTHON), "-m", "pip", "freeze"]),
               "external_code_commit": command_output(["git", "-C", str(SIMPLENAV), "rev-parse", "HEAD"]),
               "project_code_commit": "not a git repository; source hashes used",
               "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in ROOT.glob("*.py")},
               "model_gpu": 0, "habitat_gpu": 1, "old_service_stopped_by_user_authorization": "10093/PID1825348",
               "determinism": "fixed seeds; deterministic algorithms warn_only; warnings retained"})
    server_log = (ROOT / "server.log").open("w")
    gpu_log = (ROOT / "gpu_samples.csv").open("w")
    sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw",
                                "--format=csv", "-lms", "50"], stdout=gpu_log, stderr=subprocess.STDOUT)
    server = subprocess.Popen([str(MODEL_PYTHON), str(ROOT / "server.py")],
                              env={**environment, "CUDA_VISIBLE_DEVICES": "0"}, stdout=server_log, stderr=subprocess.STDOUT)
    status = "failed"
    try:
        import socket
        for attempt in range(600):
            if server.poll() is not None:
                raise RuntimeError("model server failed; see server.log")
            try:
                with socket.create_connection(("127.0.0.1", 10095), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("server startup exceeded 600 seconds")
        for phase in ("smoke", "profile", "pilot", "runs"):
            if phase == "pilot":
                commands = [("pilot", [str(MODEL_PYTHON), str(ROOT / "pilot.py")])]
            else:
                commands = [(f"{phase}_{group}", [str(HABITAT_PYTHON), str(ROOT / "run_group.py"),
                            "--group", group, "--phase", phase]) for group in ("G0", "G1", "G2", "G3")]
            for label, command in commands:
                if phase == "runs":
                    processes = []
                    streams = []
                    begin = time.perf_counter()
                    try:
                        for shard in range(2):
                            stream = (ROOT / f"{label}_shard_{shard}.log").open("w")
                            streams.append(stream)
                            processes.append(subprocess.Popen(command + ["--shard-index", str(shard), "--shard-count", "2"],
                                                              env=environment, cwd=SIMPLENAV, stdout=stream, stderr=subprocess.STDOUT))
                        for process in processes:
                            if process.wait() != 0:
                                raise RuntimeError(f"evaluation worker failed: {label}")
                    finally:
                        for process in processes:
                            if process.poll() is None:
                                process.terminate()
                                process.wait(timeout=60)
                        for stream in streams:
                            stream.close()
                    group = label.split("_")[1]
                    completions = [json.loads((ROOT / "runs" / group / f"COMPLETE_shard_{shard}.json").read_text()) for shard in range(2)]
                    write_json(ROOT / "runs" / group / "COMPLETE.json", {"group": group, "count": sum(row["count"] for row in completions),
                               "failures": sum(row["failures"] for row in completions), "workers": 2,
                               "wall_seconds": time.perf_counter() - begin, "model_instances": 1})
                    print(f"DONE {label}: two environment workers, one shared model", flush=True)
                    continue
                if phase in ("smoke", "profile") and (ROOT / phase / label.split("_")[1] / "COMPLETE.json").exists():
                    completion = json.loads((ROOT / phase / label.split("_")[1] / "COMPLETE.json").read_text())
                    if completion["failures"] == 0:
                        print(f"REUSE verified complete {label}", flush=True)
                        continue
                print(f"START {label}", flush=True)
                with (ROOT / f"{label}.log").open("w") as stream:
                    subprocess.run(command, env=environment, cwd=SIMPLENAV, stdout=stream, stderr=subprocess.STDOUT, check=True)
                if phase == "smoke":
                    completion = json.loads((ROOT / phase / label.split("_")[1] / "COMPLETE.json").read_text())
                    if completion["failures"]:
                        raise RuntimeError(f"full episode smoke failed: {label}")
                print(f"DONE {label}", flush=True)
            if phase == "profile":
                import numpy as np
                equivalence = []
                for group in ("G0", "G1", "G2", "G3"):
                    rows = [json.loads(line) for line in (ROOT / "profile" / group / "steps.jsonl").read_text().splitlines()]
                    if not rows or not all("profile_vision_current_ms" in row and "profile_vision_history_ms" in row for row in rows):
                        raise RuntimeError("missing separate current/history profile")
                    reference = [json.loads(line) for line in (ROOT / "smoke" / group / "steps.jsonl").read_text().splitlines()]
                    if len(rows) != len(reference):
                        raise RuntimeError("profiling changed full episode prediction count")
                    delta = 0.0
                    for expected, actual in zip(reference, rows):
                        np.testing.assert_allclose(actual["predicted_body_waypoints"], expected["predicted_body_waypoints"], atol=1e-5, rtol=1e-5)
                        delta = max(delta, float(np.abs(np.asarray(actual["predicted_body_waypoints"]) - np.asarray(expected["predicted_body_waypoints"])).max()))
                    equivalence.append({"group": group, "predictions": len(rows), "max_abs_action_difference": delta})
                write_json(ROOT / "COST_VALIDATION.json", {"passed": True, "method": "post-prediction split replay, all four full episodes",
                           "note": "natural online batching preserved; split timings not additive to online latency",
                           "full_episode_action_equivalence": equivalence})
        status = "evaluations_finished_requires_aggregation"
    finally:
        server.terminate()
        sampler.terminate()
        server.wait(timeout=60)
        sampler.wait(timeout=10)
        server_log.close()
        gpu_log.close()
        write_json(ROOT / "EXECUTION_STATUS.json", {"status": status, "started_epoch": started,
                   "finished_epoch": time.time(), "wall_seconds": time.time() - started,
                   "two_gpu_reservation_hours": 2 * (time.time() - started) / 3600,
                   "warning": "reservation time, not utilization-integrated compute; unrelated GPU1 service excluded"})


if __name__ == "__main__":
    main()
