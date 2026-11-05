from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time

from data_guard import ROOT, protected_snapshot, sha256, write_json

MODEL_PYTHON = "/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python"
HABITAT_PYTHON = "/home/data/czh/AdaPercVLN/Assumption/assumption1/envs/correctnav_hab033/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-service-output", type=Path)
    parser.add_argument("--port", type=int, default=10096)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    runtime = args.existing_service_output.resolve() if args.existing_service_output else args.output / "runtime"
    before = protected_snapshot()
    write_json(args.output / "protected_before.json", before)
    write_json(args.output / "environment.json", {"started": datetime.datetime.now().astimezone().isoformat(),
               "platform": platform.platform(), "model_python": MODEL_PYTHON, "habitat_python": HABITAT_PYTHON,
               "model_gpu": 0, "habitat_gpu": 1, "runtime": str(runtime),
               "initial_gpu": subprocess.check_output(["nvidia-smi"], text=True),
               "source_hashes": {path.name: sha256(path) for path in ROOT.glob("*.py")}})
    server, sampler = None, None
    server_log = (args.output / "server.log").open("w")
    gpu_log = (args.output / "gpu_samples.csv").open("w")
    commands = []
    try:
        if args.existing_service_output is None:
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                    raise RuntimeError("port already occupied; do not alter an unrelated service")
            except ConnectionRefusedError:
                pass
            command = [MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(runtime), "--port", str(args.port)]
            server = subprocess.Popen(command, env=environment | {"CUDA_VISIBLE_DEVICES": "0"}, stdout=server_log,
                                      stderr=subprocess.STDOUT)
        sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw",
                                    "--format=csv", "-lms", "50"], stdout=gpu_log, stderr=subprocess.STDOUT)
        for attempt in range(240):
            if server is not None and server.poll() is not None:
                raise RuntimeError("server failed; see server.log")
            try:
                with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("service unavailable")
        cases = [("dev_before", "dev", 2, 200, []),
                 ("train_A", "train", 1, 4, ["--regime", "A", "--update"]),
                 ("train_B", "train", 1, 4, ["--regime", "B", "--update"]),
                 ("train_C", "train", 2, 8, ["--regime", "C", "--update"]),
                 ("dev_after", "dev", 2, 200, []),
                 ("dev_restored", "dev", 2, 200, ["--resume", str(runtime / "actor_critic_C_update_002.pt")]),
                 ("diagnostics_train", "train", 1, 2, ["--mode", "diagnose"]),
                 ("diagnostics_dev", "dev", 1, 2, ["--mode", "diagnose"])]
        for label, purpose, count, steps, extra in cases:
            command = [HABITAT_PYTHON, str(ROOT / "real_rollout.py"), "--output", str(args.output / label),
                       "--purpose", purpose, "--count", str(count), "--max-steps", str(steps),
                       "--port", str(args.port)] + extra
            began = time.perf_counter()
            print(f"START {label}", flush=True)
            with (args.output / f"{label}.log").open("w") as stream:
                process = subprocess.run(command, env=environment | {"CUDA_VISIBLE_DEVICES": "1"}, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=1800)
            row = {"label": label, "command": command, "returncode": process.returncode,
                   "wall_seconds": time.perf_counter() - began}
            commands.append(row)
            write_json(args.output / "commands.json", commands)
            if process.returncode:
                raise RuntimeError(f"{label} failed; original output retained")
            print(f"DONE {label} {row['wall_seconds']:.2f}s", flush=True)
        command = [MODEL_PYTHON, str(ROOT / "learning_sanity.py"), "--rollout",
                   str(runtime / "real_train_rollout_C_1.pt"), "--output", str(args.output / "learning_sanity.json")]
        subprocess.run(command, env=environment, check=True)
        after = protected_snapshot()
        changed = [path for path, digest in before.items() if after.get(path) != digest]
        write_json(args.output / "protection_audit.json", {"passed": not changed, "changed": changed,
                   "protected_count": len(before), "checkpoint_included": True})
        if changed:
            raise AssertionError("protected historical artifacts changed")
        write_json(args.output / "COMPLETE.json", {"passed": True, "wall_seconds": time.perf_counter() - started,
                   "runtime": str(runtime), "cases": len(cases), "full_training_started": False,
                   "heldout_evaluation_started": False})
    finally:
        for process in (sampler, server):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=60)
        server_log.close()
        gpu_log.close()


if __name__ == "__main__":
    main()
