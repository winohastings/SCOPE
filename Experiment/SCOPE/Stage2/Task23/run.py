from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
sys.path.insert(0, str(ROOT.parent))
from Experiment.SCOPE.Stage2.Task23.data import create_manifest, load_manifest
from Experiment.SCOPE.Stage2.data_guard import STAGE1, write_json
from Experiment.SCOPE.Stage2.real_rollout import FeedbackBackend, FeedbackModel
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.log.artifacts import ArtifactStore
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runner.worker import _run_episode
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode, WorkerPlan
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

MODEL_PYTHON = "/home/data/czh/Third_Party/baselines/SimpleNav/.venv/bin/python"


class PPOFeedbackModel(FeedbackModel):
    reward_purposes = ("train", "val_seen")


def gpu_monitor(path, stopped):
    with Path(path).open("w") as stream:
        stream.write("timestamp,index,utilization_gpu,memory_used,power_draw\n")
        while not stopped.is_set():
            result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True)
            stream.write(result.stdout)
            stream.flush()
            stopped.wait(5)


class ExperimentDriver:
    def __init__(self, args, config, manifest, client):
        self.args, self.config, self.manifest, self.client = args, config, manifest, client
        self.backends = {}
        self.models = {}
        self.results = []
        self.started = time.monotonic()
        self.raw = yaml.safe_load((STAGE1 / "configs/G1.yaml").read_text())

    def episode(self, split, record, ordinal, max_steps, label):
        output = self.args.output / "episodes" / label
        output.mkdir(parents=True, exist_ok=False)
        raw = copy.deepcopy(self.raw)
        raw["model"].update(uri=f"ws://127.0.0.1:{self.args.port}", seed=self.config["seed"])
        raw["benchmark"]["max_steps"] = max_steps
        raw["benchmark"]["kwargs"].update(split=split, distance_metric="geodesic")
        raw["input"]["split"] = split
        raw["env"]["kwargs"].update(split=split, max_episode_steps=self.config["max_habitat_steps"])
        raw["output"].update(root=str(output), run_name=label, save_step_artifacts=False)
        (output / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_eval_config(output / "config.yaml")
        if split not in self.backends:
            backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
            self.backends[split] = backend
            self.models[split] = PPOFeedbackModel(self.client, backend, cfg, split, False)
        backend, model = self.backends[split], self.models[split]
        model.cfg = cfg
        model.report = None
        model.pending = None
        model.last_instruction = None
        episode = EvalEpisode(**record)
        model.episode = episode
        worker = WorkerPlan(0, 1, [episode], output, output / "worker.log", WorkerBackendPlan("habitat"),
                            {episode.episode_uid: label}, {episode.episode_uid: ordinal})
        began = time.monotonic()
        result = _run_episode(cfg=cfg, worker=worker, store=ArtifactStore(output), episode=episode,
                              runtime=build_benchmark_runtime(cfg), model=model, env_backend=FeedbackBackend(backend, model))
        row = asdict(result) | {"purpose": split, "wall_seconds": time.monotonic() - began, "training": model.report}
        if model.report is not None and "episode_summary" in model.report:
            summary = model.report["episode_summary"]
            success = int(summary["terminated"] and summary["final_geodesic_distance"] < 3)
            distance = summary["start_geodesic_distance"]
            row["SR"] = success
            row["SPL"] = success * distance / max(distance, result.path_length, 1e-8)
            row["reward_sum"] = summary["reward_sum"]
            row["focus_mean"] = sum(int(count) * total for count, total in summary["focus_histogram"].items()) / summary["transitions"]
        self.results.append(row)
        write_json(output / "result.json", row)
        write_json(self.args.output / "episodes_summary.json", self.results)
        print(json.dumps({key: row.get(key) for key in ("purpose", "source_episode_id", "steps", "SR", "SPL", "reward_sum", "focus_mean", "failure")}), flush=True)
        if result.failure:
            raise RuntimeError(result.failure)
        return row

    def evaluate(self, round_index):
        self.client._request("configure", {"split": "val_seen"})
        rows = []
        for index, record in enumerate(self.manifest["val_seen"]):
            rows.append(self.episode("val_seen", record, index, self.config["max_navigation_steps"],
                                     f"val_{round_index:03d}_{index:03d}"))
        metrics = {"split": "val_seen", "episodes": len(rows), "SR": float(np.mean([row["SR"] for row in rows])),
                   "SPL": float(np.mean([row["SPL"] for row in rows])), "reward": float(np.mean([row["reward_sum"] for row in rows])),
                   "focus_mean": float(np.mean([row["focus_mean"] for row in rows])),
                   "nDTW": float(np.mean([row["nDTW"] for row in rows])), "round": round_index}
        write_json(self.args.output / f"validation_{round_index:03d}.json", metrics)
        selection = self.client._request("validation_result", {"metrics": metrics})
        self.client._request("configure", {"split": "train"})
        return selection

    def train(self):
        status = self.client._request("status", {})
        cursor = status["cursor"]
        budget = self.args.budget or self.config["budget"]
        if self.args.stage == "gate":
            budget = 64
        if self.args.stage == "train" and status["collected"] == 0:
            self.evaluate(cursor["validation_round"])
            cursor["validation_round"] += 1
            cursor["next_validation_at"] = self.config["validation_every_transitions"]
            self.client._request("save", {"cursor": cursor})
        self.client._request("configure", {"split": "train"})
        deadline = time.monotonic() + self.config["wall_time_limit_hours"] * 3600
        while status["collected"] < budget:
            if time.monotonic() > deadline:
                self.client._request("save", {"cursor": cursor})
                write_json(self.args.output / "INCOMPLETE.json", {"reason": "explicit wall time budget", "target": budget, **status})
                return
            ordinal = cursor["train_episode_index"]
            record = self.manifest["train"][ordinal % len(self.manifest["train"]) ]
            self.episode("train", record, ordinal, min(self.config["max_navigation_steps"], budget - status["collected"]),
                         f"train_{ordinal:06d}")
            cursor["train_episode_index"] += 1
            status = self.client._request("status", {})
            self.client._request("save", {"cursor": cursor})
            remaining = budget - status["collected"]
            should_update = status["buffer_transitions"] >= self.config["rollout_minimum"] and (remaining >= self.config["rollout_minimum"] or remaining == 0)
            if not should_update:
                continue
            if self.args.stage == "gate":
                before = self.client._request("save", {"cursor": cursor, "label": "gate_before"})["checkpoint"]
            update = self.client._request("update", {})
            status = self.client._request("status", {})
            self.client._request("save", {"cursor": cursor})
            print(json.dumps({"update": update["update"], "trained": status["trained"], "actor_change": update["actor_parameter_l2_change"],
                              "critic_change": update["critic_parameter_l2_change"], "ratio_error": update["initial_ratio_max_error"]}), flush=True)
            if self.args.stage == "gate":
                after = self.client._request("save", {"cursor": cursor, "label": "gate_after"})["checkpoint"]
                restored = self.client._request("restore", {"checkpoint": before})
                repeated = self.client._request("update", {})
                replayed = self.client._request("save", {"cursor": cursor, "label": "gate_repeated"})["checkpoint"]
                check = subprocess.run([MODEL_PYTHON, str(ROOT / "audit.py"), "--compare", after, "--restored", replayed,
                                        "--output", str(self.args.output / "RESTORE_GATE.json")], capture_output=True, text=True)
                (self.args.output / "restore_audit.log").write_text(check.stdout + check.stderr)
                if check.returncode:
                    raise RuntimeError("checkpoint/optimizer/RNG restoration audit failed")
                if not all(value > 0 for value in update["head_gradient_norms"].values()):
                    raise AssertionError("gate requires gradients in STOP/camera/position/Critic heads")
                write_json(self.args.output / "TRAINING_GATE.json", {"passed": True, "real_transitions": status["collected"],
                           "first_update": update, "restored_update": repeated, "restored_cursor": restored["cursor"]})
            elif status["trained"] >= cursor["next_validation_at"] or status["collected"] == budget:
                self.evaluate(cursor["validation_round"])
                cursor["validation_round"] += 1
                cursor["next_validation_at"] = (status["trained"] // self.config["validation_every_transitions"] + 1) * self.config["validation_every_transitions"]
            status = self.client._request("status", {})
            saved = self.client._request("save", {"cursor": cursor})
            status = {key: value for key, value in saved.items() if key != "checkpoint"}
            write_json(self.args.output / "PROGRESS.json", {"target": budget, **status})
        if status["trained"] != budget or status["buffer_transitions"]:
            raise AssertionError("budget completion requires trained transitions and empty buffer")
        audit = self.client._request("audit", {})
        write_json(self.args.output / "COMPLETE.json", {"target": budget, **status, "freeze_audit": audit,
                   "wall_seconds": time.monotonic() - self.started, "stage": self.args.stage})

    def close(self):
        for backend in self.backends.values():
            backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("gate", "train"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--port", type=int, default=10123)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    if args.stage == "gate" and args.resume is not None:
        parser.error("gate requires a fresh 64-transition rollout; resume is only supported for formal training")
    config = json.loads(args.config.read_text())
    if args.stage == "train":
        if args.gate is None or not json.loads(args.gate.read_text())["passed"]:
            parser.error("formal training requires completed real optimizer/restore gate")
    if args.budget is not None and args.budget not in [4096, *config["continuation_budgets"]]:
        parser.error("formal budget must be 4096, 10000 or 20000")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    if args.manifest is None:
        args.manifest = args.output / "DATA_MANIFEST.json"
        manifest = create_manifest(args.manifest, config)
    else:
        manifest = load_manifest(args.manifest)
        manifest_bytes = args.manifest.read_bytes()
        args.manifest = args.output / "DATA_MANIFEST.json"
        args.manifest.write_bytes(manifest_bytes)
    write_json(args.output / "config.json", config)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen([MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(args.output / "runtime"),
                               "--manifest", str(args.manifest), "--config", str(args.config), "--port", str(args.port)],
                              stdout=log, stderr=subprocess.STDOUT, env=environment)
    stopped = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(args.output / "gpu.csv", stopped), daemon=True)
    monitor.start()
    client = None
    driver = None
    try:
        deadline = time.monotonic() + 240
        while not (args.output / "runtime/ready.json").exists():
            if server.poll() is not None:
                raise RuntimeError("model server failed; inspect server.log")
            if time.monotonic() > deadline:
                raise TimeoutError("model startup deadline exceeded")
            time.sleep(1)
        client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
        client._ws.ping_timeout = 1200
        if args.resume:
            client._request("restore", {"checkpoint": str(args.resume)})
        driver = ExperimentDriver(args, config, manifest, client)
        driver.train()
    except BaseException as error:
        status = {"available": False}
        if client is not None:
            try:
                status = client._request("status", {})
            except Exception:
                pass
        write_json(args.output / "INCOMPLETE.json", {"reason": repr(error), "stage": args.stage,
                   "target": 64 if args.stage == "gate" else args.budget or config["budget"],
                   "last_observed_status": status,
                   "recovery": "use the last completed episode-boundary checkpoint; never replay a partial update"})
        raise
    finally:
        if driver is not None:
            driver.close()
        if client is not None:
            client.close()
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stopped.set()
        monitor.join(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
