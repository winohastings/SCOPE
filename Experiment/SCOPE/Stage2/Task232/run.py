from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.run import ExperimentDriver, MODEL_PYTHON, WebsocketClientPolicy, gpu_monitor
from Experiment.SCOPE.Stage2.Task231.counterfactual import balanced_indices
from Experiment.SCOPE.Stage2.data_guard import write_json


class RepairDriver(ExperimentDriver):
    def save_probe(self, added):
        return self.client._request("save", {"cursor": self.cursor, "label": f"added_{added:04d}", "probe": True})

    def run_audit(self, first, second, mode, name):
        result = subprocess.run([MODEL_PYTHON, str(ROOT / "audit.py"), "--first", str(first), "--second", str(second),
                                 "--mode", mode, "--output", str(self.args.output / f"{name}.json")],
                                capture_output=True, text=True)
        (self.args.output / f"{name}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stderr[-2000:]}")

    def repair_train(self):
        status = self.client._request("status", {})
        start = self.config["fork_transitions"]
        self.cursor = status["cursor"]
        self.save_probe(status["trained"] - start)
        self.client._request("configure", {"split": "train"})
        first_episode = True
        first_update = True
        milestones = [64] if self.args.mode == "gate" else [value for value in (256, 512, 768, 1024) if status["collected"] - start < value <= self.args.budget]
        for milestone in milestones:
            target = start + milestone
            while status["collected"] < target:
                ordinal = self.cursor["train_episode_index"]
                record = self.manifest["train"][ordinal % len(self.manifest["train"])]
                label = f"train_{ordinal:06d}"
                self.episode("train", record, ordinal, min(self.config["max_navigation_steps"], target - status["collected"]), label)
                if first_episode and (self.args.mode == "gate" or self.args.branch == "C0"):
                    self.run_audit(self.args.output / "episodes" / label / "result.json",
                                   ROOT.parent / "Task23/artifacts/ppo4096/episodes" / label / "result.json", "forward", "UNCHANGED_FORWARD_GATE")
                first_episode = False
                self.cursor["train_episode_index"] += 1
                status = self.client._request("status", {})
                self.client._request("save", {"cursor": self.cursor})
                remaining = target - status["collected"]
                minimum = self.config["rollout_minimum"]
                should_update = status["buffer_transitions"] >= minimum and (remaining >= minimum or remaining == 0)
                if not should_update:
                    continue
                before = self.client._request("save", {"cursor": self.cursor, "label": "restore_before"})["checkpoint"] if first_update else None
                update = self.client._request("update", {})
                status = self.client._request("status", {})
                print(json.dumps({"branch": self.args.branch, "added_trained": status["trained"] - start,
                                  "batch": update["transitions"], "ratio_error": update["initial_ratio_max_error"],
                                  "actor_change": update["actor_parameter_l2_change"], "critic_change": update["critic_parameter_l2_change"]}), flush=True)
                if first_update:
                    after = self.client._request("save", {"cursor": self.cursor, "label": "restore_after"})["checkpoint"]
                    self.client._request("restore", {"checkpoint": before})
                    self.client._request("update", {})
                    repeated = self.client._request("save", {"cursor": self.cursor, "label": "restore_repeated"})["checkpoint"]
                    self.run_audit(after, repeated, "restore", "RESTORE_GATE")
                    first_update = False
                self.client._request("save", {"cursor": self.cursor})
            assert status["trained"] == target and not status["buffer_transitions"]
            saved = self.save_probe(milestone)
            write_json(self.args.output / f"MILESTONE_{milestone:04d}.json", {"added_transitions": milestone, "checkpoint": saved["checkpoint"],
                                                                          "trained": status["trained"], "update": status["updates"]})
        if self.args.mode == "gate":
            write_json(self.args.output / "TRAINING_GATE.json", {"passed": True, "fresh_transitions": 64,
                       "historical_forward_bitwise_equal": True, "restore_bitwise_equal": True, "freeze": self.client._request("audit", {})})
        else:
            self.evaluate_subsets()

    def evaluate_subsets(self):
        self.client._request("configure", {"split": "val_seen"})
        subsets = [("independent", self.manifest["independent_val_seen"])]
        if self.args.mode != "baseline":
            subsets.append(("historical", self.manifest["historical_val_seen"]))
        for subset, records in subsets:
            rows = []
            for index, record in enumerate(sorted(records, key=lambda row: row["scene_id"])):
                row = self.episode("val_seen", record, index, self.config["max_navigation_steps"], f"{subset}_{index:03d}")
                rows.append(row)
                write_json(self.args.output / f"evaluation_{subset}.json", rows)

    def thresholds(self):
        checkpoint = ROOT.parent / "Task23/artifacts/ppo4096/runtime/checkpoints/update_0056.pt"
        historical = self.manifest["historical_val_seen"]
        indices = balanced_indices(historical, 12)
        rows = []
        for index in indices:
            for threshold in (.45, .60):
                self.client._request("restore", {"checkpoint": str(checkpoint), "original_evaluation": True, "stop_threshold": threshold})
                self.client._request("configure", {"split": "val_seen"})
                row = self.episode("val_seen", historical[index], index, self.config["max_navigation_steps"], f"threshold{threshold:.2f}_{index:03d}")
                rows.append(row | {"stop_threshold": threshold, "manifest_index": index})
                write_json(self.args.output / "threshold_results.json", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", choices=("U", "A", "B", "C", "C0"), required=True)
    parser.add_argument("--mode", choices=("gate", "baseline", "train", "thresholds"), required=True)
    parser.add_argument("--budget", type=int, choices=(512, 1024), default=512)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--extension-decision", type=Path)
    args = parser.parse_args()
    if args.branch in ("C", "C0") and (args.mode != "train" or args.budget != 512 or args.resume):
        parser.error("C/C0 protocol authorizes only fresh 512-transition training")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        parser.error("Task232-only outputs")
    if args.mode == "gate" and args.branch != "U":
        parser.error("unchanged gate must use U")
    if args.mode == "train":
        if args.branch == "U" or not args.gate or not json.loads(args.gate.read_text())["passed"]:
            parser.error("A/B require a completed unchanged-fork gate")
        if args.budget == 1024 and (not args.resume or not args.extension_decision or not json.loads(args.extension_decision.read_text())[args.branch]["eligible"]):
            parser.error("1024 requires explicit evidence-based extension eligibility and a 512 checkpoint")
    config_path = ROOT / "frozen" / f"config_{args.branch}.json"
    manifest_path = ROOT / "frozen/DATA_MANIFEST.json"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", config)
    write_json(args.output / "LAUNCH.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.model_gpu, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    stopped = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(args.output / "gpu.csv", stopped), daemon=True)
    monitor.start()
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen([MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(args.output / "runtime"),
                                   "--config", str(config_path), "--manifest", str(manifest_path), "--port", str(args.port)],
                                   stdout=log, stderr=subprocess.STDOUT, env=environment)
        client = driver = None
        began = time.monotonic()
        try:
            deadline = time.monotonic() + 300
            while not (args.output / "runtime/ready.json").exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("model startup failed; inspect server.log")
                time.sleep(1)
            client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
            client._ws.ping_timeout = 1200
            checkpoint = args.resume or ROOT / "frozen" / f"fork_{args.branch}.pt"
            initial_status = client._request("restore", {"checkpoint": str(checkpoint)})
            driver = RepairDriver(args, config, manifest, client)
            if args.mode in ("gate", "train"):
                driver.repair_train()
            elif args.mode == "baseline":
                driver.cursor = client._request("status", {})["cursor"]
                driver.save_probe(0)
                driver.evaluate_subsets()
            else:
                driver.thresholds()
            final_status = client._request("status", {})
            write_json(args.output / "COMPLETE.json", {"mode": args.mode, "branch": args.branch, "status": final_status,
                       "freeze_audit": client._request("audit", {}), "wall_seconds": time.monotonic() - began,
                       "new_training_transitions": final_status["collected"] - initial_status["collected"] if args.mode in ("gate", "train") else 0,
                       "fork_added_trained_transitions": final_status["trained"] - 1033 if args.mode in ("gate", "train") else 0})
        except BaseException as error:
            write_json(args.output / "INCOMPLETE.json", {"error": repr(error), "mode": args.mode, "branch": args.branch})
            raise
        finally:
            try:
                if driver is not None:
                    driver.close()
                if client is not None:
                    client.close()
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                stopped.set()
                monitor.join(timeout=10)


if __name__ == "__main__":
    main()
