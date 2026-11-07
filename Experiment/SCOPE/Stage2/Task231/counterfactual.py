from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.run import ExperimentDriver, MODEL_PYTHON, WebsocketClientPolicy
from Experiment.SCOPE.Stage2.data_guard import write_json


def balanced_indices(records, count):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["scene_id"]].append(index)
    ordered = []
    while any(groups.values()):
        for scene in sorted(groups):
            if groups[scene]:
                ordered.append(groups[scene].pop(0))
    return ordered[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Task23/artifacts/ppo4096")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, choices=(12, 32), default=12)
    parser.add_argument("--port", type=int, default=10231)
    parser.add_argument("--completed", type=Path)
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT):
        parser.error("all outputs must remain inside Task231")
    if args.shards < 1 or not 0 <= args.shard_id < args.shards:
        parser.error("invalid diagnostic shard")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.source / "config.json").read_text())
    manifest = json.loads((args.source / "DATA_MANIFEST.json").read_text())
    indices = balanced_indices(manifest["val_seen"], args.episodes)[args.shard_id::args.shards]
    write_json(args.output / "selection.json", {"indices": indices, "records": [manifest["val_seen"][index] for index in indices],
                                               "selection": "scene round-robin; original within-scene order; no outcome selection"})
    completed = json.loads(args.completed.read_text()) if args.completed else []
    completed = [row for row in completed if row["manifest_index"] in indices]
    completed_keys = {(row["checkpoint_update"], row["mode"], row["manifest_index"]) for row in completed}
    if len(completed_keys) != len(completed):
        raise ValueError("duplicate completed diagnostic episodes")
    write_json(args.output / "continuation.json", {"source": str(args.completed) if args.completed else None,
                                                 "reused_completed_episodes": len(completed), "partial_episodes_reused": 0})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.model_gpu, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen([MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(args.output / "runtime"),
                                   "--manifest", str(args.source / "DATA_MANIFEST.json"), "--config", str(args.source / "config.json"),
                                   "--port", str(args.port)], stdout=log, stderr=subprocess.STDOUT, env=environment)
        client = driver = None
        try:
            deadline = time.monotonic() + 300
            while not (args.output / "runtime/ready.json").exists():
                if server.poll() is not None:
                    raise RuntimeError("diagnostic server failed; inspect server.log")
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup timeout")
                time.sleep(1)
            client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
            client._ws.ping_timeout = 1200
            driver = ExperimentDriver(args, config, manifest, client)
            results = list(completed)
            for index in indices:
                for update, transitions in ((56, 4096), (14, 1033), (28, 2054)):
                    for mode in ("native", "forced4"):
                        if (update, mode, index) in completed_keys:
                            continue
                        checkpoint = args.source / "runtime/checkpoints" / f"update_{update:04d}.pt"
                        client._request("restore", {"checkpoint": str(checkpoint.resolve()), "diagnostic_mode": mode})
                        client._request("configure", {"split": "val_seen"})
                        row = driver.episode("val_seen", manifest["val_seen"][index], index, config["max_navigation_steps"],
                                             f"u{update:04d}_{mode}_{index:03d}")
                        results.append(row | {"checkpoint_update": update, "transitions": transitions, "mode": mode, "manifest_index": index})
                        write_json(args.output / "counterfactual_results.json", results)
            write_json(args.output / "COMPLETE.json", {"episodes": len(results), "freeze_audit": client._request("audit", {}),
                                                       "training_updates_performed": 0})
        except BaseException as error:
            write_json(args.output / "INCOMPLETE.json", {"error": repr(error)})
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


if __name__ == "__main__":
    main()
