from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
sys.path.insert(0, str(ROOT.parent))
from Experiment.SCOPE.Stage2.data_guard import STAGE1, episodes, write_json
from Experiment.SCOPE.Stage2.real_rollout import navigate
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10122)
    parser.add_argument("--mode", choices=("deterministic", "training-sampling"), default="training-sampling")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=222)
    args = parser.parse_args()
    if not 1 <= args.count <= 2 or not 1 <= args.max_steps <= 200:
        parser.error("inference check only: 1..2 episodes and 1..200 navigation steps")
    args.purpose, args.update, args.episode_start = "train", False, 0
    args.output.mkdir(parents=True, exist_ok=False)
    raw = yaml.safe_load((STAGE1 / "configs/G1.yaml").read_text())
    raw["model"]["uri"] = f"ws://127.0.0.1:{args.port}"
    raw["model"]["seed"] = args.seed
    raw["benchmark"]["max_steps"] = args.max_steps
    raw["output"]["root"] = str(args.output)
    raw["output"]["run_name"] = f"task22-{args.mode}"
    (args.output / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    records = episodes("train", args.count)
    write_json(args.output / "episodes.json", records)
    cfg = load_eval_config(args.output / "config.yaml")
    client = WebsocketClientPolicy(uri=raw["model"]["uri"], timeout=240)
    backend = None
    try:
        client._request("configure", {"mode": args.mode})
        backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
        navigate(args, cfg, backend, client, records)
        write_json(args.output / "freeze_audit.json", client._request("audit", {}))
    finally:
        if backend is not None:
            backend.close()
        client.close()


if __name__ == "__main__":
    main()
