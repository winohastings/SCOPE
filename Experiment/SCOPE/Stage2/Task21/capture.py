from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from data_guard import STAGE1, episodes, sha256, write_json

sys.path.insert(0, "/home/data/czh/Third_Party/baselines/SimpleNav")
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_eval_config(STAGE1 / "configs/G1.yaml")
    runtime = build_benchmark_runtime(cfg)
    records = episodes("train", settings["capture_episodes"])
    backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
    captured = []
    try:
        for record in records:
            episode = EvalEpisode(**record)
            backend.start_episode(episode, runtime.initial_pose(episode))
            observation = backend.get_observation()
            destination = args.output / f"episode_{episode.source_episode_id}.npz"
            np.savez_compressed(destination, **observation["images"], body_pose=observation["body_pose"])
            captured.append({"path": destination.name, "sha256": sha256(destination), **record})
            write_json(args.output / "manifest.json", captured)
            backend.close_episode()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
