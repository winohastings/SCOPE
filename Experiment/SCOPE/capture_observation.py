from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/data/czh/Third_Party/baselines/SimpleNav")
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.worker import load_worker_plan
from benchmark.common.runtime_components import build_benchmark_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker = load_worker_plan(args.worker_plan)
    cfg = load_eval_config(worker.run_root / "config.yaml")
    runtime = build_benchmark_runtime(cfg)
    env = create_environment_backend(cfg=cfg.env, worker_backend=worker.backend, physical_gpu_id=worker.physical_gpu_id)
    try:
        episode = worker.episodes[0]
        env.start_episode(episode, runtime.initial_pose(episode))
        observation = env.get_observation()
        np.savez_compressed(args.output, **observation["images"], body_pose=observation["body_pose"],
                            instruction=np.asarray(episode.instruction), episode_id=np.asarray(episode.source_episode_id))
    finally:
        env.close_episode()
        env.close()


if __name__ == "__main__":
    main()
