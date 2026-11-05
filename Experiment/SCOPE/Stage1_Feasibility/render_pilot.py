from __future__ import annotations

import json
import sys
import time

import numpy as np

from compile_configs import ROOT, SIMPLENAV, write_json

sys.path.insert(0, str(SIMPLENAV))
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode


def main():
    cfg = load_eval_config(ROOT / "configs/G0.yaml")
    episode = EvalEpisode(**json.loads((ROOT / "DEV_EPISODES.json").read_text())[0])
    backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
    try:
        reset = time.perf_counter()
        backend.start_episode(episode, build_benchmark_runtime(cfg).initial_pose(episode))
        reset_ms = (time.perf_counter() - reset) * 1000
        measurements = []
        for index in range(220):
            begin = time.perf_counter()
            images = backend.runtime.env.sim.get_sensor_observations()
            measurements.append({"index": index, "phase": "warmup" if index < 20 else "timed",
                                 "render_observation_wall_ms": (time.perf_counter() - begin) * 1000})
        values = [row["render_observation_wall_ms"] for row in measurements if row["phase"] == "timed"]
        write_json(ROOT / "RENDER_PILOT.json", {"episode_id": episode.source_episode_id, "reset_scene_combined_ms": reset_ms,
                   "warmup": 20, "timed": 200, "p50_ms": float(np.percentile(values, 50)),
                   "p90_ms": float(np.percentile(values, 90)), "measurements": measurements,
                   "scope": "fixed-state four-camera 512 replay; sensor readback included, not pure GPU raster kernel; no navigation state advanced"})
    finally:
        backend.close()


if __name__ == "__main__":
    main()
