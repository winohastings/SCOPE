from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from compile_configs import ROOT, SIMPLENAV, validate_freeze, write_json
from policies import TOKENS

sys.path.insert(0, str(SIMPLENAV))
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.log.artifacts import ArtifactStore
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runner.worker import _run_episode
from benchmark.common.runtime_components import build_benchmark_runtime, build_model
from benchmark.common.types import EvalEpisode, WorkerPlan
from benchmark.common.log.artifacts import EpisodeArtifactWriter


class MeasuredModel:
    def __init__(self, model, sink, profile=False):
        self.model, self.sink, self.profile = model, sink, profile

    def reset(self, **kwargs):
        return self.model.reset(**kwargs)

    def close_session(self, session_id):
        return self.model.close_session(session_id)

    def predict(self, **kwargs):
        if self.profile:
            kwargs["frames"][-1]["cost_profile"] = True
        started = time.perf_counter()
        response = self.model.predict(**kwargs)
        self.sink.append({**response.get("metadata", {}).get("scope", {}),
                          "rpc_wall_ms": (time.perf_counter() - started) * 1000,
                          "client_started": started, "stop": bool(response.get("stop", False))})
        return response


class MeasuredBackend:
    def __init__(self, backend, records):
        self.backend, self.records = backend, records
        self.reset_ms = 0.0
        self.initial_render_ms = 0.0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def start_episode(self, *args):
        started = time.perf_counter()
        result = self.backend.start_episode(*args)
        self.reset_ms = (time.perf_counter() - started) * 1000
        self.initial_render_ms = self.backend.get_observation().get("scope_render_ms", 0.0)
        fixture = ROOT / "pilot_observation.npz"
        if not fixture.exists():
            observation = self.backend.get_observation()
            np.savez_compressed(fixture, **observation["images"], body_pose=observation["pose"].as_array(),
                                instruction=args[0].instruction)
        return result

    def apply_action(self, *args):
        started = time.perf_counter()
        result = self.backend.apply_action(*args)
        elapsed = (time.perf_counter() - started) * 1000
        observations = result.action_observations or [result.observation]
        render = sum(observation.get("scope_render_ms", 0.0) for observation in observations)
        if self.records:
            self.records[-1].update(sim_step_combined_ms=elapsed, render_observation_wall_ms=render,
                                    sim_nonrender_estimate_ms=max(0.0, elapsed - render),
                                    policy_cycle_wall_ms=(time.perf_counter() - self.records[-1]["client_started"]) * 1000,
                                    actual_waypoint_poses=result.diagnostics.get("actual_waypoint_poses", []))
        return result


def run(group, phase, shard_index=0, shard_count=1):
    validate_freeze()
    if phase == "runs" and not (ROOT / "COST_VALIDATION.json").exists():
        raise RuntimeError("formal evaluation blocked until split-vision profiling is validated")
    if phase == "runs" and not (ROOT / "SESSION_ISOLATION.json").exists():
        raise RuntimeError("formal evaluation blocked until shared-model session isolation is validated")
    cfg = load_eval_config(ROOT / "configs" / f"{group}.yaml")
    episodes = [EvalEpisode(**episode) for episode in json.loads((ROOT / "DEV_EPISODES.json").read_text())]
    if phase in ("smoke", "profile"):
        episodes = episodes[:1]
    indexed_episodes = [(ordinal, episode) for ordinal, episode in enumerate(episodes) if ordinal % shard_count == shard_index]
    episodes = [episode for ordinal, episode in indexed_episodes]
    suffix = f"_shard_{shard_index}" if shard_count > 1 else ""
    run_root = ROOT / phase / group
    completion_path = run_root / f"COMPLETE{suffix}.json"
    if completion_path.exists():
        raise FileExistsError(f"completed run must not be overwritten: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    worker = WorkerPlan(shard_index, 1, episodes, run_root, run_root / f"worker{suffix}.log", WorkerBackendPlan("habitat"),
                        {episode.episode_uid: f"{phase}-{group}-v1" for episode in episodes},
                        {episode.episode_uid: ordinal for ordinal, episode in indexed_episodes})
    write_json(run_root / f"worker_plan{suffix}.json", asdict(worker) | {"run_root": str(run_root), "worker_log_path": str(worker.worker_log_path),
                                                               "backend": worker.backend.to_jsonable()})
    runtime = build_benchmark_runtime(cfg)
    client = build_model(cfg)
    records = []
    backend = MeasuredBackend(create_environment_backend(cfg=cfg.env, worker_backend=worker.backend, physical_gpu_id=1), records)
    model = MeasuredModel(client, records, profile=phase == "profile")
    artifact_times = []
    for method_name in ("write_step_json", "write_eval_info"):
        original = getattr(EpisodeArtifactWriter, method_name)

        def measured_write(self, *args, _original=original, **kwargs):
            begin = time.perf_counter()
            try:
                return _original(self, *args, **kwargs)
            finally:
                artifact_times.append((time.perf_counter() - begin) * 1000)

        setattr(EpisodeArtifactWriter, method_name, measured_write)
    results = []
    started = time.perf_counter()
    try:
        for ordinal, episode in indexed_episodes:
            records.clear()
            artifact_times.clear()
            begin = time.perf_counter()
            try:
                result = _run_episode(cfg=cfg, worker=worker, store=ArtifactStore(run_root), episode=episode,
                                      runtime=runtime, model=model, env_backend=backend)
                row = asdict(result)
            except Exception as error:
                row = {"source_episode_id": episode.source_episode_id, "scene_id": episode.scene_id,
                       "failure": repr(error), "failure_traceback": traceback.format_exc()}
            row.update(group=group, ordinal=ordinal, session_seed=42 + ordinal,
                       episode_wall_inclusive_ms=(time.perf_counter() - begin) * 1000,
                       reset_scene_combined_ms=backend.reset_ms, initial_render_ms=backend.initial_render_ms,
                       artifact_io_ms=sum(artifact_times),
                       prediction_count=len(records))
            for record in records:
                if record.get("current_visual_tokens") != TOKENS[group]:
                    raise ValueError("current token budget mismatch")
                record.update(group=group, ordinal=ordinal, scene_id=episode.scene_id)
                record.setdefault("policy_cycle_wall_ms", record["rpc_wall_ms"])
                record.setdefault("render_observation_wall_ms", 0.0)
                record.pop("client_started", None)
            write_json(run_root / f"episode_{ordinal:03d}.json", row)
            with (run_root / f"steps{suffix}.jsonl").open("a") as stream:
                for record in records:
                    stream.write(json.dumps(record, default=str) + "\n")
            results.append(row)
            print(json.dumps({"group": group, "phase": phase, "ordinal": ordinal,
                              "episode": episode.source_episode_id, "steps": len(records),
                              "failure": row.get("failure"), "seconds": row["episode_wall_inclusive_ms"] / 1000}), flush=True)
        write_json(completion_path, {"group": group, "phase": phase, "count": len(results),
                   "failures": sum(bool(row.get("failure")) for row in results),
                   "wall_seconds": time.perf_counter() - started})
    finally:
        client.close()
        backend.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(TOKENS), required=True)
    parser.add_argument("--phase", choices=("smoke", "profile", "runs"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    run(args.group, args.phase, args.shard_index, args.shard_count)
