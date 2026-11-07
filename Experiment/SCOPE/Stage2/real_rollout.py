from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from data_guard import ROOT, STAGE1, episodes, write_json

SIMPLENAV = Path("/home/data/czh/Third_Party/baselines/SimpleNav")
sys.path.insert(0, str(SIMPLENAV))
sys.path.insert(0, str(ROOT.parents[2]))
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.log.artifacts import ArtifactStore
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runner.worker import _run_episode, observation_frame
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode, WorkerPlan
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def frame_from(observation, episode, purpose, tick=0):
    frame = observation_frame(observation, frame_id=tick, timestamp_s=tick, pose=observation["pose"])
    frame.update(stage2_purpose=purpose, stage2_scene=episode.scene_id)
    return frame


class FeedbackModel:
    reward_purposes = ("train",)

    def __init__(self, client, backend, cfg, purpose, update):
        self.client, self.backend, self.cfg = client, backend, cfg
        self.purpose, self.update = purpose, update
        self.pending = None
        self.report = None
        self.last_instruction = None
        self.episode = None
        self.collision_events = []
        original_step = backend.runtime.step

        def measured_step(payload):
            simulator = backend.runtime.env.sim
            original_filter = simulator.step_filter
            filtered_contact = []

            def measured_filter(start, requested):
                actual = original_filter(start, requested)
                filtered_contact.append(bool(np.linalg.norm(np.asarray(actual) - np.asarray(requested)) > 1e-4))
                return actual

            simulator.step_filter = measured_filter
            try:
                result = original_step(payload)
                self.collision_events.append(any(filtered_contact))
                return result
            finally:
                simulator.step_filter = original_filter

        backend.runtime.step = measured_step

    def reset(self, **kwargs):
        return self.client.reset(**kwargs)

    def predict(self, *, frames, instruction):
        if self.pending is not None:
            raise RuntimeError("previous action feedback missing")
        for frame in frames:
            frame.update(stage2_purpose=self.purpose, stage2_scene=self.episode.scene_id)
        self.last_instruction = instruction
        before = self.backend.get_observation()
        response = self.client.predict(frames=frames, instruction=instruction)
        actions = np.asarray(response["actions"])
        motion = float(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean())
        self.pending = (before, response)
        if response.get("stop") or motion <= self.cfg.raw["stop_threshold"]:
            self.feedback(before, [], True, "waypoint_motion_stop", False, False)
        return response

    def feedback(self, after, actual_poses, done, termination, executed, collision):
        before, response = self.pending
        record = response["metadata"]["stage2"]
        actions = np.asarray(response["actions"])
        converted = []
        for row in actual_poses:
            pose = np.asarray(row).copy()
            pose[3] = (pose[3] + np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
            converted.append(pose.tolist())
        execution = {"before": before["body_pose"].tolist(), "after": after["body_pose"].tolist(),
                     "predicted_waypoints": actions.tolist(), "executed_poses": converted,
                     "current_tokens": record.get("current_visual_tokens", 64 + 64 * len(record["bank"]["selected"])),
                     "perception_ms": record["perception_ms"], "model_ms": record["model_ms"],
                     "navigation_done": done, "termination": termination, "action_executed": executed}
        teacher = None
        if self.purpose in self.reward_purposes:
            metrics = after["metrics"]
            teacher = {"distance_before": before["metrics"]["geodesic_distance_to_goal"],
                       "distance_after": metrics["geodesic_distance_to_goal"], "collision": collision,
                       "terminal": done and termination != "environment_time_limit", "success": metrics["geodesic_distance_to_goal"] < 3,
                       "spl": float(metrics["spl"])}
        result = self.client._request("feedback", {"execution": execution, "teacher": teacher})
        self.pending = None
        return result

    def close_session(self, session_id):
        if self.pending is not None:
            raise RuntimeError("incomplete navigation transition; do not train on infrastructure failure")
        observation = self.backend.get_observation()
        self.report = self.client._request("finish", {"bootstrap_frame": frame_from(observation, self.episode, self.purpose),
                                                      "instruction": self.last_instruction, "update": self.update})
        self.client.close_session(session_id)


class FeedbackBackend:
    def __init__(self, backend, model):
        self.backend, self.model = backend, model

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def apply_action(self, before, actions):
        self.model.collision_events.clear()
        result = self.backend.apply_action(before, actions)
        native_steps = self.backend.runtime.env._elapsed_steps
        time_limit = result.data_done and native_steps >= int(self.model.cfg.env.kwargs["max_episode_steps"])
        self.model.feedback(result.observation, result.diagnostics["actual_waypoint_poses"], result.data_done,
                            "environment_time_limit" if time_limit else ("environment_done" if result.data_done else "running"),
                            True, any(self.model.collision_events))
        return result


def navigate(args, cfg, backend, client, records):
    runtime = build_benchmark_runtime(cfg)
    model = FeedbackModel(client, backend, cfg, args.purpose, args.update)
    measured = FeedbackBackend(backend, model)
    results = []
    for ordinal, record in enumerate(records, start=args.episode_start):
        episode = EvalEpisode(**record)
        model.episode = episode
        worker = WorkerPlan(0, 1, [episode], args.output, args.output / "worker.log", WorkerBackendPlan("habitat"),
                            {episode.episode_uid: "stage2-pilot"}, {episode.episode_uid: ordinal})
        begin = time.perf_counter()
        result = _run_episode(cfg=cfg, worker=worker, store=ArtifactStore(args.output), episode=episode,
                              runtime=runtime, model=model, env_backend=measured)
        row = asdict(result) | {"wall_seconds": time.perf_counter() - begin, "training": model.report}
        results.append(row)
        write_json(args.output / "results.json", results)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if result.failure:
            raise RuntimeError(result.failure)


def image_hash(images):
    digest = hashlib.sha256()
    for camera in ("front", "left", "right", "rear"):
        digest.update(images[camera].tobytes())
    return digest.hexdigest()


def diagnose(args, cfg, backend, client, records):
    runtime = build_benchmark_runtime(cfg)
    variants = [("glance", [], False), ("fixed", [0, 3], False), ("fixed_spatial", [0, 3], True),
                ("cross_camera", [0, 4], False), ("cross_spatial", [0, 4], True),
                ("right_rear", [8, 12], False), ("fixed_low_detail", [0, 3], False)]
    variants.append(("fixed_native_scatter", [0, 3], False))
    variants.append(("single_front", [0], False))
    results = []
    for record in records:
        episode = EvalEpisode(**record)
        prefix = []
        expected_hashes = {}
        for state_index in range(args.max_steps):
            for history_enabled in ((True,) if state_index == 0 else (True, False)):
                for name, selected, annotated in variants:
                    begin = time.perf_counter()
                    backend.start_episode(episode, runtime.initial_pose(episode))
                    observation = backend.get_observation()
                    frames = [frame_from(observation, episode, args.purpose)]
                    for actions in prefix:
                        execution = backend.apply_action(observation["pose"], np.asarray(actions))
                        observation = execution.observation
                        for intermediate in execution.action_observations:
                            frames.append(frame_from(intermediate, episode, args.purpose, len(frames)))
                    actual_hash = image_hash(observation["images"])
                    expected_hashes.setdefault(state_index, actual_hash)
                    if actual_hash != expected_hashes[state_index]:
                        raise AssertionError("counterfactual branches differ in their fixed visual state")
                    current = frames[-1]
                    current.update(stage2_forced=selected, stage2_annotate=annotated,
                                   stage2_low_detail=name.endswith("low_detail"), stage2_no_history=not history_enabled,
                                   stage2_diffusion_seed=4242,
                                   stage2_verify_reencode=name in ("fixed", "fixed_native_scatter"),
                                   stage2_use_native_features=name == "fixed_native_scatter")
                    identifier = f"diag-{args.purpose}-{episode.source_episode_id}-{state_index}-{history_enabled}-{name}"
                    client.reset(session_id=identifier, episode_id=episode.source_episode_id, statistics_key="vln_train_train", seed=42)
                    response = client.predict(frames=frames if history_enabled else [current], instruction=episode.instruction)
                    actions = np.asarray(response["actions"])
                    distance_before = observation["metrics"]["geodesic_distance_to_goal"]
                    motion = float(np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1).mean())
                    if motion <= cfg.raw["stop_threshold"]:
                        distance_after = distance_before
                    else:
                        execution = backend.apply_action(observation["pose"], actions)
                        distance_after = execution.observation["metrics"]["geodesic_distance_to_goal"]
                    row = {"purpose": args.purpose, "episode": episode.source_episode_id, "scene": episode.scene_id,
                           "state_index": state_index, "history": history_enabled, "variant": name,
                           "image_sha256": actual_hash, "distance_before": distance_before, "distance_after": distance_after,
                           "progress": distance_before - distance_after, "actions": actions.tolist(),
                           "metadata": response["metadata"]["stage2"], "wall_seconds": time.perf_counter() - begin}
                    results.append(row)
                    if history_enabled and name == "glance" and state_index == len(prefix):
                        next_actions = actions.tolist()
                    client.close_session(identifier)
                    write_json(args.output / "diagnostics.json", results)
                    print(json.dumps({key: value for key, value in row.items() if key not in ("actions", "metadata")}), flush=True)
            prefix.append(next_actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("train", "dev"), required=True)
    parser.add_argument("--mode", choices=("navigate", "diagnose"), default="navigate")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--port", type=int, default=10096)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--regime", choices=("A", "B", "C"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--episode-start", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.count <= 2 or not 1 <= args.max_steps <= (16 if args.purpose == "train" else 200):
        parser.error("small pilot only: count<=2, train steps<=16, dev steps<=200")
    if args.mode == "diagnose" and args.max_steps > 2:
        parser.error("controlled diagnostic states limited to two per episode")
    if args.update and (args.purpose != "train" or args.mode != "navigate"):
        parser.error("updates only on train navigation rollouts")
    args.output.mkdir(parents=True, exist_ok=False)
    raw = yaml.safe_load((STAGE1 / "configs/G1.yaml").read_text())
    raw["model"]["uri"] = f"ws://127.0.0.1:{args.port}"
    raw["benchmark"]["max_steps"] = args.max_steps
    raw["output"]["root"] = str(args.output)
    raw["output"]["run_name"] = f"stage2-{args.purpose}"
    (args.output / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    if args.episode_start < 0:
        parser.error("episode start must be nonnegative")
    records = episodes(args.purpose, args.episode_start + args.count)[args.episode_start:]
    write_json(args.output / "episodes.json", records)
    cfg = load_eval_config(args.output / "config.yaml")
    client = WebsocketClientPolicy(uri=raw["model"]["uri"], timeout=240)
    if args.regime:
        client._request("configure", {"regime": args.regime})
    if args.resume:
        cursor = client._request("restore", {"checkpoint": str(args.resume)})
        if args.purpose == "train":
            args.episode_start = cursor["next_training_episode_index"]
            records = episodes(args.purpose, args.episode_start + args.count)[args.episode_start:]
            write_json(args.output / "episodes.json", records)
        write_json(args.output / "resume_cursor.json", cursor)
    backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
    try:
        function = navigate if args.mode == "navigate" else diagnose
        function(args, cfg, backend, client, records)
    finally:
        backend.close()
        client.close()


if __name__ == "__main__":
    main()
