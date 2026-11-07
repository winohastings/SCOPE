from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task231.counterfactual import balanced_indices
from Experiment.SCOPE.Stage2.Task23.run import MODEL_PYTHON, PPOFeedbackModel, WebsocketClientPolicy
from Experiment.SCOPE.Stage2.real_rollout import FeedbackBackend, frame_from, image_hash
from Experiment.SCOPE.Stage2.data_guard import STAGE1, write_json
from benchmark.common.config import load_eval_config
from benchmark.common.env.backends import create_environment_backend
from benchmark.common.runner.backend_plan import WorkerBackendPlan
from benchmark.common.runtime_components import build_benchmark_runtime
from benchmark.common.types import EvalEpisode


def history_hash(frames, instruction):
    checksum = hashlib.sha256(instruction.encode())
    for frame in frames:
        checksum.update(image_hash(frame["images"]).encode())
        checksum.update(np.asarray(frame["body_pose"]).tobytes())
    return checksum.hexdigest()


def execute(args, client):
    manifest = json.loads((args.source / "DATA_MANIFEST.json").read_text())
    raw = yaml.safe_load((STAGE1 / "configs/G1.yaml").read_text())
    raw["model"].update(uri=f"ws://127.0.0.1:{args.port}", seed=233)
    raw["benchmark"]["kwargs"].update(split="val_seen", distance_metric="geodesic")
    raw["input"]["split"] = "val_seen"
    raw["env"]["kwargs"].update(split="val_seen", max_episode_steps=200)
    raw["output"].update(root=str(args.output), run_name="matched_chunks", save_step_artifacts=False)
    (args.output / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_eval_config(args.output / "config.yaml")
    runtime = build_benchmark_runtime(cfg)
    backend = create_environment_backend(cfg=cfg.env, worker_backend=WorkerBackendPlan("habitat"), physical_gpu_id=1)
    model = PPOFeedbackModel(client, backend, cfg, "val_seen", False)
    measured = FeedbackBackend(backend, model)
    rows = []
    indices = balanced_indices(manifest["val_seen"], args.episodes)
    write_json(args.output / "selection.json", {"indices": indices, "states_per_episode": 2,
                                               "prefix": "1033 no-Focus deterministic first chunk, independently reset and replayed for every branch"})
    variants = ((14, "none"), (14, "learned"), (28, "learned"), (56, "learned"), (28, "fixed"),
                (28, "random"), (28, "random_seed234"), (28, "random_seed235"))
    try:
        for index in indices:
            episode = EvalEpisode(**manifest["val_seen"][index])
            model.episode = episode
            prefix = []
            for state_index in range(2):
                reference = None
                next_actions = None
                for update, variant in variants:
                    branch = "random" if variant.startswith("random") else variant
                    client._request("restore", {"checkpoint": str((args.source / f"runtime/checkpoints/update_{update:04d}.pt").resolve()),
                                                "diagnostic_mode": "forced4"})
                    client._request("configure", {"split": "val_seen"})
                    backend.start_episode(episode, runtime.initial_pose(episode))
                    observation = backend.get_observation()
                    frames = [frame_from(observation, episode, "val_seen")]
                    for actions in prefix:
                        replayed = backend.apply_action(observation["pose"], np.asarray(actions))
                        observation = replayed.observation
                        for intermediate in replayed.action_observations:
                            frames.append(frame_from(intermediate, episode, "val_seen", len(frames)))
                    frames[-1]["task231_variant"] = branch
                    identifier = f"matched-{index}-{state_index}-{update}-{variant}"
                    model.pending = None
                    model.last_instruction = None
                    seed = 234 if variant == "random_seed234" else 235 if variant == "random_seed235" else 233
                    model.reset(session_id=identifier, episode_id=episode.source_episode_id, statistics_key="vln_train_train", seed=233)
                    frames[-1]["task231_roi_seed"] = seed
                    before = observation["metrics"]["geodesic_distance_to_goal"]
                    response = model.predict(frames=frames, instruction=episode.instruction)
                    diagnostic = response["metadata"]["task231"]
                    identity = {"images": image_hash(observation["images"]), "history": history_hash(frames, episode.instruction),
                                "pose": np.asarray(observation["body_pose"]).tolist(), "pre_focus": diagnostic["pre_focus_sha256"],
                                "bats": diagnostic["context_blocks_sha256"], "grids": diagnostic["context_grids"],
                                "instruction": diagnostic["instruction_sha256"], "history_frames": diagnostic["history_frames"]}
                    if reference is None:
                        reference = copy.deepcopy(identity)
                    valid = identity == reference
                    actions = np.asarray(response["actions"])
                    stopped = model.pending is None
                    after = before
                    if not stopped:
                        result = measured.apply_action(observation["pose"], actions)
                        after = result.observation["metrics"]["geodesic_distance_to_goal"]
                    model.close_session(identifier)
                    row = {"episode_uid": episode.episode_uid, "scene": episode.scene_id, "state_index": state_index,
                           "checkpoint_update": update, "variant": variant, "restoration_valid": valid,
                           "restoration_identity": identity, "distance_before": before, "distance_after": after,
                           "progress": before - after, "navigation_stopped": stopped, "actions": actions.tolist(),
                           "diagnostic": diagnostic, "focus_count": response["metadata"]["stage2"]["focus_count"],
                           "trace_paths": model.report["trace_paths"]}
                    rows.append(row)
                    write_json(args.output / "matched_results.json", rows)
                    print(json.dumps({key: row[key] for key in ("episode_uid", "state_index", "checkpoint_update", "variant", "restoration_valid", "progress")}), flush=True)
                    if update == 14 and variant == "none" and not stopped:
                        next_actions = actions.tolist()
                if next_actions is None:
                    break
                prefix.append(next_actions)
        write_json(args.output / "COMPLETE.json", {"chunks": len(rows), "valid": sum(row["restoration_valid"] for row in rows),
                                                   "freeze_audit": client._request("audit", {}), "training_updates": 0})
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Task23/artifacts/ppo4096")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--port", type=int, default=10232)
    parser.add_argument("--model-gpu", default="1")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT) or not 1 <= args.episodes <= 12:
        parser.error("Task231-only output; 1..12 diagnostic episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "server.log").open("w") as log:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.model_gpu, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        server = subprocess.Popen([MODEL_PYTHON, str(ROOT / "server.py"), "--output", str(args.output / "runtime"),
                                   "--manifest", str(args.source / "DATA_MANIFEST.json"), "--config", str(args.source / "config.json"),
                                   "--port", str(args.port)], stdout=log, stderr=subprocess.STDOUT, env=environment)
        client = None
        try:
            deadline = time.monotonic() + 300
            while not (args.output / "runtime/ready.json").exists():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("model startup failed")
                time.sleep(1)
            client = WebsocketClientPolicy(uri=f"ws://127.0.0.1:{args.port}", timeout=1200)
            client._ws.ping_timeout = 1200
            execute(args, client)
        except BaseException as error:
            write_json(args.output / "INCOMPLETE.json", {"error": repr(error)})
            raise
        finally:
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
