from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from Experiment.SCOPE.Stage2.Task21.fusion import CAMERAS
from Experiment.SCOPE.Stage2.Task21.runtime import FusionPolicy
from Experiment.SCOPE.Stage2.Task22.actor import LatentAction
from Experiment.SCOPE.Stage2.Task22.actor_runtime import ActorPolicy
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save, joint_log_probability, load_episode_traces, perception_loop, replay_decisions
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, default=ROOT.parent / "Task21/artifacts/capture")
    args = parser.parse_args()
    policy = ActorPolicy(output=args.output, actor_checkpoint=args.runtime / "actor_initial.pt")
    checkpoint = torch.load(args.runtime / "actor_initial.pt", map_location="cuda", weights_only=True)
    reload_equal = all(torch.equal(value, policy.actor.state_dict()[name]) for name, value in checkpoint["actor"].items())
    paths = sorted((args.runtime / "traces").glob("*.pt"))
    if not paths:
        raise ValueError("real Habitat traces are required")
    episode_trace_counts = [len(load_episode_traces(path)) for path in sorted(args.runtime.glob("episode_boundary_*.pt"))]
    if sum(episode_trace_counts) != len(paths):
        raise ValueError("orphan or unclosed transition traces")
    replay_errors, gradient_errors, counts, cameras, entropy, execution = [], [], Counter(), Counter(), [], []
    for path in paths:
        trace = torch.load(path, weights_only=True)
        if trace["feedback"] is None:
            raise AssertionError("incomplete navigation transition")
        if trace["actor_checkpoint_sha256"] != sha256(args.runtime / "actor_initial.pt"):
            raise AssertionError("trace belongs to another Actor checkpoint")
        if trace["provenance"] != policy.provenance:
            raise AssertionError("trace provenance mismatch")
        decisions = replay_decisions(trace, "cuda")
        with torch.no_grad():
            replayed, _ = joint_log_probability(policy.actor, decisions)
        replay_errors.append(abs(float(replayed.cpu() - trace["old_sequence_log_probability"])))
        policy.actor.train()
        replayed, joint_entropy = joint_log_probability(policy.actor, decisions)
        gradient_errors.append(abs(float(replayed.detach().cpu() - trace["old_sequence_log_probability"])))
        policy.actor.zero_grad(set_to_none=True)
        (-replayed - .01 * joint_entropy).backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.actor.parameters()):
            raise AssertionError("nonfinite replay gradient")
        policy.actor.eval()
        counts[(trace["mode"], trace["focus_count"]) ] += 1
        cameras.update(record["action"]["camera"] for record in trace["decisions"] if not record["action"]["stop"])
        entropy.extend(record["entropy"] for record in trace["decisions"])
        execution.append({"trace": str(path), "episode": trace["episode_id"], "mode": trace["mode"],
                          "executed": trace["feedback"]["action_executed"], "terminated": trace["terminated"],
                          "truncated": trace["truncated"], "navigation_done": trace["feedback"]["navigation_done"],
                          "focus_count": trace["focus_count"], "perception_termination": trace["termination"],
                          "current_tokens": trace["current_visual_tokens"], "history_blocks": len(trace["context_grids"]) - 4,
                          "actual_displacement": trace["feedback"]["actual_body_displacement"]})
    policy.actor.zero_grad(set_to_none=True)
    records = json.loads((args.capture / "manifest.json").read_text())[:2]
    nofocus, interventions, reproducibility = [], [], []
    for record in records:
        source = args.capture / record["path"]
        if sha256(source) != record["sha256"]:
            raise ValueError("capture hash mismatch")
        with np.load(source) as archive:
            images = {camera: archive[camera].copy() for camera in CAMERAS}
            pose = archive["body_pose"].copy()
        instruction = record["instruction"]
        language = policy.language(instruction)
        frame = {"images": images, "body_pose": pose, "policy_mode": "glance_only"}
        baseline = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=222)
        incremental = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=222)
        for tick in range(3):
            policy.enabled = False
            reference = FusionPolicy.predict_session(policy, baseline, frames=[frame], instruction=instruction)
            tokens = policy.last_current.clone()
            grids = policy.last_grids
            policy.enabled = True
            actual = FusionPolicy.predict_session(policy, incremental, frames=[frame], instruction=instruction,
                                                   controller=lambda perception: None)
            nofocus.append({"episode": record["source_episode_id"], "tick": tick,
                            "token_max_abs": float((tokens - policy.last_current).abs().max()),
                            "action_max_abs": float(np.abs(reference["actions"] - actual["actions"]).max()),
                            "grids_equal": grids == policy.last_grids,
                            "stop_equal": reference["stop"] == actual["stop"]})
        base_features, base_actions = None, None
        for count in range(5):
            session = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=222)
            requests = [LatentAction(False, camera, torch.tensor([(-1.) ** camera, .5])).focus() for camera in range(count)]
            intermediate = []
            def controller(perception):
                intermediate.append(perception.current.detach().cpu().clone())
                for focus in requests:
                    perception.acquire(focus)
                    intermediate.append(perception.current.detach().cpu().clone())
            response = FusionPolicy.predict_session(policy, session, frames=[frame], instruction=instruction, controller=controller)
            features = policy.last_current.clone()
            if count == 0:
                base_features, base_actions = features, response["actions"].copy()
            row = {"episode": record["source_episode_id"], "focus_count": count,
                   "feature_max_abs_from_no_focus": float((features - base_features).abs().max()),
                   "action_max_abs_from_no_focus": float(np.abs(response["actions"] - base_actions).max()),
                   "current_tokens": response["metadata"]["scope"]["current_visual_tokens"],
                   "roi": [focus.roi for focus in requests]}
            interventions.append(row)
            atomic_save(args.output / f"intervention_{record['source_episode_id']}_{count}.pt",
                        {"diagnostic_only": True, "ppo_eligible": False, "intermediate_features": intermediate,
                         "actions": torch.from_numpy(response["actions"].copy()), "summary": row})
        for mode in ("deterministic", "training-sampling"):
            results, traces = [], []
            for repeat in range(2):
                session = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=222)
                generator = torch.Generator(device="cuda").manual_seed(223)
                def controller(perception):
                    traces.append(perception_loop(policy.actor, perception, language, mode=mode, generator=generator,
                                                  transition_id=f"diagnostic:{record['source_episode_id']}:{mode}"))
                results.append(FusionPolicy.predict_session(policy, session, frames=[frame], instruction=instruction, controller=controller))
            difference = float(np.abs(results[0]["actions"] - results[1]["actions"]).max())
            reproducibility.append({"episode": record["source_episode_id"], "mode": mode, "action_max_abs": difference,
                                    "focus_counts": [trace["focus_count"] for trace in traces],
                                    "logprob_max_abs": float(abs(traces[0]["old_sequence_log_probability"] - traces[1]["old_sequence_log_probability"]))})
    audit = policy.freeze_audit()
    entropy_mean = {key: float(np.mean([row[key] for row in entropy])) for key in entropy[0]}
    passed = (reload_equal and max(replay_errors) < 1e-5 and max(gradient_errors) < 1e-5
              and all(row["token_max_abs"] == row["action_max_abs"] == 0 and row["grids_equal"] and row["stop_equal"] for row in nofocus)
              and all(row["action_max_abs"] == row["logprob_max_abs"] == 0 for row in reproducibility)
              and all(row["current_tokens"] == 64 for row in interventions + execution)
              and all(row["feature_max_abs_from_no_focus"] > 0 and row["action_max_abs_from_no_focus"] > 0
                      for row in interventions if row["focus_count"] > 0)
              and any(row["executed"] and row["perception_termination"] == "policy_stop" for row in execution)
              and all(audit.values()))
    write_json(args.output / "VALIDATION.json", {"passed": passed, "reload_equal": reload_equal,
               "actor_parameters": sum(parameter.numel() for parameter in policy.actor.parameters()),
               "replay_max_error": max(replay_errors), "grad_enabled_replay_max_error": max(gradient_errors),
               "focus_histogram": {f"{mode}:{count}": total for (mode, count), total in counts.items()},
               "camera_counts": dict(cameras), "entropy_mean": entropy_mean, "execution": execution,
               "no_focus": nofocus, "interventions": interventions, "reproducibility": reproducibility,
               "freeze_audit": audit, "episode_trace_counts": episode_trace_counts, "optimizer_updates": 0})
    if not passed:
        raise AssertionError("Task2.2 validation failed")
    print(json.dumps({"passed": passed, "transitions": len(paths), "entropy": entropy_mean}, indent=2), flush=True)


if __name__ == "__main__":
    main()
