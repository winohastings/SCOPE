from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from Experiment.SCOPE.Stage2.Task21.runtime import FusionPolicy
from Experiment.SCOPE.Stage2.runtime import Stage2Policy
from Experiment.SCOPE.Stage2.data_guard import PARTITION, check_episode, sha256, write_json
from Experiment.SCOPE.Stage2.trajectory import ExecutionFeedback
from .actor import ActorState, SpatialQueryActor
from .trajectory import SCHEMA, atomic_save, joint_log_probability, perception_loop, replay_decisions

ROOT = Path(__file__).resolve().parent


class ActorPolicy(FusionPolicy):
    language = Stage2Policy.language

    def __init__(self, *, output, seed=222, mode="training-sampling", actor_checkpoint=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.config = json.loads((ROOT / "config.json").read_text())
        checkpoint = ROOT.parents[3] / self.config["adapter_checkpoint"]
        super().__init__(output=self.output, checkpoint=checkpoint)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            self.actor = SpatialQueryActor().cuda().eval()
        self.actor_mode = mode
        if mode not in ("deterministic", "training-sampling"):
            raise ValueError("invalid Actor mode")
        self.provenance = {"backbone_sha256": self.backbone_hash, "adapter_sha256": sha256(checkpoint),
                           "partition_sha256": sha256(PARTITION), "schema": SCHEMA}
        if actor_checkpoint is not None:
            saved = torch.load(actor_checkpoint, map_location="cuda", weights_only=True)
            if saved["provenance"] != self.provenance or saved["config"] != self.config:
                raise ValueError("Actor checkpoint contract mismatch")
            self.actor.load_state_dict(saved["actor"])
        if not all(torch.isfinite(parameter).all() for parameter in self.actor.parameters()):
            raise ValueError("nonfinite Actor checkpoint")
        actor_path = self.output / "actor_initial.pt"
        atomic_save(actor_path, {"actor": self.actor.state_dict(), "config": self.config, "provenance": self.provenance,
                                 "seed": seed, "episode_boundary": True, "optimizer_updates": 0})
        self.actor_hash = sha256(actor_path)
        self.initial_actor = {name: parameter.detach().clone() for name, parameter in self.actor.named_parameters()}
        self.backbone_versions = {name: parameter._version for name, parameter in self.model.named_parameters()}
        self.fusion_initial = {name: parameter.detach().clone() for name, parameter in self.fusion.named_parameters()}
        self.initialized_seed = seed
        self.summary_records = []
        write_json(self.output / "model_contract.json", {"actor_parameters": sum(parameter.numel() for parameter in self.actor.parameters()),
                   "provenance": self.provenance, "mode": self.actor_mode, "seed": seed,
                   "source_hashes": {str(path): sha256(path) for path in ROOT.glob("*.py")}})

    def new_session(self, **kwargs):
        session = super().new_session(**kwargs)
        session.update(actor_generator=torch.Generator(device="cuda").manual_seed(session["seed"]),
                       pending=None, trace_paths=[], instruction=None, language=None, terminal=False,
                       run_id=uuid.uuid4().hex, completed=0)
        return session

    def validate_episode(self, session, frame):
        check_episode(session["episode_id"], frame["stage2_scene"], frame["stage2_purpose"])

    def predict_session(self, session, *, frames, instruction):
        if session["pending"] is not None or session["terminal"]:
            raise RuntimeError("require previous action feedback and live navigation episode")
        if session["prediction_index"] >= self.config["max_navigation_transitions_per_episode"]:
            raise RuntimeError("explicit episode trace limit reached; finish instead of implicit truncation")
        frame = frames[-1]
        self.validate_episode(session, frame)
        if session["instruction"] is None:
            session["instruction"] = instruction
            session["language"] = self.language(instruction)
        elif instruction != session["instruction"]:
            raise ValueError("instruction changed inside an episode")
        transition_id = f"{session['run_id']}:{session['episode_id']}:{session['prediction_index']}"
        trace = None

        def controller(perception):
            nonlocal trace
            trace = perception_loop(self.actor, perception, session["language"], mode=self.actor_mode,
                                    generator=session["actor_generator"], transition_id=transition_id)

        torch.cuda.synchronize()
        began = time.perf_counter()
        response = super().predict_session(session, frames=frames, instruction=instruction, controller=controller)
        torch.cuda.synchronize()
        if trace is None:
            raise RuntimeError("incremental Actor did not run")
        if response["metadata"]["scope"]["current_visual_tokens"] != 64:
            raise AssertionError("current visual token budget changed")
        trace.update(episode_id=session["episode_id"], scene_id=frame["stage2_scene"],
                     provenance=self.provenance, actor_checkpoint_sha256=self.actor_hash,
                     navigation_actions=torch.from_numpy(response["actions"].copy()), navigation_stop=bool(response["stop"]),
                     frame_count=len(frames), body_pose=torch.as_tensor(np.asarray(frame["body_pose"])).clone(),
                     current_visual_tokens=64, context_grids=self.last_grids,
                     elapsed_model_ms=(time.perf_counter() - began) * 1000)
        with torch.no_grad():
            replayed, _ = joint_log_probability(self.actor, replay_decisions(trace, "cuda"))
        trace["replay_logprob_error"] = abs(float(replayed.cpu() - trace["old_sequence_log_probability"]))
        if trace["replay_logprob_error"] > 1e-5:
            raise AssertionError("stored Actor probability replay mismatch")
        path = self.output / "traces" / f"{session['run_id']}_{session['prediction_index'] - 1:04d}.pt"
        atomic_save(path, trace)
        session["pending"] = (path, trace)
        session["trace_paths"].append(str(path))
        summary = {"transition_id": transition_id, "trace_path": str(path), "mode": self.actor_mode,
                   "focus_count": trace["focus_count"], "perception_termination": trace["termination"],
                   "current_visual_tokens": 64, "perception_ms": self.last_focus_ms,
                   "model_ms": trace["elapsed_model_ms"], "replay_logprob_error": trace["replay_logprob_error"],
                   "bank": {"selected": [record["action"]["camera"] for record in trace["decisions"] if not record["action"]["stop"]]},
                   "entropy": [record["entropy"] for record in trace["decisions"]]}
        response["metadata"]["stage2"] = summary
        response["metadata"]["fusion"]["focus_count"] = trace["focus_count"]
        self.summary_records.append(summary)
        write_json(self.output / "trace_summary.json", self.summary_records)
        return response

    def complete_step(self, session, *, execution, teacher=None):
        if session["pending"] is None:
            raise RuntimeError("no pending navigation action")
        feedback = ExecutionFeedback(**execution)
        if feedback.current_tokens != 64:
            raise ValueError("feedback must use fixed64 contract")
        path, trace = session["pending"]
        if not np.array_equal(np.asarray(feedback.predicted_waypoints, dtype=np.float32), trace["navigation_actions"].numpy()):
            raise ValueError("feedback action differs from issued action")
        if not np.allclose(feedback.before, trace["body_pose"].numpy(), rtol=0, atol=1e-5):
            raise ValueError("feedback starts from wrong pose")
        trace["feedback"] = feedback.record()
        trace["teacher_for_reward_only"] = teacher
        trace["terminated"] = feedback.navigation_done and feedback.termination != "environment_time_limit"
        trace["truncated"] = feedback.termination == "environment_time_limit"
        atomic_save(path, trace)
        session["pending"] = None
        session["terminal"] = feedback.navigation_done
        session["completed"] += 1
        return {"completed_steps": session["completed"], "updated": False}

    def finish(self, session, *, bootstrap_frame=None, instruction=None, update=False):
        if update:
            raise PermissionError("Task2.2 does not authorize RL updates")
        if session["pending"] is not None:
            raise RuntimeError("cannot finish incomplete navigation feedback")
        bootstrap = None
        if session["trace_paths"]:
            path = Path(session["trace_paths"][-1])
            last = torch.load(path, weights_only=True)
            if not last["terminated"]:
                if bootstrap_frame is None or instruction != session["instruction"]:
                    raise ValueError("truncation requires actual successor observation")
                from starVLA.model.modules.input_context import prepare_scope_images
                from Experiment.SCOPE.Stage2.Task21.fusion import CAMERAS
                _, _, glance, _ = prepare_scope_images(bootstrap_frame["images"])
                encoded = torch.stack([tokens for tokens, grid in self.native_encode([glance[camera] for camera in CAMERAS])])
                bootstrap = ActorState(encoded, session["language"], encoded, encoded.new_zeros(4, 16, 1),
                                       encoded.new_zeros(4, 7), 0).packed()
                last["truncated"] = True
                last["bootstrap_state"] = bootstrap
                last["bootstrap_encoding"] = "four-current-images-only; recorded separately from original mixed-history batch"
                atomic_save(path, last)
        audit = self.freeze_audit()
        report = {"episode_id": session["episode_id"], "transitions": session["completed"], "updated": False,
                  "trace_paths": session["trace_paths"], "bootstrap_saved": bootstrap is not None, **audit}
        atomic_save(self.output / f"episode_boundary_{session['run_id']}.pt",
                    {"provenance": self.provenance, "actor_checkpoint_sha256": self.actor_hash,
                     "episode_boundary": True, "cursor": {"finished_episode": session["episode_id"]},
                     "action_generator": session["actor_generator"].get_state(), "report": report})
        session["terminal"] = True
        write_json(self.output / f"finish_{session['run_id']}.json", report)
        return report

    def freeze_audit(self):
        result = {"actor_unchanged": all(torch.equal(parameter, self.initial_actor[name]) for name, parameter in self.actor.named_parameters()),
                  "backbone_unchanged": all(parameter._version == self.backbone_versions[name] for name, parameter in self.model.named_parameters()),
                  "backbone_frozen_no_grad": all(not parameter.requires_grad and parameter.grad is None for parameter in self.model.parameters()),
                  "adapter_frozen_unchanged": all(not parameter.requires_grad and parameter.grad is None
                                                 and torch.equal(parameter, self.fusion_initial[name]) for name, parameter in self.fusion.named_parameters())}
        if not all(result.values()):
            raise AssertionError("Task2.2 inference mutated frozen weights")
        return result
