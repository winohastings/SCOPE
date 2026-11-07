from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from Experiment.SCOPE.Stage2.Task22.actor_runtime import ActorPolicy
from Experiment.SCOPE.Stage2.Task22.actor import ActorState
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save, load_episode_traces
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from .data import load_manifest
from .learning import PPOTrainer, navigation_reward

ROOT = Path(__file__).resolve().parent


class PPOPolicy(ActorPolicy):
    def __init__(self, *, output, config, manifest):
        settings = json.loads(Path(config).read_text())
        if settings["mode"] != "PPO-only" or settings["group_sampling"]:
            raise ValueError("Task2.3 supports PPO-only, without group branches")
        self.manifest = load_manifest(manifest)
        self.allowed = {split: {record["source_episode_id"]: record for record in self.manifest[split]}
                        for split in ("train", "val_seen")}
        initial_actor = ROOT.parents[3] / settings["initial_actor"]
        super().__init__(output=output, seed=settings["seed"], actor_checkpoint=initial_actor)
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        self.training_config = settings
        self.trainer = PPOTrainer(self.actor, settings)
        self.actor = self.trainer.old_actor
        self.provenance = self.provenance | {"training_schema": settings["schema"], "data_manifest_sha256": sha256(manifest)}
        self.buffer_boundaries = []
        self.buffer = []
        self.collected = 0
        self.validation_transitions = 0
        self.split = "train"
        self.cursor = {"train_episode_index": 0, "validation_round": 0, "next_validation_at": 0}
        self.best = None
        self.checkpoint_history = []
        self.last_episode = None
        self.publish_snapshot()
        write_json(self.output / "training_config.json", settings)
        write_json(self.output / "training_contract.json", {"actor_parameters": sum(parameter.numel() for parameter in self.trainer.actor.parameters()),
                   "critic_parameters": sum(parameter.numel() for parameter in self.trainer.critic.parameters()),
                   "provenance": self.provenance, "ratio_unit": "hierarchical perception decision",
                   "reduction": "sum decisions within sequence, mean navigation states", "group_sampling": False})

    def publish_snapshot(self):
        path = self.output / "snapshots" / f"actor_{self.trainer.updates:04d}.pt"
        if not path.exists():
            atomic_save(path, {"actor": self.actor.state_dict(), "config": self.config, "provenance": self.provenance,
                               "policy_version": self.trainer.updates, "old_critic": self.trainer.old_critic.state_dict()})
        self.actor_hash = sha256(path)

    def validate_episode(self, session, frame):
        split = frame["stage2_purpose"]
        if split != self.split or split not in self.allowed:
            raise PermissionError("wrong or forbidden dataset split")
        record = self.allowed[split].get(session["episode_id"])
        if record is None or record["scene_id"] != frame["stage2_scene"]:
            raise PermissionError("episode/scene outside frozen manifest")
        session["purpose"] = split

    def predict_session(self, session, *, frames, instruction):
        expected = self.allowed[self.split].get(session["episode_id"])
        if expected is None or instruction.strip() != expected["instruction"].strip():
            raise PermissionError(f"instruction/episode differs from manifest: {session['episode_id']} {instruction!r}")
        response = super().predict_session(session, frames=frames, instruction=instruction)
        path, trace = session["pending"]
        if tuple(trace["navigation_actions"].shape) != (8, 4):
            raise ValueError("navigation action chunk must preserve released 8x4 contract")
        with torch.no_grad():
            value = float(self.trainer.old_critic(ActorState.unpack(trace["decisions"][0]["state"], "cuda")))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite old Critic prediction")
        trace.update(old_value=value, policy_version=self.trainer.updates, purpose=session["purpose"])
        atomic_save(path, trace)
        return response

    def finish(self, session, *, bootstrap_frame=None, instruction=None, update=False):
        if update:
            raise ValueError("updates only after closing episodes, using update RPC")
        report = super().finish(session, bootstrap_frame=bootstrap_frame, instruction=instruction, update=False)
        boundary = self.output / f"episode_boundary_{session['run_id']}.pt"
        traces = load_episode_traces(boundary)
        if session["purpose"] == "train":
            prepared = self.trainer.prepare_episode(traces)
            self.buffer.extend(prepared)
            self.buffer_boundaries.append(str(boundary))
            self.collected += len(traces)
        else:
            self.validation_transitions += len(traces)
            for trace in traces:
                trace["navigation_reward"] = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                                                terminated=trace["terminated"], truncated=trace["truncated"])
        for path, trace in zip(session["trace_paths"], traces, strict=True):
            atomic_save(path, trace)
        focus = Counter(trace["focus_count"] for trace in traces)
        camera = Counter(decision["action"]["camera"] for trace in traces for decision in trace["decisions"] if not decision["action"]["stop"])
        entropies = [decision["entropy"] for trace in traces for decision in trace["decisions"]]
        self.last_episode = {"purpose": session["purpose"], "episode_id": session["episode_id"], "boundary": str(boundary),
                             "transitions": len(traces), "reward_sum": sum(trace["navigation_reward"] for trace in traces),
                             "focus_histogram": {str(key): value for key, value in focus.items()},
                             "camera_histogram": {str(key): value for key, value in camera.items()},
                             "entropy": {key: float(np.mean([entropy[key] for entropy in entropies]))
                                         for key in ("stop", "camera", "position_latent")},
                             "start_geodesic_distance": traces[0]["teacher_for_reward_only"]["distance_before"],
                             "final_geodesic_distance": traces[-1]["teacher_for_reward_only"]["distance_after"],
                             "terminated": traces[-1]["terminated"], "truncated": traces[-1]["truncated"],
                             "model_ms": sum(trace["elapsed_model_ms"] for trace in traces),
                             "perception_ms": sum(trace["perception_ms"] for trace in traces)}
        report.update(episode_summary=self.last_episode, collected=self.collected, buffer_transitions=len(self.buffer))
        write_json(self.output / f"finish_{session['run_id']}.json", report)
        return report

    def freeze_audit(self):
        result = {"backbone_unchanged": all(parameter._version == self.backbone_versions[name] for name, parameter in self.model.named_parameters()),
                  "backbone_frozen_no_grad": all(not parameter.requires_grad and parameter.grad is None for parameter in self.model.parameters()),
                  "adapter_frozen_unchanged": all(not parameter.requires_grad and parameter.grad is None
                                                 and torch.equal(parameter, self.fusion_initial[name]) for name, parameter in self.fusion.named_parameters()),
                  "old_snapshots_frozen": all(not parameter.requires_grad and parameter.grad is None
                                              for model in (self.trainer.old_actor, self.trainer.old_critic) for parameter in model.parameters())}
        if not all(result.values()):
            raise AssertionError("frozen model/snapshot was modified")
        return result

    def status(self):
        return {"collected": self.collected, "trained": self.trainer.transitions_seen, "buffer_transitions": len(self.buffer),
                "updates": self.trainer.updates, "optimizer_steps": self.trainer.optimizer_steps,
                "validation_transitions": self.validation_transitions, "cursor": self.cursor, "best": self.best,
                "last_episode": self.last_episode, "split": self.split}

    def update(self, *, diagnostic=False):
        if len(self.buffer) < self.training_config["rollout_minimum"] and not diagnostic:
            raise ValueError("formal PPO requires at least 64 fresh navigation transitions")
        snapshot_version = self.trainer.updates
        report, targets = self.trainer.update(self.buffer)
        report.update(frozen_audit=self.freeze_audit(), old_policy_version=snapshot_version,
                      diagnostic=diagnostic, boundary_paths=list(self.buffer_boundaries))
        atomic_save(self.output / "updates" / f"targets_{self.trainer.updates:04d}.pt", targets)
        write_json(self.output / "updates" / f"update_{self.trainer.updates:04d}.json", report)
        self.buffer.clear()
        self.buffer_boundaries.clear()
        self.publish_snapshot()
        return report

    def save(self, *, cursor=None, label="latest"):
        if cursor is not None:
            self.cursor = cursor
        path = self.output / "checkpoints" / f"{label}.pt"
        atomic_save(path, {"schema": self.training_config["schema"], "provenance": self.provenance,
                           "trainer": self.trainer.payload(), "cursor": self.cursor, "collected": self.collected,
                           "validation_transitions": self.validation_transitions, "buffer_boundaries": self.buffer_boundaries,
                           "best": self.best, "episode_boundary": True, "last_episode": self.last_episode,
                           "checkpoint_history": self.checkpoint_history})
        return {"checkpoint": str(path), **self.status()}

    def restore(self, *, checkpoint):
        payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if payload["provenance"] != self.provenance or not payload["episode_boundary"]:
            raise ValueError("checkpoint provenance/boundary mismatch")
        self.trainer.restore(payload["trainer"])
        self.cursor, self.collected, self.best = payload["cursor"], payload["collected"], payload["best"]
        self.validation_transitions = payload["validation_transitions"]
        self.buffer_boundaries = payload["buffer_boundaries"]
        self.checkpoint_history = payload["checkpoint_history"]
        self.last_episode = payload["last_episode"]
        self.buffer = []
        for boundary in self.buffer_boundaries:
            self.buffer.extend(self.trainer.prepare_episode(load_episode_traces(boundary)))
        self.publish_snapshot()
        self.freeze_audit()
        return self.status()

    def validation_result(self, *, metrics):
        if self.buffer or metrics["split"] != "val_seen" or metrics["episodes"] != len(self.allowed["val_seen"]):
            raise ValueError("selection requires full predefined val_seen subset with empty on-policy buffer")
        if not all(np.isfinite(metrics[key]) for key in ("SR", "SPL")):
            raise ValueError("nonfinite validation metric")
        checkpoint = self.save(label=f"update_{self.trainer.updates:04d}")["checkpoint"]
        record = {"checkpoint": checkpoint, "transitions": self.trainer.transitions_seen, "update": self.trainer.updates, **metrics}
        self.checkpoint_history.append(record)
        if self.best is None or (metrics["SR"], metrics["SPL"]) > (self.best["SR"], self.best["SPL"]):
            self.best = record
            self.save(label="best_val_seen")
        write_json(self.output / "validation_history.json", self.checkpoint_history)
        return {"best": self.best, "record": record}
