from __future__ import annotations

import functools
import json
from pathlib import Path

import torch

from Experiment.SCOPE.Stage2.Task23 import learning as original_learning
from Experiment.SCOPE.Stage2.Task23 import runtime as original_runtime
from Experiment.SCOPE.Stage2.Task22.actor import HierarchicalDistribution, LatentAction
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task232.metrics import fixed_probe
from Experiment.SCOPE.Stage2.Task232.novelty import augment_transitions, coverage_novelty

ROOT = Path(__file__).resolve().parent
ORIGINAL_REWARD = original_learning.navigation_reward


def repair_reward(teacher, focus_count, *, terminated, truncated, focus_cost):
    return ORIGINAL_REWARD(teacher, focus_count, terminated=terminated, truncated=truncated) + (.01 - focus_cost) * focus_count


class RepairPolicy(original_runtime.PPOPolicy):
    def __init__(self, *, output, config, manifest):
        settings = json.loads(Path(config).read_text())
        reward = functools.partial(repair_reward, focus_cost=settings["focus_cost"])
        original_learning.navigation_reward = reward
        original_runtime.navigation_reward = reward
        self.stop_threshold = .5
        original_choose = HierarchicalDistribution.choose

        def choose(distribution, mode, generator=None):
            if mode == "deterministic":
                if bool(distribution.stop.probs >= self.stop_threshold):
                    return LatentAction(True)
                camera = int(distribution.camera.probs.argmax())
                return LatentAction(False, camera, distribution.position.loc[camera].detach().clone())
            return original_choose(distribution, mode, generator)

        HierarchicalDistribution.choose = choose
        super().__init__(output=output, config=config, manifest=manifest)
        self.coverage_beta = settings.get("coverage_beta", 0.)
        if "coverage_beta" in settings:
            prepare_episode = self.trainer.prepare_episode

            def prepare_with_novelty(traces):
                return augment_transitions(prepare_episode(traces), self.coverage_beta)

            self.trainer.prepare_episode = prepare_with_novelty
        self.provenance["task232_config_sha256"] = sha256(config)
        write_json(self.output / "fork_contract.json", {"config_sha256": sha256(config), "manifest_sha256": sha256(manifest),
                   "parent_checkpoint_sha256": settings["parent_checkpoint_sha256"], "branch": settings["task232_branch"],
                   "original_source_modified": False, "reward_coefficient": settings["focus_cost"],
                   "optimizer_rule": "retained AdamW state; explicit LR migration in prepared fork"})

    def update(self, **kwargs):
        for boundary in self.buffer_boundaries:
            if not Path(boundary).resolve().is_relative_to(self.output.resolve()):
                raise PermissionError("only fresh rollouts produced in this run may enter optimizer updates")
        if any(transition.trace["purpose"] != "train" or transition.trace["mode"] != "training-sampling" for transition in self.buffer):
            raise PermissionError("non-on-policy or evaluation data forbidden")
        return super().update(**kwargs)

    def finish(self, session, **kwargs):
        report = super().finish(session, **kwargs)
        components = {"progress": 0., "focus_cost": 0., "step_cost": 0., "terminal": 0.}
        novelty_enabled = "coverage_beta" in self.training_config
        if novelty_enabled:
            components["coverage_bonus"] = 0.
        episode_reward = 0.
        for path in report["trace_paths"]:
            trace = torch.load(path, map_location="cpu", weights_only=True)
            bonus = 0.
            if novelty_enabled:
                novelty = coverage_novelty(trace["decisions"], self.coverage_beta)
                bonus = novelty["reward"]
                if session["purpose"] != "train":
                    trace["navigation_reward"] += bonus
                else:
                    assert trace["task232_coverage_novelty"] == novelty
                trace["task232_coverage_novelty"] = novelty
            reward = trace["navigation_reward"]
            terminal = (2. if trace["teacher_for_reward_only"]["success"] else -.5) if trace["terminated"] else 0.
            focus_cost = -self.training_config["focus_cost"] * trace["focus_count"]
            trace["task232_reward_components"] = {"progress": reward - bonus - terminal - focus_cost + .01,
                                                  "focus_cost": focus_cost, "step_cost": -.01, "terminal": terminal}
            if novelty_enabled:
                trace["task232_reward_components"]["coverage_bonus"] = bonus
            episode_reward += reward
            trace["task232_branch"] = self.training_config["task232_branch"]
            for key, value in trace["task232_reward_components"].items():
                components[key] += value
            atomic_save(path, trace)
        report["reward_components"] = components
        report["episode_summary"]["reward_sum"] = episode_reward
        write_json(self.output / f"finish_{session['run_id']}.json", report)
        return report

    def save(self, *, cursor=None, label="latest", probe=False):
        result = super().save(cursor=cursor, label=label)
        if probe:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                fixed_probe(self.actor, self.output / "probes", ROOT / "frozen/PROBE_MANIFEST.json", ROOT / "frozen/fork_U.pt", label)
            result["probe"] = str(self.output / "probes" / f"{label}.json")
        return result

    def restore(self, *, checkpoint, stop_threshold=.5, original_evaluation=False):
        self.stop_threshold = float(stop_threshold)
        if not 0 < self.stop_threshold < 1:
            raise ValueError("invalid STOP threshold")
        if original_evaluation:
            if self.buffer:
                raise ValueError("cannot replace policy with a nonempty rollout buffer")
            payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
            for key in ("backbone_sha256", "adapter_sha256", "partition_sha256"):
                assert payload["provenance"][key] == self.provenance[key]
            for name in ("actor", "critic", "old_actor", "old_critic"):
                getattr(self.trainer, name).load_state_dict(payload["trainer"][name])
            self.trainer.updates = payload["trainer"]["updates"]
            self.trainer.transitions_seen = payload["trainer"]["transitions_seen"]
            self.publish_snapshot()
            self.freeze_audit()
            return self.status()
        return super().restore(checkpoint=checkpoint)
