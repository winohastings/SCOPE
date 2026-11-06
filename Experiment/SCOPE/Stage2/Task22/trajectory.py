from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

import torch

from .actor import ActorState, LatentAction

SCHEMA = "agm-task22-v1"


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def perception_loop(actor, perception, language, *, mode, generator, transition_id):
    decisions = []
    started = time.perf_counter()
    generator_before = generator.get_state().cpu().clone()
    while len(perception.selected) < 4:
        state = ActorState(perception.glance, language, perception.current, perception.coverage,
                           perception.history, len(perception.selected))
        with torch.no_grad():
            distribution = actor(state)
            action = distribution.choose(mode, generator)
            old_logprob = distribution.log_prob(action).detach().cpu().clone()
            entropies = {key: float(value) for key, value in distribution.entropies().items()}
        decision = {"index": len(decisions), "transition_id": transition_id, "state": state.packed(),
                    "action": action.packed(), "old_log_probability": old_logprob, "entropy": entropies,
                    "center_pixels": None if action.stop else action.center().cpu(),
                    "roi": None if action.stop else list(action.focus().roi), "focus_count_before": state.count,
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "old_distribution": {"stop_probability": distribution.stop.probs.detach().cpu(),
                                         "camera_probabilities": distribution.camera.probs.detach().cpu(),
                                         "mean": distribution.position.loc.detach().cpu(),
                                         "std": distribution.position.scale.detach().cpu()}}
        if action.stop:
            decision["focus_count_after"] = state.count
            decisions.append(decision)
            break
        perception.acquire(action.focus())
        decision["focus_count_after"] = len(perception.selected)
        decisions.append(decision)
    final = ActorState(perception.glance, language, perception.current, perception.coverage,
                       perception.history, len(perception.selected))
    return {"schema": SCHEMA, "transition_id": transition_id, "mode": mode,
            "ppo_eligible": mode == "training-sampling", "decisions": decisions,
            "termination": "budget" if len(perception.selected) == 4 else "policy_stop",
            "forced_stop_has_log_probability": False, "final_state": final.packed(),
            "focus_count": len(perception.selected), "rng_before": generator_before,
            "rng_after": generator.get_state().cpu().clone(), "perception_ms": (time.perf_counter() - started) * 1000,
            "old_sequence_log_probability": torch.stack([record["old_log_probability"] for record in decisions]).sum(),
            "feedback": None, "teacher_for_reward_only": None}


@dataclass
class ReplayDecision:
    observation: ActorState
    action: LatentAction
    old_log_probability: torch.Tensor


def replay_decisions(trace, device):
    if trace["schema"] != SCHEMA or not 1 <= len(trace["decisions"]) <= 4:
        raise ValueError("invalid perception trace")
    decisions = []
    count = 0
    for index, record in enumerate(trace["decisions"]):
        state = ActorState.unpack(record["state"], device)
        action = LatentAction.unpack(record["action"], device)
        if record["transition_id"] != trace["transition_id"] or record["index"] != index or state.count != count:
            raise ValueError("trace state/transition order mismatch")
        if action.stop and index != len(trace["decisions"]) - 1:
            raise ValueError("decisions after STOP")
        if not action.stop:
            if record["roi"] != list(action.focus().roi) or not torch.allclose(record["center_pixels"].to(device), action.center(), rtol=0, atol=1e-10):
                raise ValueError("latent/center/ROI mismatch")
            count += 1
        if record["focus_count_before"] != state.count or record["focus_count_after"] != count:
            raise ValueError("decision count mismatch")
        old = record["old_log_probability"].to(device)
        if old.requires_grad or old.ndim != 0 or not torch.isfinite(old):
            raise ValueError("invalid old log probability")
        decisions.append(ReplayDecision(state, action, old))
    if count != trace["focus_count"] or trace["final_state"]["count"] != count:
        raise ValueError("final Focus count mismatch")
    if (trace["termination"] == "budget") != (count == 4):
        raise ValueError("budget termination mismatch")
    if count < 4 and not decisions[-1].action.stop:
        raise ValueError("missing sampled STOP")
    if trace["forced_stop_has_log_probability"]:
        raise ValueError("forced termination cannot be a policy action")
    final = ActorState.unpack(trace["final_state"], device)
    if not torch.equal(final.glance, decisions[0].observation.glance) or not torch.equal(final.language, decisions[0].observation.language):
        raise ValueError("Glance/instruction changed during perception")
    old_sum = torch.stack([decision.old_log_probability for decision in decisions]).sum()
    if not torch.allclose(old_sum, trace["old_sequence_log_probability"].to(device), atol=1e-6, rtol=1e-6):
        raise ValueError("stored joint probability does not match decision sum")
    return decisions


def joint_log_probability(actor, decisions):
    values = [actor.evaluate(decision.observation, decision.action) for decision in decisions]
    return torch.stack([value[0] for value in values]).sum(), torch.stack([value[1]["joint_latent"] for value in values]).sum()


def navigation_transition(trace, *, old_value, reward, terminated, truncated, next_value, device):
    if not trace["ppo_eligible"] or trace["mode"] != "training-sampling" or trace["feedback"] is None:
        raise ValueError("PPO requires sampled actions and actual completed navigation feedback")
    if terminated and truncated:
        raise ValueError("terminal and time-limit truncation are distinct")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Experiment.SCOPE.Stage2.training import Transition
    decisions = replay_decisions(trace, device)
    return Transition(decisions, decisions[0].observation, torch.as_tensor(old_value, device=device).detach(),
                      float(reward), bool(terminated), bool(truncated), float(next_value), trace["episode_id"])


def load_episode_traces(boundary_path):
    boundary = torch.load(boundary_path, map_location="cpu", weights_only=True)
    if not boundary["episode_boundary"]:
        raise ValueError("only completed episode boundaries can be replayed")
    paths = boundary["report"]["trace_paths"]
    traces = []
    for index, path in enumerate(paths):
        trace = torch.load(path, map_location="cpu", weights_only=True)
        replay_decisions(trace, "cpu")
        if trace["feedback"] is None or trace["provenance"] != boundary["provenance"]:
            raise ValueError("incomplete or foreign episode trace")
        if trace["actor_checkpoint_sha256"] != boundary["actor_checkpoint_sha256"]:
            raise ValueError("Actor version changes inside episode")
        if trace["transition_id"].rsplit(":", 1)[-1] != str(index):
            raise ValueError("noncontiguous navigation transitions")
        if trace["episode_id"] != boundary["report"]["episode_id"]:
            raise ValueError("mixed episode IDs")
        if traces and (traces[-1]["terminated"] or traces[-1]["truncated"]):
            raise ValueError("transitions after episode termination")
        traces.append(trace)
    if not traces:
        raise ValueError("empty or failed episode cannot train")
    if not traces[-1]["terminated"] and (not traces[-1]["truncated"] or "bootstrap_state" not in traces[-1]):
        raise ValueError("episode truncation requires explicit bootstrap state")
    for index, trace in enumerate(traces):
        if trace["terminated"]:
            trace["next_state"] = None
        elif index + 1 < len(traces):
            trace["next_state"] = traces[index + 1]["decisions"][0]["state"]
        else:
            trace["next_state"] = trace["bootstrap_state"]
    return traces
