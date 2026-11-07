from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import random
import time

import numpy as np
import torch

from Experiment.SCOPE.Stage2.Task22.actor import ActorState
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save, replay_decisions
from .critic import NavigationCritic


def navigation_reward(teacher, focus_count, *, terminated, truncated):
    if teacher is None or type(focus_count) is not int or not 0 <= focus_count <= 4:
        raise ValueError("actual navigation reward measurements required")
    before, after = teacher["distance_before"], teacher["distance_after"]
    if not all(math.isfinite(value) and value >= 0 for value in (before, after)):
        raise ValueError("finite nonnegative geodesic distances required")
    if terminated and truncated:
        raise ValueError("terminal and truncation are exclusive")
    if bool(teacher["terminal"]) != bool(terminated):
        raise ValueError("teacher terminal disagrees with actual navigation boundary")
    if bool(teacher["success"]) != bool(after < 3):
        raise ValueError("success must use actual geodesic threshold")
    terminal = (2.0 if teacher["success"] else -0.5) if terminated else 0.0
    return .2 * float(np.clip((before - after) / 3, -1, 1)) - .01 - .01 * focus_count + terminal


@dataclass
class NavigationTransition:
    trace: dict
    old_value: float
    next_value: float
    reward: float
    episode_key: str


def gae(transitions, gamma=.99, gae_lambda=.95):
    if not transitions:
        raise ValueError("empty on-policy buffer")
    advantages = torch.zeros(len(transitions), dtype=torch.float32)
    carry = 0.0
    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        terminated, truncated = transition.trace["terminated"], transition.trace["truncated"]
        if terminated and truncated:
            raise ValueError("invalid episode boundary")
        bootstrap = 0.0 if terminated else transition.next_value
        continuing = (index + 1 < len(transitions) and not terminated and not truncated
                      and transition.episode_key == transitions[index + 1].episode_key)
        if not terminated and not truncated and not continuing:
            raise ValueError("buffer must end at an explicit episode boundary")
        delta = transition.reward + gamma * bootstrap - transition.old_value
        carry = delta + gamma * gae_lambda * carry * continuing
        advantages[index] = carry
    values = torch.tensor([transition.old_value for transition in transitions])
    return advantages.detach(), (advantages + values).detach()


def sequence_objective(actor, decisions, advantage, config):
    losses, ratios, logratios, entropies = [], [], [], []
    for decision in decisions:
        logprob, entropy = actor.evaluate(decision.observation, decision.action)
        logratio = logprob - decision.old_log_probability
        ratio = logratio.exp()
        clipped = ratio.clamp(1 - config["clip"], 1 + config["clip"])
        loss = -torch.minimum(ratio * advantage, clipped * advantage)
        regularizer = sum(config["entropy_weights"][key] * entropy[key] for key in config["entropy_weights"])
        losses.append(loss - regularizer)
        ratios.append(ratio)
        logratios.append(logratio)
        entropies.append(entropy)
    return torch.stack(losses).sum(), torch.stack(ratios), torch.stack(logratios), entropies


def parameter_change(before, model):
    return math.sqrt(sum(float((parameter.detach() - before[name]).square().sum()) for name, parameter in model.named_parameters()))


class PPOTrainer:
    def __init__(self, actor, config, device="cuda"):
        self.config = copy.deepcopy(config)
        self.device = torch.device(device)
        self.actor = actor.to(device).requires_grad_(True)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.device.type == "cuda" else []):
            torch.manual_seed(config["seed"])
            self.critic = NavigationCritic().to(device)
        self.old_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.old_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config["actor_learning_rate"], weight_decay=config["weight_decay"])
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=config["critic_learning_rate"], weight_decay=config["weight_decay"])
        if {id(parameter) for parameter in self.actor.parameters()} & {id(parameter) for parameter in self.critic.parameters()}:
            raise ValueError("Actor/Critic parameters must be independent")
        self.generator = torch.Generator().manual_seed(config["seed"])
        self.updates = 0
        self.transitions_seen = 0
        self.optimizer_steps = 0
        self.last_report = None

    def synchronize_old(self):
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())
        self.old_actor.eval().requires_grad_(False)
        self.old_critic.eval().requires_grad_(False)

    @torch.no_grad()
    def prepare_episode(self, traces):
        result = []
        for trace in traces:
            if not trace["ppo_eligible"] or trace["mode"] != "training-sampling" or trace["feedback"] is None:
                raise ValueError("on-policy sampling and actual feedback required")
            if trace["policy_version"] != self.updates or trace["purpose"] != "train":
                raise ValueError("stale or validation rollout cannot train")
            state = ActorState.unpack(trace["decisions"][0]["state"], self.device)
            value = float(self.old_critic(state))
            if abs(value - trace["old_value"]) > 1e-5:
                raise ValueError("old Critic snapshot/value mismatch")
            next_value = 0.0 if trace["terminated"] else float(self.old_critic(ActorState.unpack(trace["next_state"], self.device)))
            reward = navigation_reward(trace["teacher_for_reward_only"], trace["focus_count"],
                                       terminated=trace["terminated"], truncated=trace["truncated"])
            trace["navigation_reward"] = reward
            trace["old_next_value"] = next_value
            result.append(NavigationTransition(trace, value, next_value, reward, trace["transition_id"].rsplit(":", 1)[0]))
        return result

    def update(self, transitions):
        if any(transition.trace["policy_version"] != self.updates or transition.trace["purpose"] != "train" for transition in transitions):
            raise ValueError("stale/foreign buffer")
        advantages, returns = gae(transitions, self.config["gamma"], self.config["gae_lambda"])
        if self.config["normalize_advantage"]:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        decisions = [replay_decisions(transition.trace, self.device) for transition in transitions]
        old_values = torch.tensor([transition.old_value for transition in transitions], device=self.device)
        advantages, returns = advantages.to(self.device), returns.to(self.device)
        before_actor = {name: parameter.detach().clone() for name, parameter in self.actor.named_parameters()}
        before_critic = {name: parameter.detach().clone() for name, parameter in self.critic.named_parameters()}
        started = time.perf_counter()
        self.actor.train()
        self.critic.train()
        initial_error = 0.0
        with torch.no_grad():
            for sequence in decisions:
                for decision in sequence:
                    probability, _ = self.actor.evaluate(decision.observation, decision.action)
                    initial_error = max(initial_error, abs(float((probability - decision.old_log_probability).exp()) - 1))
        if initial_error > 1e-4:
            raise AssertionError(f"initial PPO ratio is not one: {initial_error}")
        statistics = []
        head_gradients = {key: 0.0 for key in ("stop", "camera", "position", "critic")}
        stop_early = False
        for epoch in range(self.config["epochs"]):
            order = torch.randperm(len(transitions), generator=self.generator).tolist()
            for start in range(0, len(order), self.config["minibatch_states"]):
                indices = order[start:start + self.config["minibatch_states"]]
                actor_losses, ratio_values, logratio_values, entropy_values = [], [], [], []
                for index in indices:
                    loss, ratio, logratio, entropy = sequence_objective(self.actor, decisions[index], advantages[index], self.config)
                    actor_losses.append(loss)
                    ratio_values.append(ratio)
                    logratio_values.append(logratio)
                    entropy_values.extend(entropy)
                actor_loss = torch.stack(actor_losses).mean()
                predictions = torch.stack([self.critic(decisions[index][0].observation) for index in indices])
                critic_mse = (predictions - returns[indices]).square().mean()
                critic_loss = self.config["critic_loss_weight"] * critic_mse
                ratios, logratios = torch.cat(ratio_values), torch.cat(logratio_values)
                approximate_kl = (ratios - 1 - logratios).mean()
                if not torch.isfinite(actor_loss + critic_loss + approximate_kl):
                    raise FloatingPointError("nonfinite PPO loss or KL")
                if statistics and float(approximate_kl.detach()) > self.config["kl_stop_threshold"]:
                    stop_early = True
                    break
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                for model in (self.actor, self.critic):
                    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                        raise FloatingPointError("nonfinite gradient")
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config["gradient_clip"])
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config["gradient_clip"])
                if not torch.isfinite(actor_norm + critic_norm):
                    raise FloatingPointError("nonfinite global gradient norm")
                heads = {"stop": self.actor.stop_head[-1].weight, "camera": self.actor.camera_head[-1].weight,
                         "position": self.actor.position_head[-1].weight, "critic": self.critic.value[-1].weight}
                for key, parameter in heads.items():
                    head_gradients[key] = max(head_gradients[key], float(parameter.grad.norm()) if parameter.grad is not None else 0.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                self.optimizer_steps += 1
                statistics.append({"epoch": epoch, "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                                   "critic_mse": float(critic_mse.detach()), "ratio_mean": float(ratios.mean().detach()),
                                   "approximate_kl": float(approximate_kl.detach()),
                                   "clip_fraction": float(((ratios - 1).abs() > self.config["clip"]).float().mean()),
                                   "actor_grad_norm": float(actor_norm), "critic_grad_norm": float(critic_norm),
                                   "entropy": {key: float(torch.stack([entropy[key] for entropy in entropy_values]).mean().detach())
                                               for key in ("stop", "camera", "position_latent")}})
            if stop_early:
                break
        actor_change, critic_change = parameter_change(before_actor, self.actor), parameter_change(before_critic, self.critic)
        if not actor_change > 0 or not critic_change > 0:
            raise AssertionError("optimizer.step must update both Actor and Critic")
        if not all(torch.isfinite(parameter).all() for model in (self.actor, self.critic) for parameter in model.parameters()):
            raise FloatingPointError("nonfinite updated weights")
        with torch.no_grad():
            predictions = torch.stack([self.critic(sequence[0].observation) for sequence in decisions])
            residual = returns - predictions
            explained = float(1 - residual.var(unbiased=False) / returns.var(unbiased=False).clamp_min(1e-8))
        self.updates += 1
        self.transitions_seen += len(transitions)
        self.last_report = {"update": self.updates, "transitions": len(transitions), "total_transitions": self.transitions_seen,
                            "optimizer_steps": self.optimizer_steps, "initial_ratio_max_error": initial_error,
                            "actor_parameter_l2_change": actor_change, "critic_parameter_l2_change": critic_change,
                            "head_gradient_norms": head_gradients, "minibatches": statistics, "kl_early_stop": stop_early,
                            "explained_variance": explained, "return_mean": float(returns.mean()),
                            "old_value_mean": float(old_values.mean()), "value_mean": float(predictions.mean()),
                            "seconds": time.perf_counter() - started}
        self.synchronize_old()
        return self.last_report, {"advantages": advantages.cpu(), "returns": returns.cpu(), "old_values": old_values.cpu()}

    def payload(self):
        return {"config": self.config, "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "old_actor": self.old_actor.state_dict(), "old_critic": self.old_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(), "critic_optimizer": self.critic_optimizer.state_dict(),
                "updates": self.updates, "transitions_seen": self.transitions_seen, "optimizer_steps": self.optimizer_steps,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "minibatch_rng": self.generator.get_state()}

    def restore(self, payload):
        if payload["config"] != self.config:
            raise ValueError("training config changed on restore")
        for name in ("actor", "critic", "old_actor", "old_critic"):
            getattr(self, name).load_state_dict(payload[name])
        for name in ("actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.updates, self.transitions_seen, self.optimizer_steps = payload["updates"], payload["transitions_seen"], payload["optimizer_steps"]
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        self.generator.set_state(payload["minibatch_rng"].cpu())
