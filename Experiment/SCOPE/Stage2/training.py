from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random

import numpy as np
import torch

from policy import ActorConfig, IndependentCritic, PolicyObservation, ROIActor


@dataclass
class Decision:
    observation: PolicyObservation
    action: int
    mask: torch.Tensor
    old_log_probability: torch.Tensor


@dataclass
class Transition:
    decisions: list[Decision]
    value_observation: PolicyObservation
    old_value: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    next_value: float
    episode_id: str

    @property
    def old_log_probability(self):
        return torch.stack([decision.old_log_probability for decision in self.decisions]).sum().detach()


@dataclass(frozen=True)
class FixedTargets:
    returns: torch.Tensor
    advantages: torch.Tensor
    old_log_probabilities: torch.Tensor


def gae_targets(rollout, gamma=0.99, gae_lambda=0.95):
    if not rollout or any(not transition.decisions for transition in rollout):
        raise ValueError("rollout must contain completed perception decisions")
    device = rollout[0].old_value.device
    advantages = torch.zeros(len(rollout), device=device)
    accumulator = 0.0
    for index in reversed(range(len(rollout))):
        transition = rollout[index]
        bootstrap = 0.0 if transition.terminated else transition.next_value
        delta = transition.reward + gamma * bootstrap - float(transition.old_value)
        same_episode = index + 1 < len(rollout) and rollout[index + 1].episode_id == transition.episode_id
        trace_continues = same_episode and not transition.terminated and not transition.truncated
        accumulator = delta + gamma * gae_lambda * accumulator * trace_continues
        advantages[index] = accumulator
    values = torch.stack([transition.old_value.detach() for transition in rollout])
    return FixedTargets((advantages + values).detach().clone(), advantages.detach().clone(),
                        torch.stack([transition.old_log_probability for transition in rollout]).detach().clone())


def joint_log_probability(actor, decisions):
    probabilities, entropies = [], []
    for decision in decisions:
        distribution = actor(decision.observation, decision.mask)
        if not bool(decision.mask[decision.action]):
            raise ValueError("rollout contains invalid action")
        probabilities.append(distribution.log_prob(torch.tensor(decision.action, device=decision.mask.device)))
        entropies.append(distribution.entropy())
    return torch.stack(probabilities).sum(), torch.stack(entropies).sum()


class PPOTrainer:
    def __init__(self, actor, critic, learning_rate=3e-4):
        self.actor, self.critic = actor, critic
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        if actor_ids & {id(parameter) for parameter in critic.parameters()}:
            raise ValueError("Actor and Critic must have independent parameters")
        self.actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=learning_rate)
        self.updates = 0
        self.transitions_seen = 0

    def update(self, rollout, *, epochs=4, minibatch_size=16, targets=None):
        targets = targets or gae_targets(rollout)
        if any(value.requires_grad for value in (targets.returns, targets.advantages, targets.old_log_probabilities)):
            raise ValueError("rollout targets must be fixed and detached")
        advantages = targets.advantages.clone()
        if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        statistics = []
        for epoch in range(epochs):
            ordering = torch.randperm(len(rollout)).tolist()
            for begin in range(0, len(ordering), minibatch_size):
                indices = ordering[begin:begin + minibatch_size]
                joint, entropy = zip(*(joint_log_probability(self.actor, rollout[index].decisions) for index in indices))
                joint, entropy = torch.stack(joint), torch.stack(entropy)
                ratio = (joint - targets.old_log_probabilities[indices]).exp()
                surrogate = torch.minimum(ratio * advantages[indices], ratio.clamp(0.8, 1.2) * advantages[indices])
                actor_loss = -surrogate.mean() - 0.01 * entropy.mean()
                predictions = torch.stack([self.critic(rollout[index].value_observation) for index in indices])
                critic_loss = 0.5 * (predictions - targets.returns[indices]).square().mean()
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                actor_gru = self.actor.encoder.state_gru.weight_ih.grad
                critic_gru = self.critic.encoder.state_gru.weight_ih.grad
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                statistics.append({"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                                   "ratio_mean": float(ratio.mean().detach()), "actor_grad_norm": float(actor_norm),
                                   "critic_grad_norm": float(critic_norm), "entropy": float(entropy.mean().detach())})
                statistics[-1].update(actor_state_gru_grad_norm=0.0 if actor_gru is None else float(actor_gru.norm()),
                                      critic_state_gru_grad_norm=0.0 if critic_gru is None else float(critic_gru.norm()))
        self.updates += 1
        self.transitions_seen += len(rollout)
        return statistics

    def save(self, path, *, provenance, cursor, episode_boundary=True):
        if not episode_boundary:
            raise ValueError("Habitat cannot serialize mid-episode: checkpoint only at episode boundaries")
        payload = {"schema": 1, "config": asdict(self.actor.config), "actor": self.actor.state_dict(),
                   "critic": self.critic.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(), "updates": self.updates,
                   "transitions_seen": self.transitions_seen, "python_rng": random.getstate(),
                   "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   "provenance": provenance, "cursor": cursor, "episode_boundary": True}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def restore(self, path, *, provenance):
        payload = torch.load(path, map_location=next(self.actor.parameters()).device, weights_only=False)
        if payload["provenance"] != provenance or payload["config"] != asdict(self.actor.config):
            raise ValueError("checkpoint provenance/config mismatch")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.updates, self.transitions_seen = payload["updates"], payload["transitions_seen"]
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if payload["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
        return payload["cursor"]


def build_trainer(regime="C", device="cpu", seed=101):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = ActorConfig.regime(regime)
    return PPOTrainer(ROIActor(config).to(device), IndependentCritic(config).to(device))
