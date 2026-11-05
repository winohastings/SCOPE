from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributions import Categorical

from features import Geometry, candidate_summaries

PERCEPTION_STOP = 16
HIDDEN = 64
FEEDBACK_SIZE = 18


@dataclass(frozen=True)
class ActorConfig:
    use_language: bool = True
    use_focus_memory: bool = True
    use_navigation_state: bool = True
    use_action_feedback: bool = True
    enable_perception_stop: bool = True

    @classmethod
    def regime(cls, name):
        if name == "A":
            return cls(use_navigation_state=False, use_action_feedback=False, enable_perception_stop=False)
        if name == "B":
            return cls(enable_perception_stop=False)
        if name == "C":
            return cls()
        raise ValueError("regime must be A/B/C")


@dataclass
class PolicyObservation:
    glance: torch.Tensor
    language: torch.Tensor
    focus: tuple[torch.Tensor, ...]
    selected: tuple[int, ...]
    navigation_previous: torch.Tensor
    navigation_visual: torch.Tensor
    feedback: torch.Tensor
    has_feedback: bool

    def snapshot(self):
        return PolicyObservation(self.glance.detach().clone(), self.language.detach().clone(),
                                 tuple(value.detach().clone() for value in self.focus), self.selected,
                                 self.navigation_previous.detach().clone(), self.navigation_visual.detach().clone(),
                                 self.feedback.detach().clone(), self.has_feedback)


def project():
    return nn.Sequential(nn.Linear(2560, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = project()
        self.language = project()
        self.focus_projection = project()
        self.focus_gru = nn.GRUCell(HIDDEN + 8, HIDDEN)
        self.feedback_projection = nn.Sequential(nn.Linear(FEEDBACK_SIZE, HIDDEN), nn.Tanh())
        self.state_gru = nn.GRUCell(HIDDEN * 2, HIDDEN)

    def navigation_state(self, observation, config):
        previous = observation.navigation_previous.float()
        if not config.use_navigation_state:
            return torch.zeros_like(previous)
        if not observation.has_feedback:
            return previous
        visual = self.visual(observation.navigation_visual.float())
        feedback = (self.feedback_projection(observation.feedback.float()) if config.use_action_feedback
                    else torch.zeros_like(visual))
        return self.state_gru(torch.cat((visual, feedback)), previous)

    def forward(self, observation, config):
        global_visual = self.visual(observation.glance.float().mean((0, 1)))
        language = self.language(observation.language.float()) if config.use_language else torch.zeros_like(global_visual)
        focus_state = torch.zeros_like(global_visual)
        if config.use_focus_memory:
            for candidate, feature in zip(observation.selected, observation.focus, strict=True):
                geometry = global_visual.new_tensor(Geometry.focus(candidate).vector())
                focus_state = self.focus_gru(torch.cat((self.focus_projection(feature.float().mean(0)), geometry)), focus_state)
        return torch.cat((global_visual, language, focus_state, self.navigation_state(observation, config)))


class ROIActor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ActorConfig()
        self.encoder = StateEncoder()
        self.geometry = nn.Linear(8, HIDDEN)
        self.query = nn.Sequential(nn.Linear(HIDDEN * 4, HIDDEN), nn.Tanh())
        self.key = nn.Linear(HIDDEN, HIDDEN)
        self.stop = nn.Linear(HIDDEN * 4, 1)

    def action_mask(self, selected, device):
        mask = torch.ones(17, dtype=torch.bool, device=device)
        if len(selected) > 2 or len(set(selected)) != len(selected) or any(not 0 <= item < 16 for item in selected):
            raise ValueError("invalid selected ROI sequence")
        mask[list(selected)] = False
        mask[16] = self.config.enable_perception_stop
        if len(selected) == 2:
            mask[:16] = False
            mask[16] = True
        return mask

    def forward(self, observation, mask=None):
        if len(observation.focus) != len(observation.selected):
            raise ValueError("only acquired Focus may enter the Actor")
        context = self.encoder(observation, self.config)
        local = self.encoder.visual(candidate_summaries(observation.glance).float())
        geometries = local.new_tensor([Geometry.focus(candidate).vector() for candidate in range(16)])
        keys = self.key(local + self.geometry(geometries))
        logits = keys @ self.query(context) / HIDDEN ** 0.5
        logits = torch.cat((logits, self.stop(context)))
        expected = self.action_mask(observation.selected, logits.device)
        if mask is not None and not torch.equal(mask, expected):
            raise ValueError("stored valid-action mask differs from rollout state")
        return Categorical(logits=logits.masked_fill(~expected, -torch.inf))


class IndependentCritic(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ActorConfig()
        self.encoder = StateEncoder()
        self.value = nn.Sequential(nn.Linear(HIDDEN * 4, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))

    def forward(self, observation):
        if observation.selected or observation.focus:
            raise ValueError("navigation value is evaluated before perception actions")
        return self.value(self.encoder(observation, self.config)).squeeze(-1)


class RecurrentSession:
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.actor_previous = torch.zeros(HIDDEN, device=self.device)
        self.critic_previous = torch.zeros(HIDDEN, device=self.device)
        self.actor_state = torch.zeros(HIDDEN, device=self.device)
        self.critic_state = torch.zeros(HIDDEN, device=self.device)
        self.visual = torch.zeros(2560, device=self.device)
        self.feedback = torch.zeros(FEEDBACK_SIZE, device=self.device)
        self.has_feedback = False
        self.completed_steps = 0

    def observation(self, bank, language, *, critic=False):
        return PolicyObservation(bank.glance(), language,
                                 tuple(bank.read_focus(candidate).tokens for candidate in bank.selected),
                                 tuple(bank.selected), self.critic_previous if critic else self.actor_previous,
                                 self.visual, self.feedback, self.has_feedback)

    @torch.no_grad()
    def complete(self, actor, critic, initial_actor, initial_critic, visual_summary, actual_feedback):
        if tuple(actual_feedback.shape) != (FEEDBACK_SIZE,) or not torch.isfinite(actual_feedback).all():
            raise ValueError("missing or invalid actual action feedback")
        self.actor_previous = actor.encoder.navigation_state(initial_actor, actor.config).detach()
        self.critic_previous = critic.encoder.navigation_state(initial_critic, critic.config).detach()
        self.visual = visual_summary.detach().clone()
        self.feedback = actual_feedback.detach().clone()
        self.has_feedback = True
        actor_next = replace(initial_actor, navigation_previous=self.actor_previous, navigation_visual=self.visual,
                             feedback=self.feedback, has_feedback=True)
        critic_next = replace(initial_critic, navigation_previous=self.critic_previous, navigation_visual=self.visual,
                              feedback=self.feedback, has_feedback=True)
        self.actor_state = actor.encoder.navigation_state(actor_next, actor.config).detach()
        self.critic_state = critic.encoder.navigation_state(critic_next, critic.config).detach()
        self.completed_steps += 1
