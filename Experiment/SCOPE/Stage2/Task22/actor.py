from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

from Experiment.SCOPE.Stage2.Task21.fusion import Focus


@dataclass(frozen=True)
class ActorState:
    glance: torch.Tensor
    language: torch.Tensor
    fused: torch.Tensor
    coverage: torch.Tensor
    history: torch.Tensor
    count: int

    def validate(self):
        expected = ((self.glance, (4, 16, 2560)), (self.fused, (4, 16, 2560)),
                    (self.language, (2560,)), (self.coverage, (4, 16, 1)), (self.history, (4, 7)))
        if type(self.count) is not int or not 0 <= self.count <= 4:
            raise ValueError("invalid Focus count")
        for tensor, shape in expected:
            if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all() or tensor.requires_grad:
                raise ValueError("Actor observations must be finite, detached and correctly shaped")
            if tensor.device != self.glance.device:
                raise ValueError("state device mismatch")
        if not ((self.coverage >= 0) & (self.coverage <= 1)).all():
            raise ValueError("coverage outside [0,1]")
        if not (self.history[:self.count, 6] == 1).all() or self.history[self.count:].count_nonzero():
            raise ValueError("invalid history mask")

    def packed(self):
        return {name: getattr(self, name).detach().cpu().clone()
                for name in ("glance", "language", "fused", "coverage", "history")} | {"count": self.count}

    @classmethod
    def unpack(cls, record, device):
        state = cls(**{name: value.to(device) if isinstance(value, torch.Tensor) else value
                       for name, value in record.items()})
        state.validate()
        return state


@dataclass(frozen=True)
class LatentAction:
    stop: bool
    camera: int | None = None
    latent: torch.Tensor | None = None

    def validate(self):
        if type(self.stop) is not bool:
            raise ValueError("stop must be boolean")
        if self.stop:
            if self.camera is not None or self.latent is not None:
                raise ValueError("STOP has no camera/position action")
        elif (type(self.camera) is not int or not 0 <= self.camera < 4 or self.latent is None
              or tuple(self.latent.shape) != (2,) or not torch.isfinite(self.latent).all()
              or self.latent.requires_grad):
            raise ValueError("non-STOP requires camera and detached finite latent u")

    def center(self):
        self.validate()
        if self.stop:
            return None
        return 96.0 + 320.0 * torch.sigmoid(self.latent.double())

    def focus(self):
        center = self.center()
        if center is None:
            raise ValueError("STOP does not crop")
        return Focus(self.camera, float(center[0]) / 512, float(center[1]) / 512)

    def packed(self):
        return {"stop": self.stop, "camera": self.camera,
                "latent": None if self.latent is None else self.latent.detach().cpu().clone()}

    @classmethod
    def unpack(cls, record, device):
        action = cls(record["stop"], record["camera"],
                     None if record["latent"] is None else record["latent"].to(device))
        action.validate()
        return action


class HierarchicalDistribution:
    def __init__(self, stop_logits, camera_logits, mean, std):
        self.stop = Bernoulli(logits=stop_logits)
        self.camera = Categorical(logits=camera_logits)
        self.position = Normal(mean, std)

    def choose(self, mode, generator=None):
        if mode not in ("deterministic", "training-sampling"):
            raise ValueError("explicit deterministic or training-sampling mode required")
        if mode == "deterministic":
            stop = bool(self.stop.probs >= 0.5)
            camera = int(self.camera.probs.argmax())
            latent = self.position.loc[camera].detach().clone()
        else:
            stop = bool(torch.rand((), device=self.stop.probs.device, generator=generator) < self.stop.probs)
            if stop:
                return LatentAction(True)
            camera = int(torch.multinomial(self.camera.probs, 1, generator=generator))
            noise = torch.randn(2, device=self.position.loc.device, generator=generator)
            latent = (self.position.loc[camera] + self.position.scale[camera] * noise).detach()
        return LatentAction(True) if stop else LatentAction(False, camera, latent)

    def log_prob(self, action):
        action.validate()
        result = self.stop.log_prob(self.stop.logits.new_tensor(float(action.stop)))
        if not action.stop:
            result = result + self.camera.log_prob(self.camera.logits.new_tensor(action.camera, dtype=torch.long))
            result = result + Normal(self.position.loc[action.camera], self.position.scale[action.camera]).log_prob(action.latent).sum()
        return result

    def entropies(self):
        position = (self.camera.probs * self.position.entropy().sum(-1)).sum()
        return {"stop": self.stop.entropy(), "camera": self.camera.entropy(), "position_latent": position,
                "joint_latent": self.stop.entropy() + (1 - self.stop.probs) * (self.camera.entropy() + position)}


class ReplayStableSpatialLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        normalized = self.norm1(src)
        attended = self.self_attn(normalized.clone(), normalized, normalized, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask, need_weights=False,
                                 is_causal=is_causal)[0]
        current = src + self.dropout1(attended)
        return current + self._ff_block(self.norm2(current))


class SpatialQueryActor(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 192
        self.visual_projection = nn.Sequential(nn.Linear(2560, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.language_projection = nn.Sequential(nn.Linear(2560, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.fused_projection = nn.Linear(hidden, hidden)
        self.camera_embedding = nn.Embedding(4, hidden)
        self.position_projection = nn.Linear(16, hidden)
        self.coverage_projection = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.history_projection = nn.Sequential(nn.Linear(29, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = ReplayStableSpatialLayer(hidden, 4, 768, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.cross_attention = nn.MultiheadAttention(hidden, 4, dropout=0.0, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden)
        self.stop_head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))
        self.camera_head = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 4))
        self.position_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 128),
                                           nn.GELU(), nn.Linear(128, 4))
        self.log_std = nn.Parameter(torch.full((4, 2), math.log(0.8)))
        nn.init.zeros_(self.stop_head[-1].weight)
        nn.init.constant_(self.stop_head[-1].bias, math.log(0.25 / 0.75))
        nn.init.zeros_(self.position_head[-1].weight)
        nn.init.zeros_(self.position_head[-1].bias)
        rows, columns = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        coordinates = torch.stack(((columns + .5) / 4, (rows + .5) / 4), dim=-1).reshape(16, 2)
        angles = coordinates[:, :, None] * (2 ** torch.arange(4)) * (2 * math.pi)
        self.register_buffer("spatial_positions", torch.cat((angles.sin(), angles.cos()), -1).reshape(16, 16))

    def forward(self, state):
        state.validate()
        if state.count == 4:
            raise ValueError("budget termination has no trainable Actor action")
        visual = self.visual_projection(state.glance.float())
        visual = visual + self.fused_projection(self.visual_projection(state.fused.float()))
        visual = visual + self.camera_embedding.weight[:, None] + self.position_projection(self.spatial_positions)[None]
        visual = visual + self.coverage_projection(state.coverage.float())
        encoded = self.spatial(visual.reshape(1, 64, 192))
        history = torch.cat((state.history.flatten(), state.history.new_tensor([state.count / 4])))
        query = self.language_projection(state.language.float()) + self.history_projection(history)
        attended, _ = self.cross_attention(query.reshape(1, 1, 192), encoded, encoded, need_weights=False)
        context = self.query_norm(query + attended.reshape(192))
        camera_features = encoded.reshape(4, 16, 192).mean(1)
        position = self.position_head(torch.cat((context.expand(4, -1), camera_features), -1))
        std = (self.log_std + position[:, 2:]).clamp(math.log(.1), math.log(1.5)).exp()
        return HierarchicalDistribution(self.stop_head(context).squeeze(-1), self.camera_head(context), position[:, :2], std)

    def evaluate(self, state, action):
        distribution = self(state)
        return distribution.log_prob(action), distribution.entropies()
