from __future__ import annotations

import torch
from torch import nn

from Experiment.SCOPE.Stage2.Task22.actor import ReplayStableSpatialLayer


class NavigationCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_projection = nn.Sequential(nn.Linear(2560, 128), nn.LayerNorm(128), nn.GELU())
        self.language_projection = nn.Sequential(nn.Linear(2560, 128), nn.LayerNorm(128), nn.GELU())
        self.camera_embedding = nn.Embedding(4, 128)
        self.position_embedding = nn.Embedding(16, 128)
        layer = ReplayStableSpatialLayer(128, 4, 256, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pooling = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.value = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, state):
        state.validate()
        if state.count != 0 or state.coverage.count_nonzero() or state.history.count_nonzero():
            raise ValueError("Critic evaluates pre-Focus navigation state only")
        if not torch.equal(state.glance, state.fused):
            raise ValueError("Critic cannot observe post-Focus information")
        visual = self.visual_projection(state.glance.float())
        visual = visual + self.camera_embedding.weight[:, None] + self.position_embedding.weight[None]
        spatial = self.spatial(visual.reshape(1, 64, 128))
        language = self.language_projection(state.language.float()).reshape(1, 1, 128)
        pooled, _ = self.pooling(language, spatial, spatial, need_weights=False)
        return self.value(self.norm(language + pooled)).reshape(())
