from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
import torch
from torch import nn

CAMERAS = ("front", "left", "right", "rear")
VERSION = "task21-area-overlap-v1"


@dataclass(frozen=True)
class Focus:
    camera: int
    x: float = 0.5
    y: float = 0.5

    def __post_init__(self):
        if type(self.camera) is not int or not 0 <= self.camera < 4:
            raise ValueError("camera must be an integer in [0, 4)")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (self.x, self.y)):
            raise ValueError("normalized center coordinates must be finite and in [0, 1]")

    @property
    def roi(self):
        left = min(320, max(0, math.floor(self.x * 512 - 96 + 0.5)))
        top = min(320, max(0, math.floor(self.y * 512 - 96 + 0.5)))
        return left, top, left + 192, top + 192

    def crop(self, images):
        image = np.asarray(images[CAMERAS[self.camera]])
        if image.shape != (512, 512, 3) or image.dtype != np.uint8:
            raise ValueError("Focus requires uint8 RGB 512x512 camera images")
        return Image.fromarray(image).crop(self.roi).resize((256, 256), Image.Resampling.BICUBIC)


def align_area(tokens, source_roi, visible_roi):
    if tokens.ndim != 2 or not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError("finite floating point [spatial_tokens, channels] required")
    side = math.isqrt(tokens.shape[0])
    if side * side != tokens.shape[0] or side == 0:
        raise ValueError("source grid must be square")
    for bounds in (source_roi, visible_roi):
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("invalid ROI")
        if not (0 <= bounds[0] < bounds[2] <= 512 and 0 <= bounds[1] < bounds[3] <= 512):
            raise ValueError("ROI outside camera")
    if not (source_roi[0] <= visible_roi[0] and source_roi[1] <= visible_roi[1]
            and source_roi[2] >= visible_roi[2] and source_roi[3] >= visible_roi[3]):
        raise ValueError("visible ROI must be contained in source ROI")
    axes = []
    for axis in (0, 1):
        source_edges = torch.linspace(source_roi[axis], source_roi[axis + 2], side + 1, device=tokens.device)
        target_edges = torch.arange(5, device=tokens.device, dtype=torch.float32) * 128
        lower = torch.maximum(target_edges[:-1, None], source_edges[None, :-1]).clamp_min(visible_roi[axis])
        upper = torch.minimum(target_edges[1:, None], source_edges[None, 1:]).clamp_max(visible_roi[axis + 2])
        axes.append((upper - lower).clamp_min(0))
    weights = torch.einsum("ys,xt->yxst", axes[1], axes[0]).reshape(16, side * side)
    area = weights.sum(-1, keepdim=True)
    aligned = weights @ tokens.float() / area.clamp_min(1)
    return aligned, area / (128 * 128)


class SpatialFusion(nn.Module):
    def __init__(self, initial_gate=0.02):
        super().__init__()
        if not 0 < initial_gate < 1:
            raise ValueError("gate must be in (0, 1)")
        self.gate_logit = nn.Parameter(torch.tensor(math.log(initial_gate / (1 - initial_gate))))

    def forward(self, glance, updates=()):
        if tuple(glance.shape) != (4, 16, 2560):
            raise ValueError(f"unexpected current grid {tuple(glance.shape)}")
        if len(updates) > 4:
            raise ValueError("at most four Focus operations")
        if not updates:
            return glance
        if not glance.is_floating_point() or not torch.isfinite(glance).all():
            raise ValueError("invalid Glance features")
        current = glance
        for focus, tokens in updates:
            if tuple(tokens.shape) != (64, 2560) or tokens.device != glance.device:
                raise ValueError("Focus must encode to 64x2560 on Glance device")
            aligned, coverage = align_area(tokens, focus.roi, focus.roi)
            original = current[focus.camera]
            changed = original.float() + self.gate_logit.sigmoid() * coverage * (aligned - original.float())
            replacement = torch.where(coverage > 0, changed.to(original.dtype), original)
            current = torch.stack([replacement if camera == focus.camera else current[camera] for camera in range(4)])
        return current

    def freeze(self):
        return self.requires_grad_(False).eval()


def calibration_target(glance, teacher, focus):
    if tuple(teacher.shape) != (256, 2560):
        raise ValueError("512px teacher must have 16x16 postmerge grid")
    aligned, coverage = align_area(teacher, (0, 0, 512, 512), focus.roi)
    target = glance.float().clone()
    target[focus.camera] += coverage * (aligned - target[focus.camera])
    mask = torch.zeros((4, 16, 1), device=glance.device)
    mask[focus.camera] = coverage
    return target, mask
