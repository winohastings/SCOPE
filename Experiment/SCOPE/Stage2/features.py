from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

import torch

CAMERAS = ("front", "left", "right", "rear")
CAMERA_YAWS = (0.0, -math.pi / 2, math.pi / 2, math.pi)
ENCODING_VERSION = "task1-native-bicubic-v1"


@dataclass(frozen=True)
class Geometry:
    camera: int
    roi: tuple[int, int, int, int]
    resolution: int
    hfov: float = math.pi / 2
    height: float = 1.25

    @classmethod
    def glance(cls, camera):
        return cls(camera, (0, 0, 512, 512), 128)

    @classmethod
    def focus(cls, candidate):
        if not 0 <= candidate < 16:
            raise ValueError("ROI must be in [0,16)")
        camera, region = divmod(candidate, 4)
        left, top = region % 2 * 256, region // 2 * 256
        return cls(camera, (left, top, left + 256, top + 256), 256)

    def vector(self):
        left, top, right, bottom = self.roi
        return [math.sin(CAMERA_YAWS[self.camera]), math.cos(CAMERA_YAWS[self.camera]),
                left / 512, top / 512, right / 512, bottom / 512, self.hfov, self.height]

    def rays(self):
        count = self.resolution // 32
        left, top, right, bottom = self.roi
        horizontal = torch.linspace(left + (right - left) / count / 2,
                                    right - (right - left) / count / 2, count)
        vertical = torch.linspace(top + (bottom - top) / count / 2,
                                  bottom - (bottom - top) / count / 2, count)
        rows, columns = torch.meshgrid(vertical, horizontal, indexing="ij")
        focal = 256 / math.tan(self.hfov / 2)
        local_right, local_down = (columns - 256) / focal, (rows - 256) / focal
        yaw = CAMERA_YAWS[self.camera]
        rays = torch.stack((math.cos(yaw) - local_right * math.sin(yaw),
                            math.sin(yaw) + local_right * math.cos(yaw), local_down), dim=-1)
        return torch.nn.functional.normalize(rays.reshape(-1, 3), dim=-1)


@dataclass(frozen=True)
class CurrentFeature:
    observation_key: str
    encoder_hash: str
    geometry: Geometry
    grid: tuple[int, int, int]
    tokens: torch.Tensor
    preprocessing: str = ENCODING_VERSION

    def __post_init__(self):
        count = 16 if self.geometry.resolution == 128 else 64
        expected_grid = (1, 8, 8) if count == 16 else (1, 16, 16)
        if self.grid != expected_grid or tuple(self.tokens.shape) != (count, 2560):
            raise ValueError("native feature/grid mismatch")
        if self.tokens.requires_grad or not torch.isfinite(self.tokens).all():
            raise ValueError("current visual features must be finite and frozen")


class CurrentFeatureBank:
    def __init__(self, observation_key: str, encoder_hash: str, encode: Callable):
        self.observation_key = observation_key
        self.encoder_hash = encoder_hash
        self._encode = encode
        self._entries: dict[Geometry, CurrentFeature] = {}
        self.selected: list[int] = []
        self.encoder_images = 0

    def acquire(self, images, geometries):
        missing = [(image, geometry) for image, geometry in zip(images, geometries, strict=True)
                   if geometry not in self._entries]
        if missing:
            outputs = self._encode([image for image, geometry in missing])
            for (image, geometry), (tokens, grid) in zip(missing, outputs, strict=True):
                feature = CurrentFeature(self.observation_key, self.encoder_hash, geometry,
                                         tuple(int(value) for value in grid), tokens.detach())
                self._entries[geometry] = feature
                self.encoder_images += 1
        return [self._entries[geometry] for geometry in geometries]

    def glance(self):
        return torch.stack([self._entries[Geometry.glance(camera)].tokens for camera in range(4)])

    def select(self, candidate, image):
        if candidate in self.selected:
            raise ValueError("repeated ROI")
        feature = self.acquire([image], [Geometry.focus(candidate)])[0]
        self.selected.append(candidate)
        return feature

    def read_focus(self, candidate):
        if candidate not in self.selected:
            raise PermissionError("unselected high-resolution Focus is inaccessible")
        return self._entries[Geometry.focus(candidate)]

    def read(self, geometry):
        if geometry.resolution == 256:
            candidate = geometry.camera * 4 + (geometry.roi[1] // 256) * 2 + geometry.roi[0] // 256
            return self.read_focus(candidate)
        return self._entries[geometry]

    def audit(self):
        return {"observation_key": self.observation_key, "encoder_hash": self.encoder_hash,
                "selected": list(self.selected), "encoder_images": self.encoder_images,
                "entries": [{"camera": geometry.camera, "roi": geometry.roi,
                             "grid": feature.grid, "shape": list(feature.tokens.shape),
                             "dtype": str(feature.tokens.dtype)} for geometry, feature in self._entries.items()]}


def observation_key(images, pose):
    digest = hashlib.sha256(str(tuple(float(value) for value in pose)).encode())
    for camera in CAMERAS:
        digest.update(images[camera].tobytes())
    return digest.hexdigest()


def candidate_summaries(glance):
    if tuple(glance.shape) != (4, 16, 2560):
        raise ValueError("Glance must be [4,16,2560]")
    spatial = glance.reshape(4, 4, 4, 2560)
    return torch.stack([spatial[camera, row * 2:row * 2 + 2, column * 2:column * 2 + 2].mean((0, 1))
                        for camera in range(4) for row in range(2) for column in range(2)])


class SpatialAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(8, 2560, bias=False)
        torch.nn.init.zeros_(self.projection.weight)

    def forward(self, feature):
        geometry = torch.tensor(feature.geometry.vector(), device=feature.tokens.device,
                                dtype=self.projection.weight.dtype)
        return feature.tokens + self.projection(geometry).to(feature.tokens.dtype)
