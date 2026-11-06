from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
sys.path.insert(0, str(ROOT.parent))
from scope_server import ScopePolicy
from data_guard import PARTITION, STAGE1, sha256
sys.path.insert(0, str(ROOT))
if __package__:
    from .fusion import CAMERAS, VERSION, Focus, SpatialFusion
else:
    from fusion import CAMERAS, VERSION, Focus, SpatialFusion


class FusionPolicy(ScopePolicy):
    def __init__(self, *, output, checkpoint=None):
        manifest = json.loads((STAGE1 / "EXECUTION_MANIFEST.json").read_text())["checkpoint"]
        if sha256(manifest["path"]) != manifest["sha256"]:
            raise ValueError("frozen navigation checkpoint hash mismatch")
        super().__init__(checkpoint=manifest["path"], legacy_root=str(ROOT.parent.parent / "legacy_runtime"),
                         qwen_root=manifest["qwen_root"], telemetry=Path(output) / "scope.jsonl")
        self.model.requires_grad_(False).eval()
        self.backbone_hash = manifest["sha256"]
        self.fusion = SpatialFusion().to("cuda").freeze()
        self.calibrated = False
        if checkpoint is not None:
            self.load_fusion(checkpoint)
        self.enabled = True
        self.requests = ()
        self.last_current = None
        self.last_grids = []
        self.last_blocks = []
        self.last_focus_ms = 0.0
        original_build = self.model._build_qwen35_inputs
        original_encode = self.model._encode_online_postmerge

        def build(*args, **kwargs):
            inputs, blocks = original_build(*args, **kwargs)
            self.last_blocks = blocks
            self.last_grids = inputs["_nav_context_image_grid_thw"].cpu().tolist()
            return inputs, blocks

        def encode(inputs):
            encoded = original_encode(inputs)
            positions = [position for position, index in enumerate(inputs["_nav_online_indices"])
                         if not self.last_blocks[index].get("is_history", False)]
            names = [self.last_blocks[inputs["_nav_online_indices"][position]]["camera_name"] for position in positions]
            if names != list(CAMERAS):
                raise ValueError(f"current camera order changed: {names}")
            current = torch.stack([encoded[position] for position in positions])
            if tuple(current.shape) != (4, 16, 2560):
                raise ValueError(f"native current contract changed: {current.shape}")
            self.last_native = current.detach().clone()
            if self.enabled and self.requests:
                torch.cuda.synchronize()
                began = time.perf_counter()
                updates = [(focus, self.native_encode([focus.crop(self.raw)])[0][0]) for focus in self.requests]
                current = self.fusion(current, updates)
                torch.cuda.synchronize()
                self.last_focus_ms = (time.perf_counter() - began) * 1000
                for position, tokens in zip(positions, current, strict=True):
                    encoded[position] = tokens
            self.last_current = current.detach().clone()
            return encoded

        self.model._build_qwen35_inputs = build
        self.model._encode_online_postmerge = encode

    def load_fusion(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
        if (payload["version"] != VERSION or payload["backbone_sha256"] != self.backbone_hash
                or payload["partition_sha256"] != sha256(PARTITION) or not payload["training_complete"]):
            raise ValueError("adapter/backbone provenance mismatch")
        self.fusion.load_state_dict(payload["adapter"])
        if not all(torch.isfinite(parameter).all() for parameter in self.fusion.parameters()):
            raise ValueError("nonfinite adapter checkpoint")
        self.fusion.freeze()
        self.calibrated = True

    @torch.no_grad()
    def native_encode(self, images):
        from starVLA.model.modules.qwen35_vision import encode_qwen35_postmerge_batched
        interface = self.model.qwen35_vl_interface
        processed = interface.processor.image_processor(images=images, do_resize=False, return_tensors="pt")
        visual = interface.model.model.visual
        grids = processed["image_grid_thw"].to("cuda")
        tokens = encode_qwen35_postmerge_batched(visual, processed["pixel_values"].to(device="cuda", dtype=visual.dtype), grids)
        return list(zip(tokens, grids.cpu().tolist(), strict=True))

    def predict_session(self, session, *, frames, instruction, focus=()):
        if len(focus) > 4 or not all(isinstance(request, Focus) for request in focus):
            raise ValueError("expected up to four Focus requests")
        if focus and not self.calibrated:
            raise RuntimeError("Focus navigation requires a calibrated adapter checkpoint")
        if any(frame.get("policy_mode", "glance_only") != "glance_only" for frame in frames):
            raise ValueError("fusion runtime uses Glance base, never appended Focus blocks")
        self.requests = tuple(focus)
        self.last_focus_ms = 0.0
        frames = [dict(frame, policy_mode="glance_only") for frame in frames]
        try:
            response = super().predict_session(session, frames=frames, instruction=instruction)
            response["metadata"]["fusion"] = {"focus_count": len(focus), "current_shape": list(self.last_current.shape),
                                               "focus_ms": self.last_focus_ms, "roi": [request.roi for request in focus]}
            return response
        finally:
            self.requests = ()

    def predict_center(self, session, *, frames, instruction, camera=0):
        return self.predict_session(session, frames=frames, instruction=instruction, focus=(Focus(camera),))
