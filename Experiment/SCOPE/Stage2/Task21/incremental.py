from __future__ import annotations

import torch
import torch.nn.functional as functional


class IncrementalPerception:
    def __init__(self, glance, adapter, encode_focus):
        if tuple(glance.shape) != (4, 16, 2560) or glance.requires_grad:
            raise ValueError("frozen four-camera Glance required")
        if any(parameter.requires_grad for parameter in adapter.parameters()):
            raise ValueError("incremental perception requires frozen Adapter")
        self.glance = glance
        self.current = glance
        self.adapter = adapter
        self.encode_focus = encode_focus
        self.selected = []
        self.coverage = torch.zeros(4, 16, 1, device=glance.device)
        self.history = torch.zeros(4, 7, device=glance.device)
        self._pixels = torch.zeros(4, 512, 512, device=glance.device, dtype=torch.bool)
        self.closed = False

    @torch.no_grad()
    def acquire(self, focus):
        if self.closed or len(self.selected) >= 4:
            raise RuntimeError("perception closed or four-Focus limit reached")
        tokens = self.encode_focus(focus)
        updated = self.adapter(self.current, [(focus, tokens)])
        left, top, right, bottom = focus.roi
        self._pixels[focus.camera, top:bottom, left:right] = True
        self.coverage = functional.avg_pool2d(self._pixels[:, None].float(), 128).reshape(4, 16, 1)
        self.history = self.history.clone()
        self.history[len(self.selected), focus.camera] = 1
        self.history[len(self.selected), 4:] = self.history.new_tensor([focus.x, focus.y, 1])
        self.selected.append(focus)
        self.current = updated.detach()
        return self.current

    def finish(self):
        self.closed = True
        return self.current
