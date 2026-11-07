from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task22.actor import HierarchicalDistribution, LatentAction
from Experiment.SCOPE.Stage2.Task23.runtime import PPOPolicy
from Experiment.SCOPE.Stage2.Task23.server import serve
from Experiment.SCOPE.Stage2.Task21.fusion import CAMERAS, align_area


def fingerprint(value):
    checksum = hashlib.sha256()

    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                checksum.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        elif isinstance(item, torch.Tensor):
            checksum.update(str((tuple(item.shape), item.dtype)).encode())
            checksum.update(item.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            checksum.update(str((item.shape, item.dtype)).encode())
            checksum.update(item.tobytes())
        elif hasattr(item, "__array_interface__"):
            visit(np.asarray(item))
        else:
            checksum.update(json.dumps(item, sort_keys=True, default=str).encode())

    visit(value)
    return checksum.hexdigest()


class DiagnosticPolicy(PPOPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostic_mode = "native"
        self.branch_mode = None
        self.branch_index = 0
        original_choose = HierarchicalDistribution.choose

        def choose(distribution, mode, generator=None):
            if self.branch_mode == "none":
                return LatentAction(True)
            if self.branch_mode in ("fixed", "random"):
                camera = self.branch_index % 4
                latent = torch.zeros(2, device=distribution.position.loc.device)
                if self.branch_mode == "random":
                    camera = int(torch.randint(4, (), device=latent.device, generator=generator))
                    center = torch.rand(2, device=latent.device, generator=generator).clamp(1e-6, 1 - 1e-6)
                    latent = torch.logit(center)
                self.branch_index += 1
                return LatentAction(False, camera, latent)
            if self.diagnostic_mode == "forced4":
                camera = int(distribution.camera.probs.argmax())
                return LatentAction(False, camera, distribution.position.loc[camera].detach().clone())
            return original_choose(distribution, mode, generator)

        HierarchicalDistribution.choose = choose

    def predict_session(self, session, *, frames, instruction):
        self.branch_mode = frames[-1].get("task231_variant")
        self.branch_index = 0
        if self.branch_mode == "random":
            session["actor_generator"].manual_seed(frames[-1].get("task231_roi_seed", 233))
        response = super().predict_session(session, frames=frames, instruction=instruction)
        if self.branch_mode is not None:
            _, trace = session["pending"]
            blocks = self.last_blocks
            diagnostic = {"context_blocks_sha256": fingerprint(blocks), "context_grids": self.last_grids,
                          "history_frames": len(session["history"].observations), "instruction_sha256": fingerprint(instruction),
                          "pre_focus_sha256": fingerprint(trace["decisions"][0]["state"]),
                          "body_pose": trace["body_pose"].tolist(), "camera_order": list(CAMERAS),
                          "roi": [record["roi"] for record in trace["decisions"] if not record["action"]["stop"]]}
            with torch.no_grad():
                teachers = self.native_encode([self.raw[camera] for camera in CAMERAS])
                target = torch.stack([align_area(tokens, (0, 0, 512, 512), (0, 0, 512, 512))[0] for tokens, _ in teachers]).cpu()
                final = trace["final_state"]
                diagnostic["teacher_mse_glance"] = float((final["glance"].float() - target).square().mean())
                diagnostic["teacher_mse_fused"] = float((final["fused"].float() - target).square().mean())
                diagnostic["coverage_mean"] = float(final["coverage"].mean())
                diagnostic["token_change_mse"] = float((final["fused"].float() - final["glance"].float()).square().mean())
                covered = final["coverage"].squeeze(-1) > 0
                diagnostic["covered_teacher_mse_glance"] = float((final["glance"][covered].float() - target[covered]).square().mean()) if covered.any() else None
                diagnostic["covered_teacher_mse_fused"] = float((final["fused"][covered].float() - target[covered]).square().mean()) if covered.any() else None
                diagnostic["camera_coverage"] = final["coverage"].mean(dim=1).flatten().tolist()
            response["metadata"]["task231"] = diagnostic
        return response

    def restore(self, *, checkpoint, diagnostic_mode="native"):
        if diagnostic_mode not in ("native", "forced4"):
            raise ValueError(diagnostic_mode)
        result = super().restore(checkpoint=checkpoint)
        self.diagnostic_mode = diagnostic_mode
        return result | {"diagnostic_mode": diagnostic_mode}

    def update(self, **kwargs):
        raise RuntimeError("Task231 is diagnostic-only; optimizer updates are forbidden")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10231)
    args = parser.parse_args()
    policy = DiagnosticPolicy(output=args.output, config=args.config, manifest=args.manifest)
    asyncio.run(serve(policy, args.port))


if __name__ == "__main__":
    main()
