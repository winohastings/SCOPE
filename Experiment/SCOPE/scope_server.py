from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SIMPLENAV = Path("/home/data/czh/Third_Party/baselines/SimpleNav")
sys.path.insert(0, str(SIMPLENAV))
from deployment.model_server.legacy_rxr_server import LegacyRXRPolicy, serve


class ScopePolicy(LegacyRXRPolicy):
    def __init__(self, *, telemetry, selector=None, smoke_selector=False, **kwargs):
        super().__init__(**kwargs)
        self.telemetry = Path(telemetry)
        self.telemetry.parent.mkdir(parents=True, exist_ok=True)
        self.selector = None
        self.selector_label = "none"
        if selector:
            spec = importlib.util.spec_from_file_location("scope_selector", selector)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.selector = module.select
            self.selector_label = str(Path(selector).resolve())
        elif smoke_selector:
            self.selector = smoke_select
            self.selector_label = "SCRIPTED_INTERFACE_SMOKE_NOT_LEARNED"
        self.timings = {}
        self.starts = {}
        backbone = self.model.qwen35_vl_interface.model.model
        for name, module in (("vision", backbone.visual), ("llm", backbone.language_model)):
            module.register_forward_pre_hook(self._start_hook(name))
            module.register_forward_hook(self._end_hook(name))
        original_action = self.model.action_model.predict_action

        def timed_action(*args, **kwargs):
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            result = original_action(*args, **kwargs)
            self.torch.cuda.synchronize()
            self.timings["dit_b_ms"] = (time.perf_counter() - started) * 1000
            return result

        self.model.action_model.predict_action = timed_action
        original_build = self.model._build_qwen35_inputs

        def build(*args, **kwargs):
            inputs, blocks = original_build(*args, **kwargs)
            from starVLA.model.modules.qwen35_vision import qwen35_postmerge_token_count
            grids = inputs["_nav_context_image_grid_thw"].cpu().tolist()
            self.block_records = [{"camera_name": block["camera_name"],
                                   "is_history": bool(block.get("is_history", False)),
                                   "grid_thw": grid, "tokens": qwen35_postmerge_token_count(grid),
                                   **{key: block[key] for key in ("scope_kind", "camera_id", "roi", "focus_index") if key in block}}
                                  for block, grid in zip(blocks, grids, strict=True)]
            current = [record["tokens"] for record in self.block_records if not record["is_history"]]
            expected = [64] * 4 if self.mode == "original" else [16] * 4 + [64] * len(self.focus)
            if current != expected:
                raise ValueError(f"Unexpected current tokens {current} != {expected}")
            return inputs, blocks

        self.model._build_qwen35_inputs = build

    def _start_hook(self, name):
        def hook(module, args):
            self.torch.cuda.synchronize()
            self.starts[name] = time.perf_counter()
        return hook

    def _end_hook(self, name):
        def hook(module, args, output):
            self.torch.cuda.synchronize()
            self.timings[name + "_ms"] = self.timings.get(name + "_ms", 0) + (time.perf_counter() - self.starts[name]) * 1000
        return hook

    @staticmethod
    def _observation(frame, frame_index):
        observation = LegacyRXRPolicy._observation(frame, frame_index)
        observation["images"] = {camera: np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BICUBIC))
                                 if image.shape[:2] != (256, 256) else image
                                 for camera, image in observation["images"].items()}
        observation["image"] = observation["images"]["front"]
        return observation

    def _seeded_prediction(self, sample, seed):
        if self.mode != "original":
            sample["images"] = self.glance
            sample["scope_blocks"] = self.focus
        return super()._seeded_prediction(sample, seed)

    def predict_session(self, session, *, frames, instruction):
        from starVLA.model.modules.input_context import POLICY_MODES, prepare_scope_images, select_focus
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        self.mode = frames[-1].get("policy_mode", "original")
        if self.mode not in POLICY_MODES:
            raise ValueError(self.mode)
        self.focus, trace = [], []
        self.model.scope_enabled = self.mode != "original"
        self.model.qwen35_vl_interface.scope_enabled = self.mode != "original"
        if self.mode != "original":
            self.raw, compatible, self.glance, candidates = prepare_scope_images(frames[-1]["images"])
            self.focus, trace = select_focus(self.mode, seed=session["seed"], episode_id=session["episode_id"],
                                             step=session["prediction_index"], glance=self.glance,
                                             candidates=candidates, selector=self.selector)
        self.timings = {}
        self.torch.cuda.reset_peak_memory_stats()
        response = super().predict_session(session, frames=frames, instruction=instruction)
        self.torch.cuda.synchronize()
        record = {"episode_id": session["episode_id"], "step": session["prediction_index"] - 1,
                  "policy_mode": self.mode, "selector": self.selector_label,
                  "focus_count": len(self.focus), "selection_trace": trace,
                  "focus": [{key: value for key, value in focus.items() if key != "image"} for focus in self.focus],
                  "blocks": self.block_records,
                  "current_visual_tokens": sum(block["tokens"] for block in self.block_records if not block["is_history"]),
                  "total_visual_tokens": sum(block["tokens"] for block in self.block_records),
                  **self.timings, "server_total_ms": (time.perf_counter() - started) * 1000,
                  "peak_gpu_memory_bytes": self.torch.cuda.max_memory_allocated(),
                  "predicted_body_waypoints": response["actions"].tolist()}
        response.setdefault("metadata", {})["scope"] = record
        with self.telemetry.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return response


def smoke_select(*, selected, available, allow_stop, focus_index, **kwargs):
    if allow_stop and focus_index == 1:
        return 16
    return available[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--legacy-root", default=str(Path(__file__).parent / "legacy_runtime"))
    parser.add_argument("--qwen-root", default="/home/data/czh/Third_Party/models/Qwen3.5-4B")
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--selector")
    parser.add_argument("--smoke-selector", action="store_true")
    parser.add_argument("--port", type=int, default=10094)
    args = parser.parse_args()
    if args.selector and args.smoke_selector:
        parser.error("--selector and --smoke-selector are mutually exclusive")
    policy = ScopePolicy(checkpoint=args.checkpoint, legacy_root=args.legacy_root, qwen_root=args.qwen_root,
                         telemetry=args.telemetry, selector=args.selector, smoke_selector=args.smoke_selector)
    asyncio.run(serve(policy, "127.0.0.1", args.port, args.checkpoint))


if __name__ == "__main__":
    main()
