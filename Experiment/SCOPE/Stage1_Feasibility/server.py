from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scope_server import ScopePolicy, serve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_configs import ROOT, sha256, validate_freeze, write_json
from policies import select_pair


class FeasibilityPolicy(ScopePolicy):
    def __init__(self, **kwargs):
        self.vision_kind = "current"
        self.events = []
        self.calls = []
        super().__init__(**kwargs)
        self.selector_label = "feasibility-v1.0-cross-camera"
        self.torch.backends.cudnn.benchmark = False
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.backends.cudnn.allow_tf32 = False
        self.torch.use_deterministic_algorithms(True, warn_only=True)
        self.model.requires_grad_(False)
        original_history = self.model.encode_history_images

        def history(*args, **kwargs):
            previous = self.vision_kind
            self.vision_kind = "history"
            try:
                return original_history(*args, **kwargs)
            finally:
                self.vision_kind = previous

        self.model.encode_history_images = history
        original_build = self.model._build_qwen35_inputs

        def build(*args, **kwargs):
            started = time.perf_counter()
            inputs, blocks = original_build(*args, **kwargs)
            self.build_ms = (time.perf_counter() - started) * 1000
            for record, block in zip(self.block_records, blocks, strict=True):
                record["is_cached_history"] = bool(block.get("is_cached_history", False))
                record["frame_index"] = int(block.get("frame_index", 0))
            self.online_kinds = ["history" if blocks[index].get("is_history", False) else "current"
                                 for index in inputs["_nav_online_indices"]]
            return inputs, blocks

        self.model._build_qwen35_inputs = build
        action_wrapper = self.model.action_model.predict_action
        closure = dict(zip(action_wrapper.__code__.co_freevars, action_wrapper.__closure__))
        original_action = closure["original_action"].cell_contents

        def action(*args, **kwargs):
            return self._measure_call("dit_b", original_action, *args, **kwargs)

        self.model.action_model.predict_action = action
        visual = self.model.qwen35_vl_interface.model.model.visual
        visual.register_forward_pre_hook(self._visual_call, with_kwargs=True)
        write_json(ROOT / "VISION_ARCHITECTURE.json", {
            "class": type(visual).__name__, "repr": str(visual),
            "config": self.model.qwen35_vl_interface.model.config.vision_config.to_dict(),
            "flops_status": "shape-based estimate; excludes elementwise, norm, softmax, RoPE",
            "linear_shapes": {name: [module.in_features, module.out_features]
                              for name, module in visual.named_modules() if isinstance(module, self.torch.nn.Linear)}})

    def _measure_call(self, name, function, *args, **kwargs):
        begin = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        try:
            return function(*args, **kwargs)
        finally:
            end.record()
            self.events.append((name, begin, end, (time.perf_counter() - started) * 1000))

    def _start_hook(self, name):
        def hook(module, args):
            label = "vision_" + self.vision_kind if name == "vision" else name
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts[name] = (label, event, time.perf_counter())
        return hook

    def _end_hook(self, name):
        def hook(module, args, output):
            label, begin, started = self.starts[name]
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
            self.events.append((label, begin, end, (time.perf_counter() - started) * 1000))
        return hook

    def _visual_call(self, module, args, kwargs):
        grid = kwargs.get("grid_thw", args[1] if len(args) > 1 else None)
        self.calls.append({"kind": self.vision_kind, "grids": grid.detach().cpu().tolist(),
                           "pixels_shape": list(args[0].shape),
                           "online_kinds": getattr(self, "online_kinds", []) if self.vision_kind == "current" else []})
        if getattr(self, "capture_profile", False):
            self.profile_inputs.append((args[0].detach(), grid.detach(), list(self.calls[-1]["online_kinds"])))

    def _split_profile(self):
        visual = self.model.qwen35_vl_interface.model.model.visual
        result = {}
        for kind in ("current", "history"):
            total_ms = 0.0
            images = 0
            for pixels, grid, kinds in self.profile_inputs:
                chunks = self.torch.split(pixels, grid.prod(-1).tolist())
                indices = [index for index, label in enumerate(kinds) if label == kind]
                if not indices:
                    continue
                selected_pixels = self.torch.cat([chunks[index] for index in indices])
                selected_grids = grid[indices]
                begin = self.torch.cuda.Event(enable_timing=True)
                end = self.torch.cuda.Event(enable_timing=True)
                begin.record()
                with self.torch.inference_mode():
                    visual(selected_pixels, grid_thw=selected_grids, return_dict=True)
                end.record()
                self.torch.cuda.synchronize()
                total_ms += begin.elapsed_time(end)
                images += len(indices)
            result["profile_vision_" + kind + "_ms"] = total_ms
            result["profile_vision_" + kind + "_images"] = images
        return result

    def predict_session(self, session, *, frames, instruction):
        from starVLA.model.modules import input_context
        self.events, self.calls = [], []
        self.build_ms = 0.0
        self.capture_profile = bool(frames[-1].get("cost_profile", False))
        self.profile_inputs = []
        started = time.perf_counter()
        step = session["prediction_index"]
        original_select = input_context.select_focus
        selector_ms = 0.0
        digest = None

        def select(mode, *, seed, episode_id, step, candidates, **kwargs):
            nonlocal selector_ms, digest
            begin = time.perf_counter()
            pair, digest = select_pair(mode, seed=seed, episode_id=episode_id, step=step)
            selector_ms = (time.perf_counter() - begin) * 1000
            return [{**candidates[index], "focus_index": position} for position, index in enumerate(pair)], pair

        input_context.select_focus = select
        try:
            response = super().predict_session(session, frames=frames, instruction=instruction)
            self.torch.cuda.synchronize()
            record = response["metadata"]["scope"]
            for name in ("vision_current", "vision_history", "llm", "dit_b"):
                record[name + "_ms"] = sum(begin.elapsed_time(end) for label, begin, end, wall in self.events if label == name)
                record[name + "_cpu_enqueue_ms"] = sum(wall for label, begin, end, wall in self.events if label == name)
            record.update(history_visual_tokens=record["total_visual_tokens"] - record["current_visual_tokens"],
                          selector_ms=selector_ms, random_digest=digest, run_seed=42,
                          session_seed=session["seed"], cyclic_phase=step % 8,
                          encoder_calls=list(self.calls), encoder_invocation_count=len(self.calls),
                          encoder_image_count=sum(len(call["grids"]) for call in self.calls),
                          peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
                          model_wall_ms=(time.perf_counter() - started) * 1000,
                          preprocess_build_wall_ms=self.build_ms, current_feature_reuse_ms=0.0,
                          timing="CUDA events; CPU enqueue separately; one request-boundary synchronization",
                          history_contract="task1_compatible_256_pool4_BATS")
            record["vision_ms"] = record["vision_current_ms"] + record["vision_history_ms"]
            record["vision_combined_ms"] = record["vision_ms"]
            mixed = any(len(set(call["online_kinds"])) > 1 for call in self.calls)
            if mixed:
                record["vision_current_ms"] = None
                record["vision_history_ms"] = None
            if self.capture_profile:
                self.capture_profile = False
                record.update(self._split_profile())
                record["profile_scope"] = "separate post-prediction split replay; not additive to mixed-forward latency"
            self.profile_inputs = []
            with (ROOT / "server_detailed.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            return response
        except Exception as error:
            with (ROOT / "server_failures.jsonl").open("a") as stream:
                stream.write(json.dumps({"episode_id": session["episode_id"], "step": step,
                                         "error": repr(error), "elapsed_ms": (time.perf_counter() - started) * 1000}) + "\n")
            raise
        finally:
            input_context.select_focus = original_select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10095)
    args = parser.parse_args()
    manifest = validate_freeze()
    checkpoint = manifest["checkpoint"]
    if sha256(checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("checkpoint integrity failure")
    policy = FeasibilityPolicy(checkpoint=checkpoint["path"], legacy_root=str(ROOT.parent / "legacy_runtime"),
                               qwen_root=checkpoint["qwen_root"], telemetry=str(ROOT / "server_raw.jsonl"))
    asyncio.run(serve(policy, "127.0.0.1", args.port, checkpoint["path"]))


if __name__ == "__main__":
    main()
