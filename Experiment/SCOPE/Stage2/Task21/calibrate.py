from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from data_guard import PARTITION, check_episode, sha256, write_json
sys.path.insert(0, str(ROOT))
if __package__:
    from .fusion import CAMERAS, VERSION, Focus, SpatialFusion, align_area, calibration_target
    from .runtime import FusionPolicy
else:
    from fusion import CAMERAS, VERSION, Focus, SpatialFusion, align_area, calibration_target
    from runtime import FusionPolicy


def summarize(glance, delta, target, mask, gate, quantize=False):
    predicted = glance + gate * delta
    if quantize:
        predicted = predicted.bfloat16().float()
    covered = mask.expand_as(glance) > 0
    preserved = ~covered
    return {"covered_mse": float((predicted - target)[covered].square().mean()),
            "noncovered_max_abs": float((predicted - glance)[preserved].abs().max()),
            "noncovered_mse": float((predicted - glance)[preserved].square().mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.config.read_text())
    if (settings["raw_size"], settings["patch_size"], settings["focus_encode_size"],
        settings["glance_size"], settings["max_focus"]) != (512, 192, 256, 128, 4):
        raise ValueError("fixed infrastructure configuration mismatch")
    torch.manual_seed(settings["seed"])
    records = json.loads((args.capture / "manifest.json").read_text())
    if not 0 < settings["calibration_episodes"] < len(records):
        raise ValueError("need disjoint calibration and diagnostic episodes")
    if len({record["source_episode_id"] for record in records}) != len(records):
        raise ValueError("duplicate calibration episodes")
    policy = FusionPolicy(output=args.output)
    versions = {name: parameter._version for name, parameter in policy.model.named_parameters()}
    adapter = SpatialFusion(settings["initial_gate"]).cuda()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=settings["learning_rate"])
    samples, observations, grids_seen = [], [], {}
    started = time.perf_counter()
    for ordinal, record in enumerate(records):
        check_episode(record["source_episode_id"], record["scene_id"], "train")
        source = args.capture / record["path"]
        if sha256(source) != record["sha256"]:
            raise ValueError("captured image digest mismatch")
        with np.load(source) as archive:
            images = {camera: archive[camera].copy() for camera in CAMERAS}
            pose = archive["body_pose"].copy()
        observations.append((record, images, pose))
        with torch.no_grad():
            glance_encoded = policy.native_encode([Image.fromarray(images[camera]).resize((128, 128), Image.Resampling.BICUBIC)
                                                   for camera in CAMERAS])
            glance = torch.stack([tokens for tokens, grid in glance_encoded]).float()
            grids_seen["glance"] = {"grids": [grid for tokens, grid in glance_encoded], "shape": list(glance.shape)}
            for camera in range(4):
                teacher, teacher_grid = policy.native_encode([Image.fromarray(images[CAMERAS[camera]])])[0]
                grids_seen["teacher"] = {"grid": teacher_grid, "shape": list(teacher.shape)}
                for horizontal, vertical in settings["focus_positions"]:
                    focus = Focus(camera, horizontal, vertical)
                    tokens, grid = policy.native_encode([focus.crop(images)])[0]
                    grids_seen["focus"] = {"grid": grid, "shape": list(tokens.shape)}
                    aligned, coverage = align_area(tokens, focus.roi, focus.roi)
                    delta = torch.zeros_like(glance)
                    delta[camera] = coverage * (aligned - glance[camera])
                    target, mask = calibration_target(glance, teacher, focus)
                    samples.append((ordinal, glance, delta, target, mask))
        print(f"encoded episode {record['source_episode_id']} ({ordinal + 1}/{len(records)})", flush=True)
    split = settings["calibration_episodes"]
    groups = {}
    for name, predicate in (("calibration", lambda ordinal: ordinal < split), ("diagnostic", lambda ordinal: ordinal >= split)):
        selected = [sample for sample in samples if predicate(sample[0])]
        groups[name] = tuple(torch.stack([sample[index] for sample in selected]) for index in range(1, 5))
    glance, delta, target, mask = groups["calibration"]
    covered = mask.expand_as(glance) > 0
    trace = []
    for step in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        predicted = glance + adapter.gate_logit.sigmoid() * delta
        loss = (predicted - target)[covered].square().mean()
        preservation = (predicted - glance)[~covered].square().mean()
        total = loss + settings["preservation_weight"] * preservation
        total.backward()
        optimizer.step()
        trace.append({"step": step + 1, "loss": float(total.detach()), "gate": float(adapter.gate_logit.sigmoid().detach())})
    adapter.freeze()
    gate = float(adapter.gate_logit.sigmoid())
    metrics = {name: {"glance": summarize(*values, 0.0), "initial": summarize(*values, settings["initial_gate"]),
                      "calibrated": summarize(*values, gate),
                      "deployed_bfloat16": summarize(*values, gate, quantize=True),
                      "samples": len(values[0])} for name, values in groups.items()}
    checkpoint = args.output / "adapter.pt"
    torch.save({"version": VERSION, "adapter": adapter.state_dict(), "backbone_sha256": policy.backbone_hash,
                "partition_sha256": sha256(PARTITION), "capture_manifest_sha256": sha256(args.capture / "manifest.json"),
                "config": settings, "calibration_episode_ids": [record["source_episode_id"] for record in records[:split]],
                "diagnostic_episode_ids": [record["source_episode_id"] for record in records[split:]],
                "training_complete": True}, checkpoint)
    policy.load_fusion(checkpoint)
    write_json(args.output / "calibration.json", {"gate": gate, "metrics": metrics, "trace": trace, "measured_grids": grids_seen})
    consistency = []
    with torch.inference_mode():
        for record, images, pose in observations[-2:]:
            baseline = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=settings["seed"])
            fused = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=settings["seed"])
            for tick in range(3):
                frame = {"images": images, "body_pose": pose, "timestamp_s": float(tick), "policy_mode": "glance_only"}
                policy.enabled = False
                reference = policy.predict_session(baseline, frames=[frame], instruction=record["instruction"])
                reference_tokens = policy.last_current.clone()
                reference_grids = policy.last_grids
                policy.enabled = True
                actual = policy.predict_session(fused, frames=[frame], instruction=record["instruction"])
                difference = float(np.max(np.abs(reference["actions"] - actual["actions"])))
                token_difference = float((reference_tokens - policy.last_current).abs().max())
                consistency.append({"episode": record["source_episode_id"], "tick": tick, "action_max_abs": difference,
                                    "feature_max_abs": token_difference, "context_grids": policy.last_grids,
                                    "grids_equal": reference_grids == policy.last_grids,
                                    "stop_equal": reference["stop"] == actual["stop"]})
                if difference != 0 or token_difference != 0 or reference_grids != policy.last_grids:
                    raise AssertionError("no-Focus equivalence failed")
        record, images, pose = observations[-1]
        glance_images = [Image.fromarray(images[camera]).resize((128, 128), Image.Resampling.BICUBIC) for camera in CAMERAS]
        requests = [Focus(camera, *settings["focus_positions"][camera]) for camera in range(4)]
        timing = []
        for count in range(5):
            elapsed = []
            vision_elapsed = []
            for repeat in range(settings["timing_repeats"] + 1):
                policy.timings = {}
                torch.cuda.synchronize()
                began = time.perf_counter()
                base = torch.stack([tokens for tokens, grid in policy.native_encode(glance_images)])
                updates = [(request, policy.native_encode([request.crop(images)])[0][0]) for request in requests[:count]]
                result = policy.fusion(base, updates)
                torch.cuda.synchronize()
                duration = (time.perf_counter() - began) * 1000
                if repeat:
                    elapsed.append(duration)
                    vision_elapsed.append(policy.timings["vision_ms"])
            session = policy.new_session(episode_id=record["source_episode_id"], statistics_key=None, seed=settings["seed"])
            response = policy.predict_session(session, frames=[{"images": images, "body_pose": pose}],
                                              instruction=record["instruction"], focus=requests[:count])
            support = torch.zeros((4, 16, 1), device="cuda", dtype=torch.bool)
            for request in requests[:count]:
                _, coverage = align_area(torch.zeros(64, 1, device="cuda"), request.roi, request.roi)
                support[request.camera] |= coverage > 0
            difference = (policy.last_current - policy.last_native).abs()
            outside = float(difference[~support.expand_as(difference)].max())
            inside = float(difference[support.expand_as(difference)].max()) if count else 0.0
            if outside != 0 or (count and inside == 0):
                raise AssertionError("native runtime spatial writeback failed")
            timing.append({"focus_count": count, "encoding_and_fusion_ms": elapsed, "mean_ms": float(np.mean(elapsed)),
                           "vision_only_ms": vision_elapsed, "vision_only_mean_ms": float(np.mean(vision_elapsed)),
                           "shape": list(result.shape), "navigation_current_tokens": response["metadata"]["scope"]["current_visual_tokens"],
                           "runtime_noncovered_max_abs": outside, "runtime_covered_max_abs": inside,
                           "vision_encoded_postmerge_tokens": 64 + count * 64})
    unchanged = all(parameter._version == versions[name] for name, parameter in policy.model.named_parameters())
    frozen = all(not parameter.requires_grad and parameter.grad is None for parameter in policy.model.parameters())
    passed = (unchanged and frozen and all(item["navigation_current_tokens"] == 64 for item in timing)
              and all(item["calibrated"]["noncovered_max_abs"] == 0 for item in metrics.values())
              and metrics["diagnostic"]["deployed_bfloat16"]["covered_mse"] < metrics["diagnostic"]["glance"]["covered_mse"])
    report = {"passed": passed, "version": VERSION, "gate": gate, "metrics": metrics, "measured_grids": grids_seen,
              "no_focus_checks": consistency, "timing": timing, "backbone_parameters_unchanged": unchanged,
              "backbone_frozen_and_no_grad": frozen, "adapter_frozen": all(not parameter.requires_grad for parameter in policy.fusion.parameters()),
              "checkpoint_reload_equal": all(torch.equal(value, policy.fusion.state_dict()[name]) for name, value in adapter.state_dict().items()),
              "checkpoint_sha256": sha256(checkpoint), "elapsed_seconds": time.perf_counter() - started,
              "calibration_episode_ids": [record["source_episode_id"] for record in records[:split]],
              "diagnostic_episode_ids": [record["source_episode_id"] for record in records[split:]],
              "scene_ids": [record["scene_id"] for record in records],
              "limitations": ["8 train initial states only; diagnostic episodes may share calibration scenes",
                              "no-focus checks replay static observations with BATS history; not full closed-loop navigation",
                              "teacher supervision restricted geometrically; full-image encoder itself has global receptive field"]}
    write_json(args.output / "validation.json", report)
    if not passed:
        raise AssertionError("Task2.1 validation gate failed; do not enter Task2.2")
    print(json.dumps({"passed": passed, "gate": gate, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
