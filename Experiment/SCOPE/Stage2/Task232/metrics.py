from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import kl_divergence

from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def roi_metrics(selected):
    pixels = np.zeros((4, 512, 512), dtype=bool)
    repeated = 0
    previous = []
    for camera, bounds in selected:
        left, top, right, bottom = bounds
        pixels[camera, top:bottom, left:right] = True
        overlaps = []
        for old_camera, old_bounds in previous:
            if old_camera != camera:
                continue
            old_left, old_top, old_right, old_bottom = old_bounds
            intersection = max(0, min(right, old_right) - max(left, old_left)) * max(0, min(bottom, old_bottom) - max(top, old_top))
            union = (right - left) * (bottom - top) + (old_right - old_left) * (old_bottom - old_top) - intersection
            overlaps.append(intersection / union)
        repeated += int(bool(overlaps) and max(overlaps) >= .8)
        previous.append((camera, bounds))
    return {"coverage": float(pixels.mean()), "repeated_roi_fraction": repeated / max(len(selected) - 1, 1),
            "unique_cameras": len({camera for camera, _ in selected}), "focus_count": len(selected)}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def fixed_probe(actor, output, manifest_path, parent_checkpoint, label):
    manifest = json.loads(Path(manifest_path).read_text())
    device = next(actor.parameters()).device
    initial = SpatialQueryActor().to(device).eval()
    initial.load_state_dict(torch.load(parent_checkpoint, map_location="cpu", weights_only=False)["trainer"]["actor"])
    rows = []
    for record in manifest["pre_focus"]:
        assert sha256(record["path"]) == record["sha256"]
        trace = torch.load(record["path"], map_location="cpu", weights_only=True)
        state = ActorState.unpack(trace["decisions"][0]["state"], device)
        assert state.count == 0
        current, reference = actor(state), initial(state)
        row = {"split": record["split"], "path": record["path"], "p_stop": float(current.stop.probs),
               "camera_entropy": float(current.camera.entropy()), "camera_top1": int(current.camera.probs.argmax()),
               "stop_KL_from_fork": float(kl_divergence(reference.stop, current.stop)),
               "camera_KL_from_fork": float(kl_divergence(reference.camera, current.camera)),
               "position_KL_from_fork": float((reference.camera.probs * kl_divergence(reference.position, current.position).sum(-1)).sum())}
        for camera in range(4):
            row[f"camera_p_{camera}"] = float(current.camera.probs[camera])
            for axis in range(2):
                row[f"position_mean_{camera}_{axis}"] = float(current.position.loc[camera, axis])
                row[f"position_sigma_{camera}_{axis}"] = float(current.position.scale[camera, axis])
        rows.append(row)
    geometry = []
    for record in manifest["geometry"]:
        assert sha256(record["path"]) == record["sha256"]
        trace = torch.load(record["path"], map_location="cpu", weights_only=True)
        selected, native, stopped = [], [], False
        for decision in trace["decisions"]:
            state = ActorState.unpack(decision["state"], device)
            distribution = actor(state)
            camera = int(distribution.camera.probs.argmax())
            from Experiment.SCOPE.Stage2.Task22.actor import LatentAction
            action = LatentAction(False, camera, distribution.position.loc[camera].detach())
            selected.append((camera, action.focus().roi))
            stopped = stopped or bool(distribution.stop.probs >= .5)
            if not stopped:
                native.append(selected[-1])
        geometry.append({"path": record["path"], "scene": record["scene"],
                         **{f"forced4_{key}": value for key, value in roi_metrics(selected).items()},
                         **{f"native_{key}": value for key, value in roi_metrics(native).items()}})
    summaries = {}
    for split in ("train", "val_seen"):
        group = [row for row in rows if row["split"] == split]
        probabilities = np.asarray([row["p_stop"] for row in group])
        summaries[split] = {"states": len(group), "p_stop_P10_P50_P90": np.quantile(probabilities, [.1, .5, .9]).tolist(),
                            "zero_focus_first_step_fraction": float(np.mean(probabilities >= .5)),
                            "stop_ge_045_fraction": float(np.mean(probabilities >= .45)),
                            "camera_top1_fractions": {str(camera): sum(row["camera_top1"] == camera for row in group) / len(group) for camera in range(4)},
                            "camera_entropy": float(np.mean([row["camera_entropy"] for row in group])),
                            "KL_from_fork": {name: float(np.mean([row[f"{name}_KL_from_fork"] for row in group])) for name in ("stop", "camera", "position")}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / f"{label}_heads.csv", rows)
    write_csv(output / f"{label}_geometry.csv", geometry)
    result = {"heads": summaries, "geometry_states": len(geometry),
              "geometry": {key: float(np.mean([row[key] for row in geometry])) for key in geometry[0] if key not in ("path", "scene")},
              "geometry_caveat": "frozen reference intermediate ActorStates; conditional ROI geometry, not fresh fused-feature closed-loop behavior"}
    write_json(output / f"{label}.json", result)
    return result
