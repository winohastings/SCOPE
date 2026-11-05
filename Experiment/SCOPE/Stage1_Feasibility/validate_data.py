from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from compile_configs import ROOT, SIMPLENAV, sha256, validate_freeze, write_json

sys.path.insert(0, str(SIMPLENAV))
from benchmark.common.config import load_eval_config
from benchmark.common.runner.worker import _vlnce_gt_locations
from benchmark.common.simulators.habitat.vlnce031_datasets import vlnce_split_gt_path
from benchmark.vlnce.inputs import VLNCEInputAdapter


def main():
    validate_freeze()
    cfg = load_eval_config(ROOT / "configs/G0.yaml")
    existing = VLNCEInputAdapter().load_episodes(cfg.input, max_samples=None)
    by_identifier = {episode.source_episode_id: episode for episode in existing}
    episodes = json.loads((ROOT / "DEV_EPISODES.json").read_text())
    gt = _vlnce_gt_locations(str(cfg.env.kwargs["data_root"]), "r2r", "train", ("guide",))
    gt_path = vlnce_split_gt_path(str(cfg.env.kwargs["data_root"]), task_name="r2r", split="train", roles=("guide",))
    for episode in episodes:
        identifier = episode["source_episode_id"]
        if episode != asdict(by_identifier[identifier]):
            raise ValueError(f"materialized episode differs from existing adapter: {identifier}")
        if not gt[identifier]["locations"]:
            raise ValueError(f"missing dense GT: {identifier}")
        scene_path = Path(cfg.env.kwargs["scenes_dir"]) / episode["payload"]["scene_id"]
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
    write_json(ROOT / "DATA_RUNTIME_VALIDATION.json", {"passed": True, "count": len(episodes),
               "adapter_payloads_exactly_equal": True, "all_scene_files_exist": True,
               "all_dense_gt_locations_exist": True, "gt_path": gt_path, "gt_sha256": sha256(gt_path),
               "baseline_reference": "docs/Logs/2026-11-05-simplenav-r2r-rxr-baseline-smoke.md"})
    print("PASS: all 64 payloads equal existing adapter, all scene/GT files present")


if __name__ == "__main__":
    main()
