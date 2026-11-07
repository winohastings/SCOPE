from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task22.actor import ActorState
from Experiment.SCOPE.Stage2.Task231.server import fingerprint
from Experiment.SCOPE.Stage2.Task233.teacher_candidates import geometry, roi
from Experiment.SCOPE.Stage2.Task233.expert_scoring import score
from Experiment.SCOPE.Stage2.Task233.analyze import write_csv


def verify(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "ANCHOR_MANIFEST_256.json").read_text())
    anchors = manifest["anchors"]
    evidence = {path: [json.loads(line) for line in Path(path).read_text().splitlines()]
                for path in {anchor["evidence"] for anchor in anchors}}
    lookup = {}
    for path, records in evidence.items():
        groups = defaultdict(list)
        for record in records:
            groups[record["anchor_id"]].append(record)
        lookup[path] = groups
    checks = {"exactly_256_noncalibration": len(anchors) == 256,
              "unique_anchor_ids": len({anchor["anchor_id"] for anchor in anchors}) == 256,
              "all_61_train_scenes": len({anchor["scene"] for anchor in anchors}) == 61}
    max_score_error = 0.
    branches = 0
    state_counts = Counter()
    source_counts = Counter()
    scene_counts = Counter()
    spatial = []
    physical_states = set()
    partitions = defaultdict(set)
    for anchor in anchors:
        state_path = Path(anchor["state_path"])
        if sha256(state_path) != anchor["state_sha256"]:
            raise ValueError("ActorState changed")
        state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
        if state_payload["data_kind"] != "teacher_candidate" or state_payload["ppo_eligible"]:
            raise ValueError("teacher state confused with on-policy")
        state = ActorState.unpack(state_payload["state"], "cpu")
        records = lookup[anchor["evidence"]][anchor["anchor_id"]]
        baseline = next(row for row in records if row["variant"] == "stop")
        if fingerprint(state.packed()) != baseline["diagnostic"]["anchor_state_sha256"] or state.count != anchor["j"]:
            raise ValueError("stored ActorState differs from executed state")
        identity = baseline["restoration_identity"]
        physical = (anchor["scene"], str(anchor["trajectory_id"]), tuple(identity["pose"]), identity["images"])
        if physical in physical_states:
            raise ValueError("physical-state duplicate")
        physical_states.add(physical)
        partitions[anchor["partition"]].add((anchor["scene"], str(anchor["trajectory_id"])))
        state_counts[anchor["partition"]] += 1
        source_counts[anchor["state_source"]] += 1
        scene_counts[anchor["scene"]] += 1
        variants = {row["variant"] for row in records}
        if len(variants) != len(records) or not {f"candidate_{index}" for index in range(8)}.issubset(variants):
            raise ValueError("incomplete candidate set")
        for record in records:
            if not record["restoration_valid"] or record["split"] != "train" or record["ppo_eligible"]:
                raise ValueError("split/restoration/data-kind violation")
            diagnostic = record["diagnostic"]
            if diagnostic["current_tokens"] != 64:
                raise ValueError("current token budget violation")
            selected = diagnostic["selected"]
            if len(selected) > 4 or any(choice["roi"] != roi(choice) for choice in selected):
                raise ValueError("illegal executed ROI")
            if any(diagnostic[key] != value for key, value in geometry(selected).items()):
                raise ValueError("coverage/repetition mismatch")
            expected = score(record["actions"], record["navigation_stopped"], record["expert"])
            max_score_error = max(max_score_error, *(abs(record[key] - value) for key, value in expected.items()))
            branches += 1
        spatial.append({"scene": anchor["scene"], "anchor_id": anchor["anchor_id"], "partition": anchor["partition"],
                        "source": anchor["state_source"], "j": anchor["j"], "branches": len(records)})
    checks.update(physical_states_unique=len(physical_states) == 256,
                  trajectory_split_disjoint=not bool(partitions["teacher_train"] & partitions["teacher_heldout"]),
                  scoring_replay_exact=max_score_error == 0., states_verified=len(anchors) == 256)
    complete = json.loads((output / "COMPLETE.json").read_text())
    fresh_rows = evidence[str(output / "candidate_results.jsonl")]
    checks.update(frozen_parameters=all(complete["audit"]["checks"].values()),
                  no_training=complete["audit"]["optimizer_updates"] == complete["audit"]["critic_training_samples"] == 0,
                  prediction_accounting=complete["audit"]["model_predictions"] == len(fresh_rows),
                  focus_accounting=complete["audit"]["focus_encoder_calls"] == sum(row["diagnostic"]["focus_count"] for row in fresh_rows))
    frozen = json.loads((output / "SOURCE_INTEGRITY_BEFORE.json").read_text())
    changed = [path for path, expected in frozen.items() if sha256(path) != expected]
    checks["old_artifacts_and_sources_unchanged"] = not changed
    result = {"passed": all(checks.values()), "checks": checks, "unique_anchors": len(anchors),
              "evidence_branches_noncalibration": branches, "partitions": dict(state_counts), "state_sources": dict(source_counts),
              "scene_counts": dict(sorted(scene_counts.items())), "max_score_replay_error": max_score_error,
              "changed_inputs": changed, "manifest_sha256": sha256(output / "ANCHOR_MANIFEST_256.json"),
              "verifier_sha256": sha256(__file__)}
    write_json(output / "EXPANSION_VERIFICATION.json", result)
    write_csv(output / "anchor_scene_source_counts.csv", spatial)
    if not result["passed"]:
        raise AssertionError(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.output)


if __name__ == "__main__":
    main()
