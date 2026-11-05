from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
CONFIGS = ROOT / "STAGE1_GROUP_CONFIGS"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cycle(order):
    require(sorted(order) == list(range(16)), "cyclic order must contain each ROI exactly once")
    pairs = [[order[(2 * step) % 16], order[(2 * step + 1) % 16]] for step in range(16)]
    require(all(len(set(pair)) == 2 for pair in pairs), "duplicate ROI in a cyclic step")
    require(all(pair[0] // 4 != pair[1] // 4 for pair in pairs), "cyclic pair must cover different cameras")
    require(sorted(candidate for pair in pairs[:8] for candidate in pair) == list(range(16)), "incomplete cycle")
    require(pairs[:8] == pairs[8:], "cycle must repeat after eight steps")
    return pairs[:8]


def validate_manifest(path, expected_count, expected_scenes):
    manifest = json.loads(path.read_text())
    episodes = manifest["episodes"]
    require(len(episodes) == expected_count == manifest["count"], f"episode count mismatch: {path}")
    require(len({row['scene_id'] for row in episodes}) == expected_scenes == manifest["scene_count"], "scene count mismatch")
    require([row["ordinal"] for row in episodes] == list(range(expected_count)), "manifest global order changed")
    require(len({row["episode_id"] for row in episodes}) == expected_count, "duplicate episode IDs")
    require(all(len(row["annotation_sha256"]) == 64 for row in episodes), "annotation hash missing")
    for source in manifest["sources"]:
        require(sha256(Path(source["path"])) == source["sha256"], "source annotation changed")
    return manifest


def validate_bundle():
    common = yaml.safe_load((CONFIGS / "common.yaml").read_text())
    require(common["status"] == "DRAFT_REVIEW_REQUIRED", "protocol must remain unapproved")
    require(not common["execution"]["training_authorized"], "training unexpectedly authorized")
    require(not common["execution"]["full_evaluation_authorized"], "evaluation unexpectedly authorized")
    require(common["execution"]["approval_record"] is None, "unexpected approval record")
    require(common["execution"]["freeze_manifest"] is None, "unexpected freeze record")
    require(not common["stage1"]["stop_selection_enabled"], "Stage1 must not include STOP selection")
    require(not common["stage1"]["adaptive_visual_budget_enabled"], "Stage1 must not adapt budget")
    groups = {}
    for path in sorted(CONFIGS.glob("G[0-4]_*.yaml")):
        group = yaml.safe_load(path.read_text())
        require(group["group"] not in groups, "duplicate group")
        require(not group["stop_selection_enabled"], "group enables perceptual STOP")
        tokens = sum((width // 32) * (height // 32) for width, height in group["current_image_sizes"])
        require(tokens == group["current_visual_tokens"], "declared token count mismatches native dimensions")
        groups[group["group"]] = group
    require(set(groups) == {"G0", "G1", "G2", "G3", "G4"}, "five groups required")
    for identifier in ("G2", "G3", "G4"):
        require(groups[identifier]["focus_budget"] == 2, "Focus budgets must equal two")
        require(groups[identifier]["current_visual_tokens"] == 192, "equal-budget token mismatch")
    require(groups["G4"]["selector_requirements"]["checkpoint_paths"] == [], "untrained G4 must not claim weights")
    cycle = validate_cycle(groups["G2"]["schedule"]["roi_order"])
    tasks = {}
    evaluation_scenes = set()
    for name in ("r2r_val_unseen", "rxr_val_unseen"):
        config = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
        require(config["max_samples"] is None, "formal protocol must not inherit a sample cap")
        manifest = validate_manifest(CONFIGS / config["episode_manifest"], config["expected_episodes"], config["expected_scenes"])
        require(config["task"] == manifest["task"], "task/manifest mismatch")
        evaluation_scenes.update(row["scene_id"] for row in manifest["episodes"])
        tasks[config["task"]] = {"episodes": manifest["count"], "scenes": manifest["scene_count"],
                                "manifest_sha256": sha256(CONFIGS / config["episode_manifest"])}
    training = yaml.safe_load((CONFIGS / "selector_training.yaml").read_text())
    partition = json.loads((CONFIGS / training["scene_partition"]["training_scene_manifest"]).read_text())
    train_scenes, dev_scenes = set(partition["training_scenes"]), set(partition["development_scenes"])
    require(not train_scenes & dev_scenes, "train/dev scene overlap")
    require(not (train_scenes | dev_scenes) & evaluation_scenes, "heldout scene leakage")
    train_ids = set(partition["training_episode_ids"])
    dev_ids = set(partition["development_pool_episode_ids"])
    require(not train_ids & dev_ids, "train/dev episode overlap")
    require(len(train_ids) == partition["training_count"], "train count mismatch")
    require(len(dev_ids) == partition["development_pool_count"], "dev count mismatch")
    require(len(partition["development_evaluation_episode_ids"]) == training["checkpoint_selection"]["dev_episode_count"], "dev evaluation count mismatch")
    require(set(partition["development_evaluation_episode_ids"]) <= dev_ids, "dev evaluation outside dev pool")
    for source in partition["sources"]:
        require(sha256(Path(source["path"])) == source["sha256"], "training annotation changed")
    algorithm = training["algorithm"]
    require(algorithm["updates"] * algorithm["transitions_per_update"] == algorithm["maximum_navigation_transitions"], "training budget inconsistent")
    evidence_root = PROJECT / "docs/Logs/Task1"
    evidence = {}
    for name in ("fixed_equivalence.json", "fixed_habitat_equivalence.json", "R2R/fixed_habitat_equivalence.json"):
        payload = json.loads((evidence_root / name).read_text())
        require(len(payload["comparisons"]) == 2, "expected two historical fixed-output comparisons")
        require(all(row["max_abs_difference"] == 0 and row["shape"] == [8, 4] for row in payload["comparisons"]), "historical compatibility failed")
        evidence[name] = sha256(evidence_root / name)
    for name, expected in (("smoke_verification.json", 12), ("R2R/smoke_verification.json", 6)):
        payload = json.loads((evidence_root / name).read_text())
        require(len(payload) == expected and all(row["passed"] for row in payload), "historical smoke verification failed")
        evidence[name] = sha256(evidence_root / name)
    audit = json.loads((evidence_root / "R2R/protocol_audit.json").read_text())
    require(audit["passed"], "historical R2R audit failed")
    documents = ("STAGE1_EXPERIMENT_PROTOCOL.md", "SELECTOR_DESIGN.md", "METRICS_AND_STATISTICS.md",
                 "COST_MEASUREMENT_PROTOCOL.md", "REMAINING_IMPLEMENTATION_TASKS.md", "IMPLEMENTATION_AUDIT.md")
    for name in documents:
        require((ROOT / name).is_file(), f"missing deliverable: {name}")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {"status": "DRAFT_VALIDATED_NOT_APPROVED", "training_started": False, "full_evaluation_started": False,
            "cyclic_first_eight_pairs": cycle, "tasks": tasks,
            "selector_split": {"training_scenes": len(train_scenes), "development_scenes": len(dev_scenes),
                               "training_episodes": len(train_ids), "development_pool": len(dev_ids),
                               "development_evaluation_episodes": len(partition["development_evaluation_episode_ids"])},
            "historical_evidence_sha256": evidence,
            "candidate_bundle_sha256": {str(path.relative_to(PROJECT)): sha256(path) for path in files},
            "warning": "Offline schema/math/source/evidence checks only; not fresh model equivalence, trained selector, or execution approval."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = validate_bundle()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PASS: candidate protocol validated; training and full evaluation remain unauthorized")


if __name__ == "__main__":
    main()
