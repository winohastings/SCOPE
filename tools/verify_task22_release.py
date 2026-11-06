from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil

from verify_release import ROOT, digest, verify


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_task22(root=ROOT):
    archive = verify(root)
    artifacts = root / "Experiment/SCOPE/Stage2/Task22/artifacts"
    validation = read(artifacts / "verified_validation/VALIDATION.json")
    contract = read(artifacts / "final_runtime/model_contract.json")
    verified_contract = read(artifacts / "verified_validation/model_contract.json")
    summary = read(artifacts / "final_runtime/trace_summary.json")
    require(validation["passed"] and validation["reload_equal"] and validation["optimizer_updates"] == 0,
            "Validation or untrained Actor contract failed")
    require(validation["actor_parameters"] == contract["actor_parameters"] == 2293073, "Actor parameter count mismatch")
    require(all(validation["freeze_audit"].values()), "Frozen model gate failed")
    require(validation["replay_max_error"] < 1e-5 and validation["grad_enabled_replay_max_error"] < 1e-5,
            "Historical probability replay threshold failed")
    require(contract["provenance"] == verified_contract["provenance"], "Runtime/verification model provenance differs")
    for source, expected in verified_contract["source_hashes"].items():
        relative = "Experiment/" + source.split("/Experiment/", 1)[1]
        require(digest(root / relative) == expected, f"Runtime source hash mismatch: {relative}")
    require(digest(root / "Experiment/SCOPE/Stage2/Task21/artifacts/release/adapter.pt") == contract["provenance"]["adapter_sha256"],
            "Frozen Adapter changed")
    require(len(summary) == len(validation["execution"]) == 30, "Transition denominator mismatch")
    compressed = list((artifacts / "final_runtime/traces").glob("*.pt.gz"))
    require(len(compressed) == 30 and len(list((artifacts / "final_runtime").glob("episode_boundary_*.pt"))) == 4,
            "Trace or episode boundary archive missing")
    require({record["trace"] for record in validation["execution"]} == {record["trace_path"] for record in summary},
            "Validation/summary trace identifiers differ")
    require(len({record["transition_id"] for record in summary}) == 30, "Duplicate transitions")
    for record in summary:
        require((root / (record["trace_path"] + ".gz")).is_file() and record["current_visual_tokens"] == 64,
                "Missing trace or variable token budget")
    for phase, steps in (("final_sampling", 14), ("final_deterministic", 16)):
        results = read(artifacts / phase / "results.json")
        require(len(results) == 2 and sum(record["steps"] for record in results) == steps,
                "Navigation denominator mismatch")
        require(all(record["failure"] is None and not record["training"]["updated"] for record in results),
                "Unexpected navigation failure or training update")
    require(len(validation["no_focus"]) == 6 and len(validation["interventions"]) == 10 and len(validation["reproducibility"]) == 4,
            "Controlled validation denominator mismatch")
    for record in validation["no_focus"]:
        require(record["token_max_abs"] == record["action_max_abs"] == 0 and record["grids_equal"] and record["stop_equal"],
                "No-Focus equivalence failed")
    for record in validation["interventions"]:
        require(record["current_tokens"] == 64, "Intervention token budget mismatch")
        if record["focus_count"]:
            require(record["feature_max_abs_from_no_focus"] > 0 and record["action_max_abs_from_no_focus"] > 0,
                    "Focus does not affect features/actions")
    return {"archive_integrity": archive["integrity"], "transitions": 30, "episodes": 4,
            "actor_parameters": 2293073, "optimizer_updates": 0, "no_focus_checks": 6, "status": "PASS"}


def extract_traces(root=ROOT):
    paths = list((root / "Experiment/SCOPE/Stage2/Task22/artifacts/final_runtime/traces").glob("*.pt.gz"))
    if any(path.with_suffix("").exists() for path in paths):
        raise FileExistsError("Refusing to overwrite existing decompressed traces")
    for path in paths:
        target = path.with_suffix("")
        temporary = target.with_suffix(".extracting")
        try:
            with gzip.open(path, "rb") as source, temporary.open("xb") as output:
                shutil.copyfileobj(source, output)
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Task22 archive without GPU or model loading")
    parser.add_argument("--extract-traces", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_task22(), indent=2))
    if args.extract_traces:
        extract_traces()
