from __future__ import annotations

import csv
import json

from aggregate import aggregate
from compile_configs import PROJECT, ROOT, SIMPLENAV, sha256, validate_freeze, write_json


def main():
    manifest = validate_freeze()
    if sha256(manifest["checkpoint"]["path"]) != manifest["checkpoint"]["sha256"]:
        raise ValueError("checkpoint changed during evaluation")
    task2_evidence = json.loads((PROJECT / "docs/Logs/Task2/implementation_evidence.json").read_text())
    protected = task2_evidence["protected_task1_files"]
    for name, expected in protected.items():
        if sha256(PROJECT / name) != expected:
            raise ValueError(f"protected Task1 artifact changed: {name}")
    runtime = json.loads((PROJECT / "docs/Logs/Task1/runtime_provenance.json").read_text())
    for record in runtime:
        if sha256(PROJECT / record["path"]) != record["sha256"]:
            raise ValueError(f"protected runtime changed: {record['path']}")
    dependencies = [SIMPLENAV / path for path in (
        "deployment/model_server/legacy_rxr_server.py", "benchmark/common/runner/worker.py",
        "benchmark/common/simulators/habitat/backend.py", "benchmark/common/simulators/habitat/vlnce031_runtime.py",
        "benchmark/common/simulators/habitat/vlnce031_datasets.py", "benchmark/vlnce/inputs.py",
        "benchmark/vlnce/benchmark.py", "benchmark/common/log/metrics.py", "benchmark/common/config.py")]
    dependencies.append(ROOT.parent / "scope_server.py")
    write_json(ROOT / "DEPENDENCY_HASHES.json", {str(path): sha256(path) for path in dependencies})
    write_json(ROOT / "FINAL_SOURCE_HASHES.json", {str(path.relative_to(PROJECT)): sha256(path)
               for path in list(ROOT.glob("*.py")) + [PROJECT / "tests/test_scope_feasibility.py", PROJECT / "docs/index.md"]})
    write_json(ROOT / "PROTECTED_ARTIFACTS_AUDIT.json", {"passed": True, "task1_artifact_count": len(protected),
               "runtime_file_count": len(runtime), "checkpoint_unchanged": True,
               "task2_status": "original draft not executed or overwritten"})
    supplements = ("ENVIRONMENT_AMENDMENT.md", "COST_PROFILING_AMENDMENT.md", "CONCURRENCY_AMENDMENT.md")
    write_json(ROOT / "EXECUTION_AMENDMENTS.json", {"base_manifest_sha256": sha256(ROOT / "EXECUTION_MANIFEST.json"),
               "amendments": {name: sha256(ROOT / name) for name in supplements},
               "environment_workers": 2, "model_instances": 1, "training_started": False})
    rows = aggregate()
    if not all(row["recorded"] == 64 and row["failures"] == 0 for row in rows):
        raise RuntimeError("incomplete or failed formal evaluation; inspect failure report before completion")
    required = ("STAGE1_FEASIBILITY_PROTOCOL.md", "IMPLEMENTATION_REPORT.md", "EXECUTION_MANIFEST.json",
                "RUN_ENVIRONMENT.json", "RESULTS_PER_EPISODE.csv", "RESULTS_PER_STEP.csv", "RESULTS_AGGREGATED.csv",
                "NAVIGATION_METRICS.md", "COMPUTATIONAL_COST.md", "FAILURE_ANALYSIS.md", "STAGE1_FEASIBILITY_REPORT.md")
    if not all((ROOT / name).is_file() for name in required) or len(list((ROOT / "plots").glob("*.svg"))) < 6:
        raise RuntimeError("missing required deliverables")
    write_json(ROOT / "STAGE1_COMPLETION.json", {"status": "COMPLETE", "groups": 4, "episodes_per_group": 64,
               "total_episodes": 256, "execution_failures": 0, "second_stage_started": False,
               "regression_test_log_sha256": sha256(ROOT / "REGRESSION_FINAL.log"),
               "final_source_hashes_sha256": sha256(ROOT / "FINAL_SOURCE_HASHES.json"),
               "note": "EXECUTION_STATUS tracks process shutdown; this marker additionally requires aggregation and integrity audit"})
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                   and path.name not in ("FILES_MANIFEST.tsv", "FINALIZE.log"))
    files.extend((PROJECT / "tests/test_scope_feasibility.py", PROJECT / "docs/index.md"))
    with (ROOT / "FILES_MANIFEST.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("file", "bytes", "sha256"))
        for path in files:
            writer.writerow((str(path.relative_to(PROJECT)), path.stat().st_size, sha256(path)))
    print("PASS: all four groups ×64 episodes, protected artifacts and checkpoint intact, required deliverables generated")


if __name__ == "__main__":
    main()
