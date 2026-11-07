from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Experiment/SCOPE/Stage1_Feasibility"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def historical_source(root, relative, expected):
    current = root / relative
    if current.is_file() and digest(current) == expected:
        return current
    mapping = root / "docs/release/HISTORICAL_SOURCE_MAP.json"
    for record in json.loads(mapping.read_text()) if mapping.exists() else []:
        if record["source"] == relative and record["sha256"] == expected:
            snapshot = root / record["snapshot"]
            if snapshot.is_file() and not snapshot.is_symlink() and digest(snapshot) == expected:
                return snapshot
    raise ValueError(f"Historical source hash mismatch: {relative}")


def verify(root=ROOT):
    manifest = json.loads((root / "docs/release/FILE_MANIFEST.json").read_text())
    for record in manifest:
        path = root / record["path"]
        if path.is_symlink() or not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Archive integrity failure: {record['path']}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError(f"Oversized Git object: {record['path']}")
    forbidden = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)")
    for record in manifest:
        path = root / record["path"]
        if path.suffix in {".py", ".md", ".json", ".yaml", ".log", ".patch", ".txt"}:
            if forbidden.search(path.read_text(errors="replace")):
                raise ValueError(f"Potential secret found: {path}")
    source_map = json.loads((root / "docs/release/SOURCE_MAP.json").read_text())
    for record in source_map:
        path = root / record["path"]
        opener = gzip.open if record["encoding"] == "gzip" else open
        with opener(path, "rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != record["source_sha256"]:
            raise ValueError(f"Original-source content mismatch: {path}")
    results = root / "Experiment/SCOPE/Stage1_Feasibility"
    with (results / "RESULTS_PER_EPISODE.csv").open() as stream:
        episodes = list(csv.DictReader(stream))
    expected = {str(record["source_episode_id"]) for record in json.loads((results / "DEV_EPISODES.json").read_text())}
    counts = collections.Counter(record["group"] for record in episodes)
    if counts != {group: 64 for group in ("G0", "G1", "G2", "G3")}:
        raise ValueError(f"Unexpected denominators: {counts}")
    for group in counts:
        subset = [record for record in episodes if record["group"] == group]
        if {record["episode_id"] for record in subset} != expected:
            raise ValueError(f"Episode ID mismatch: {group}")
        if any(record["failure"] not in ("", "None", "null") for record in subset):
            raise ValueError(f"Recorded execution failure: {group}")
    budget = {"G0": 256, "G1": 64, "G2": 192, "G3": 192}
    step_counts = collections.Counter()
    with gzip.open(results / "RESULTS_PER_STEP.csv.gz", "rt") as stream:
        for record in csv.DictReader(stream):
            if int(record["current_visual_tokens"]) != budget[record["group"]]:
                raise ValueError("Current visual token budget mismatch")
            step_counts[record["group"]] += 1
    for group in counts:
        predictions = sum(int(record["prediction_count"]) for record in episodes if record["group"] == group)
        if step_counts[group] != predictions:
            raise ValueError(f"Prediction denominator mismatch: {group}")
    return {"files_verified": len(manifest), "episodes": dict(counts),
            "predictions": dict(step_counts), "integrity": "PASS"}


def extract_results():
    for path in RESULTS.rglob("*.gz"):
        target = path.with_suffix("")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        temporary = target.with_name(target.name + ".extracting")
        try:
            with gzip.open(path, "rb") as source, temporary.open("xb") as stream:
                shutil.copyfileobj(source, stream)
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Verify curated archive integrity and Stage1 evidence without GPU")
    parser.add_argument("--extract-results", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(), indent=2))
    if args.extract_results:
        extract_results()


if __name__ == "__main__":
    main()
