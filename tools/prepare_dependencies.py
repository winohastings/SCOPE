from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_dependency_patch(directory, patch):
    command = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(directory), "-i", str(patch)]
    subprocess.run(command + ["--dry-run"], check=True)
    subprocess.run(command, check=True)


def prepare(legacy_source, simplenav_source, output):
    lock = json.loads((ROOT / "third_party/dependencies.lock.json").read_text())
    runtime = ROOT / "Experiment/SCOPE/legacy_runtime"
    destination = output / "SimpleNav"
    if runtime.exists() or destination.exists():
        raise FileExistsError("Refusing to overwrite an existing reconstructed dependency")
    output.mkdir(parents=True, exist_ok=True)
    for record in lock["legacy_files"]:
        if record["base_sha256"] is not None:
            source = legacy_source / record["path"]
            if sha256(source) != record["base_sha256"]:
                raise ValueError(f"Legacy base hash mismatch: {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(simplenav_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != lock["simplenav_revision"]:
        raise ValueError(f"SimpleNav revision mismatch: {revision}")
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        staged_runtime = staging / "legacy_runtime"
        staged_runtime.mkdir()
        for record in lock["legacy_files"]:
            if record["base_sha256"] is not None:
                target = staged_runtime / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_source / record["path"], target)
        apply_dependency_patch(staged_runtime, ROOT / "patches/legacy_scope.patch")
        for record in lock["legacy_files"]:
            if sha256(staged_runtime / record["path"]) != record["scope_sha256"]:
                raise ValueError(f"Reconstructed runtime mismatch: {record['path']}")
        staged_simplenav = staging / "SimpleNav"
        staged_simplenav.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(simplenav_source), "archive", lock["simplenav_revision"]]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(staged_simplenav, filter="data")
        apply_dependency_patch(staged_simplenav, ROOT / "patches/simplenav_stage1.patch")
        for relative, digest in lock["simplenav_files"].items():
            if sha256(staged_simplenav / relative) != digest:
                raise ValueError(f"SimpleNav patched hash mismatch: {relative}")
        runtime.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_runtime), runtime)
        shutil.move(str(staged_simplenav), destination)
    print(json.dumps({"legacy_runtime": str(runtime), "simplenav": str(destination),
                      "hashes_verified": True}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Reconstruct ignored official dependencies from pinned local sources")
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--simplenav-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".dependencies")
    args = parser.parse_args()
    prepare(args.legacy_source.resolve(), args.simplenav_source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
