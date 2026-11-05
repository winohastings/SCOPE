from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_verification", ROOT / "tools/verify_release.py")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
DEPENDENCY_SPEC = importlib.util.spec_from_file_location("release_dependencies", ROOT / "tools/prepare_dependencies.py")
DEPENDENCIES = importlib.util.module_from_spec(DEPENDENCY_SPEC)
DEPENDENCY_SPEC.loader.exec_module(DEPENDENCIES)


class ReleaseArchiveTests(unittest.TestCase):
    def test_dependency_patch_ignores_parent_git_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "sample.txt").write_text("parent\n")
            nested = root / "nested"
            nested.mkdir()
            (nested / "sample.txt").write_text("before\n")
            patch = root / "change.patch"
            patch.write_text("--- a/sample.txt\n+++ b/sample.txt\n@@ -1 +1 @@\n-before\n+after\n")
            DEPENDENCIES.apply_dependency_patch(nested, patch)
            self.assertEqual((nested / "sample.txt").read_text(), "after\n")
            self.assertEqual((root / "sample.txt").read_text(), "parent\n")

    def test_actual_archive_and_denominators(self):
        result = VERIFIER.verify(ROOT)
        self.assertEqual(result["integrity"], "PASS")
        self.assertEqual(result["episodes"], dict.fromkeys(("G0", "G1", "G2", "G3"), 64))
        self.assertEqual(sum(result["predictions"].values()), 3768)

    def test_tampered_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/release").mkdir(parents=True)
            (root / "evidence.txt").write_text("changed")
            manifest = [{"path": "evidence.txt", "sha256": hashlib.sha256(b"original").hexdigest()}]
            (root / "docs/release/FILE_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Archive integrity failure"):
                VERIFIER.verify(root)

    def test_private_key_content_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/release").mkdir(parents=True)
            evidence = root / "evidence.txt"
            evidence.write_text("-----BEGIN " + "OPENSSH PRIVATE KEY-----")
            manifest = [{"path": "evidence.txt", "sha256": VERIFIER.digest(evidence)}]
            (root / "docs/release/FILE_MANIFEST.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Potential secret"):
                VERIFIER.verify(root)


if __name__ == "__main__":
    unittest.main()
