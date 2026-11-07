from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_release
import verify_task23_release


class Task23ArchiveTests(unittest.TestCase):
    def test_evidence(self):
        self.assertEqual(verify_task23_release.verify_evidence()["status"], "PASS")

    def test_wrong_training_denominator(self):
        original = verify_task23_release.read

        def changed(path):
            record = original(path)
            if path.name == "COMPLETE.json":
                record["trained"] = 6548
            return record

        with patch.object(verify_task23_release, "read", changed):
            with self.assertRaisesRegex(ValueError, "Training denominator"):
                verify_task23_release.verify_evidence()

    def test_uploaded_checkpoint_rejected(self):
        original = verify_task23_release.read

        def changed(path):
            record = original(path)
            if path.name.endswith(".placeholder.json"):
                record["uploaded"] = True
            return record

        with patch.object(verify_task23_release, "read", changed):
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint placeholder"):
                verify_task23_release.verify_evidence()

    def test_historical_snapshot_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs/release").mkdir(parents=True)
            (root / "current.py").write_text("new")
            snapshot = root / "old.py"
            snapshot.write_text("old")
            expected = verify_release.digest(snapshot)
            mapping = [{"source": "current.py", "snapshot": "old.py", "sha256": expected}]
            (root / "docs/release/HISTORICAL_SOURCE_MAP.json").write_text(json.dumps(mapping))
            self.assertEqual(verify_release.historical_source(root, "current.py", expected), snapshot)
            snapshot.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Historical source hash mismatch"):
                verify_release.historical_source(root, "current.py", expected)


if __name__ == "__main__":
    unittest.main()
