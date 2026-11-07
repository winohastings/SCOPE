from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_task233_release as verifier


class Task233ArchiveTests(unittest.TestCase):
    def test_actual_evidence(self):
        self.assertEqual(verifier.verify_evidence()["Gate1"], "INCONCLUSIVE")

    def test_wrong_anchor_count(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "ANCHOR_MANIFEST_256.json":
                data["anchors"] = data["anchors"][:-1]
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Anchor denominator"):
                verifier.verify_evidence()

    def test_unsupported_gate_pass(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "PROGRESS_GATE.json":
                data["Gate1"] = "PASS"
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Unsupported downstream"):
                verifier.verify_evidence()

    def test_wrong_heldout_metric(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "OFFLINE_AUDIT.json":
                data["after"]["teacher_heldout_confident"]["weighted_nll"] = 0.1
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Supervision metric"):
                verifier.verify_evidence()

    def test_checkpoint_upload_rejected(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name.endswith(".placeholder.json"):
                data["uploaded"] = True
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
                verifier.verify_evidence()


if __name__ == "__main__":
    unittest.main()
