from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_task231_release as verifier


class Task231ArchiveTests(unittest.TestCase):
    def test_actual_evidence(self):
        self.assertEqual(verifier.verify_evidence()["counterfactual_episodes"], 192)

    def test_incomplete_shard_rejected(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "counterfactual_results.json":
                return data[:-1]
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "denominator mismatch"):
                verifier.verify_evidence()

    def test_invalid_restoration_rejected(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "matched_results.json":
                data[0]["restoration_valid"] = False
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "restoration failure"):
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
