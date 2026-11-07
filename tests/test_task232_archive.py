from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import verify_task232_release as verifier


class Task232ArchiveTests(unittest.TestCase):
    def test_actual_evidence(self):
        self.assertEqual(verifier.verify_evidence()["fresh_training_transitions"], 2112)

    def test_wrong_training_denominator(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "COMPLETE.json" and path.parent.name == "branch_A512":
                data["new_training_transitions"] = 1545
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Training denominator"):
                verifier.verify_evidence()

    def test_missing_evaluation_episode(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "evaluation_independent.json":
                return data[:-1]
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Evaluation denominator"):
                verifier.verify_evidence()

    def test_false_extension_claim(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name == "C_DECISION.json":
                data["Task233_configuration_approved"] = True
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Unsupported C acceptance"):
                verifier.verify_evidence()

    def test_uploaded_checkpoint_rejected(self):
        original = verifier.read

        def changed(path):
            data = original(path)
            if path.name.endswith(".placeholder.json"):
                data["uploaded"] = True
            return data

        with patch.object(verifier, "read", changed):
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint placeholder"):
                verifier.verify_evidence()


if __name__ == "__main__":
    unittest.main()
