import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("stage2_release", ROOT / "tools/verify_stage2_release.py")
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class Stage2ArchiveTests(unittest.TestCase):
    def test_actual_stage2_evidence(self):
        self.assertEqual(RELEASE.verify_stage2(ROOT)["status"], "PASS")

    def test_incorrect_training_denominator_rejected(self):
        original = RELEASE.read

        def altered_read(path):
            data = original(path)
            if path.name == "FINAL_VALIDATION.json":
                data["primary_real_training_transitions"] = 25
            return data

        with patch.object(RELEASE, "read", side_effect=altered_read):
            with self.assertRaisesRegex(ValueError, "Primary training denominator"):
                RELEASE.verify_stage2(ROOT)


if __name__ == "__main__":
    unittest.main()
