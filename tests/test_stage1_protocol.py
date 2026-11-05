import importlib.util
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/Stage1/validate_protocol.py"
SPEC = importlib.util.spec_from_file_location("stage1_protocol", MODULE)
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


class Stage1ProtocolTest(unittest.TestCase):
    def test_cyclic_coverage_and_repeat(self):
        pairs = PROTOCOL.validate_cycle([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
        self.assertEqual(len(pairs), 8)
        self.assertEqual(sorted(candidate for pair in pairs for candidate in pair), list(range(16)))

    def test_duplicate_roi_rejected(self):
        with self.assertRaises(ValueError):
            PROTOCOL.validate_cycle([0] * 16)

    def test_same_camera_pair_rejected(self):
        with self.assertRaises(ValueError):
            PROTOCOL.validate_cycle(list(range(16)))

    def test_bundle_remains_unapproved(self):
        report = PROTOCOL.validate_bundle()
        self.assertFalse(report["training_started"])
        self.assertFalse(report["full_evaluation_started"])
        self.assertEqual(report["status"], "DRAFT_VALIDATED_NOT_APPROVED")
        self.assertEqual(report["tasks"]["r2r"]["episodes"], 1839)
        self.assertEqual(report["tasks"]["rxr"]["episodes"], 3669)


if __name__ == "__main__":
    unittest.main()
