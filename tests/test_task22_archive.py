import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_task22_release as release


class Task22ArchiveTests(unittest.TestCase):
    def test_actual_archive(self):
        self.assertEqual(release.verify_task22(ROOT)["status"], "PASS")

    def test_changed_transition_denominator_rejected(self):
        original = release.read

        def altered_read(path):
            data = original(path)
            if path.name == "trace_summary.json":
                data.pop()
            return data

        with patch.object(release, "read", side_effect=altered_read):
            with self.assertRaisesRegex(ValueError, "Transition denominator"):
                release.verify_task22(ROOT)


if __name__ == "__main__":
    unittest.main()
