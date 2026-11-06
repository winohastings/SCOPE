import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_task21_release as release


class Task21ArchiveTests(unittest.TestCase):
    def test_actual_archive(self):
        self.assertEqual(release.verify_task21(ROOT)["status"], "PASS")

    def test_changed_checkpoint_hash_rejected(self):
        with patch.object(release, "digest", return_value="invalid"):
            with self.assertRaisesRegex(ValueError, "Adapter hash mismatch"):
                release.verify_task21(ROOT)

    def test_incorrect_current_token_budget_rejected(self):
        original = release.read

        def altered_read(path):
            data = original(path)
            if path.name == "validation.json":
                data["timing"][1]["navigation_current_tokens"] = 128
            return data

        with patch.object(release, "read", side_effect=altered_read):
            with self.assertRaisesRegex(ValueError, "Fixed current token contract"):
                release.verify_task21(ROOT)


if __name__ == "__main__":
    unittest.main()
