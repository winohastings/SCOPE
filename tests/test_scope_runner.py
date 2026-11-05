import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/run_smoke.py"
SPEC = importlib.util.spec_from_file_location("scope_runner", MODULE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ScopeRunnerTest(unittest.TestCase):
    def write_task(self, root, task):
        for mode in RUNNER.MODES:
            run = root / "runs" / f"{task}_{mode}"
            run.mkdir(parents=True)
            (run / "summary.json").write_text("{}")
            for episode_id in ("1", "2"):
                episode = run / episode_id
                episode.mkdir()
                result = {"source_episode_id": episode_id, "scene_id": "scene", "steps": 1,
                          "failure": None, "metrics": dict.fromkeys(
                              ("SR", "OSR", "NE", "SPL", "nDTW", "path_length", "steps_taken"), 1)}
                (episode / "eval_info.json").write_text(json.dumps(result))
                focus = 0 if mode in ("original", "glance_only") else 1 if mode == "learned_adaptive" else 2
                record = dict.fromkeys(("total_visual_tokens", "vision_ms", "llm_ms", "dit_b_ms",
                                        "server_total_ms", "habitat_render_ms", "total_ms", "peak_gpu_memory_bytes"), 1)
                record.update(selector="fixture", focus_count=focus,
                              current_visual_tokens=256 if mode == "original" else 64 + 64 * focus)
                (episode / "scope_00000.json").write_text(json.dumps(record))

    def test_r2r_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "r2r")
            self.assertEqual(RUNNER.aggregate(root, tasks=("r2r",)), 6)
            verification = json.loads((root / "smoke_verification.json").read_text())
            self.assertEqual({item["task"] for item in verification}, {"r2r"})

    def test_default_keeps_both_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("r2r", "rxr"):
                self.write_task(root, task)
            self.assertEqual(RUNNER.aggregate(root), 12)

    def test_mismatched_episodes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "r2r")
            path = root / "runs/r2r_fixed_2/1/eval_info.json"
            result = json.loads(path.read_text())
            result["source_episode_id"] = "different"
            path.write_text(json.dumps(result))
            with self.assertRaises(AssertionError):
                RUNNER.aggregate(root, tasks=("r2r",))


if __name__ == "__main__":
    unittest.main()
