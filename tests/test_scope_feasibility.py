from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/Stage1_Feasibility"
sys.path.insert(0, str(ROOT))
from policies import CYCLE, MODES, TOKENS, select_pair


class FeasibilityTests(unittest.TestCase):
    def test_cycle_coverage_and_reset(self):
        first = [select_pair("fixed_2", seed=42, episode_id="one", step=step)[0] for step in range(8)]
        self.assertEqual(sorted(candidate for pair in first for candidate in pair), list(range(16)))
        self.assertEqual(first, [list(pair) for pair in CYCLE])
        self.assertEqual(select_pair("fixed_2", seed=7, episode_id="two", step=0)[0], first[0])
        self.assertEqual(select_pair("fixed_2", seed=7, episode_id="two", step=8)[0], first[0])

    def test_random_repeatability_and_camera_constraint(self):
        for step in range(1000):
            pair, digest = select_pair("random_2", seed=42, episode_id="9791", step=step)
            self.assertEqual((pair, digest), select_pair("random_2", seed=42, episode_id="9791", step=step))
            self.assertEqual(len(set(pair)), 2)
            self.assertNotEqual(pair[0] // 4, pair[1] // 4)
        self.assertNotEqual(select_pair("random_2", seed=42, episode_id="9791", step=1)[1],
                            select_pair("random_2", seed=43, episode_id="9791", step=1)[1])

    def test_budgets_and_exclusions(self):
        self.assertEqual(TOKENS, {"G0": 4 * 64, "G1": 4 * 16, "G2": 4 * 16 + 2 * 64, "G3": 4 * 16 + 2 * 64})
        for mode in ("learned_2", "learned_adaptive", "unknown"):
            with self.assertRaises(ValueError):
                select_pair(mode, seed=42, episode_id="episode", step=0)

    def test_materialized_schema_and_history_contract(self):
        import json
        import numpy as np
        import sys
        from compile_configs import SIMPLENAV
        sys.path.insert(0, str(SIMPLENAV))
        from benchmark.vlnce.inputs import _reference_path_length
        episodes = json.loads((ROOT / "DEV_EPISODES.json").read_text())
        self.assertEqual(len(episodes), 64)
        for episode in episodes:
            self.assertEqual(episode["payload"]["gt_path_length"],
                             _reference_path_length(episode["payload"].get("reference_path", [])))
        sys.path.insert(0, str(ROOT.parent))
        from scope_server import ScopePolicy
        generator = np.random.default_rng(42)
        images = {camera: generator.integers(0, 256, (512, 512, 3), dtype=np.uint8)
                  for camera in ("front", "left", "right", "rear")}
        reference = None
        for mode in MODES.values():
            observation = ScopePolicy._observation({"images": images, "body_pose": np.zeros(4),
                                                    "frame_id": 0, "timestamp_s": 0.0, "policy_mode": mode}, 0)
            self.assertTrue(all(image.shape == (256, 256, 3) for image in observation["images"].values()))
            if reference is not None:
                for camera in images:
                    np.testing.assert_array_equal(reference["images"][camera], observation["images"][camera])
            reference = observation

    def test_flops_shape_estimate_not_token_ratio(self):
        from aggregate import vision_flops
        config = {"hidden_size": 1024, "spatial_merge_size": 2, "in_channels": 3,
                  "temporal_patch_size": 2, "patch_size": 16, "depth": 24,
                  "intermediate_size": 4096, "out_hidden_size": 2560}
        glance = vision_flops([1, 8, 8], config)
        full = vision_flops([1, 16, 16], config)
        self.assertGreater(full, 4 * glance)
        self.assertGreater(glance, 0)


if __name__ == "__main__":
    unittest.main()
