import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/legacy_runtime/starVLA/model/modules/input_context.py"
SPEC = importlib.util.spec_from_file_location("scope_context", MODULE)
CONTEXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTEXT)


class ScopeSamplingTest(unittest.TestCase):
    def setUp(self):
        self.images = {camera: np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
                       for camera in CONTEXT.CAMERAS}
        self.raw, self.compatible, self.glance, self.candidates = CONTEXT.prepare_scope_images(self.images)

    def choose(self, mode, **kwargs):
        return CONTEXT.select_focus(mode, seed=42, episode_id="1", step=0,
                                    glance=self.glance, candidates=self.candidates, **kwargs)

    def test_raw_and_exact_crops(self):
        self.assertEqual(len(self.candidates), 16)
        for candidate in self.candidates:
            left, top, right, bottom = candidate["roi"]
            np.testing.assert_array_equal(candidate["image"], self.images[candidate["camera_name"]][top:bottom, left:right])
        self.assertEqual(self.glance["front"].size, (128, 128))
        self.assertEqual(self.compatible["front"].size, (256, 256))
        np.testing.assert_array_equal(self.raw["front"], self.images["front"])

    def test_reject_scaled_observations(self):
        with self.assertRaises(ValueError):
            CONTEXT.prepare_scope_images({camera: np.zeros((256, 256, 3), dtype=np.uint8) for camera in CONTEXT.CAMERAS})

    def test_fixed_and_random(self):
        self.assertEqual(self.choose("fixed_2")[1], [0, 3])
        self.assertEqual(self.choose("random_2")[1], self.choose("random_2")[1])
        self.assertEqual(len(set(self.choose("random_2")[1])), 2)

    def test_sequential_stop_and_invalid(self):
        calls = []
        def selector(**kwargs):
            calls.append(len(kwargs["selected"]))
            return 0 if not kwargs["selected"] else CONTEXT.STOP
        self.assertEqual(len(self.choose("learned_adaptive", selector=selector)[0]), 1)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(len(self.choose("learned_adaptive", selector=lambda **kwargs: 16)[0]), 0)
        self.assertEqual(len(self.choose("learned_adaptive", selector=lambda **kwargs: kwargs["available"][0])[0]), 2)
        with self.assertRaises(ValueError):
            self.choose("learned_2")
        with self.assertRaises(ValueError):
            self.choose("learned_2", selector=lambda **kwargs: 16)
        with self.assertRaises(ValueError):
            self.choose("learned_2", selector=lambda **kwargs: 0)


if __name__ == "__main__":
    unittest.main()
