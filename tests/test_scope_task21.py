import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/Stage2/Task21/fusion.py"
SPEC = importlib.util.spec_from_file_location("task21_fusion", PATH)
FUSION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FUSION
SPEC.loader.exec_module(FUSION)


class Task21Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.glance = torch.randn(4, 16, 2560)
        self.adapter = FUSION.SpatialFusion()

    def test_no_focus_identity(self):
        self.assertIs(self.adapter(self.glance), self.glance)

    def test_boundary_and_rounding(self):
        self.assertEqual(FUSION.Focus(0, 0, 0).roi, (0, 0, 192, 192))
        self.assertEqual(FUSION.Focus(3, 1, 1).roi, (320, 320, 512, 512))
        self.assertEqual(FUSION.Focus(0).roi, (160, 160, 352, 352))
        self.assertEqual(FUSION.Focus(0, 96.5 / 512, 96.5 / 512).roi[:2], (1, 1))

    def test_invalid_requests(self):
        for camera, horizontal, vertical in ((4, .5, .5), (-1, .5, .5), (True, .5, .5), (0, float("nan"), .5), (0, 1.1, 0)):
            with self.assertRaises(ValueError):
                FUSION.Focus(camera, horizontal, vertical)

    def test_camera_crops(self):
        images = {name: np.full((512, 512, 3), camera * 50, dtype=np.uint8) for camera, name in enumerate(FUSION.CAMERAS)}
        for camera in range(4):
            crop = np.asarray(FUSION.Focus(camera).crop(images))
            self.assertEqual(crop.shape, (256, 256, 3))
            self.assertTrue((crop == camera * 50).all())

    def test_coverage_area_and_constant(self):
        for position in ((0, 0), (1, 1), (.5, .5), (.37, .63)):
            focus = FUSION.Focus(0, *position)
            aligned, coverage = FUSION.align_area(torch.ones(64, 3), focus.roi, focus.roi)
            self.assertAlmostEqual(float(coverage.sum()), 192 ** 2 / 128 ** 2)
            torch.testing.assert_close(aligned[coverage[:, 0] > 0], torch.ones_like(aligned[coverage[:, 0] > 0]))

    def test_row_column_orientation(self):
        tokens = torch.arange(64, dtype=torch.float32).reshape(64, 1)
        focus = FUSION.Focus(0)
        aligned, coverage = FUSION.align_area(tokens, focus.roi, focus.roi)
        torch.testing.assert_close(aligned[[5, 6, 9, 10], 0], torch.tensor([13.5, 17.5, 45.5, 49.5]))
        self.assertEqual(int((coverage > 0).sum()), 4)

    def test_only_selected_camera_and_coverage(self):
        for camera in range(4):
            focus = FUSION.Focus(camera, .37, .63)
            tokens = torch.randn(64, 2560)
            result = self.adapter(self.glance, [(focus, tokens)])
            _, coverage = FUSION.align_area(tokens, focus.roi, focus.roi)
            for other in range(4):
                if other != camera:
                    self.assertTrue(torch.equal(result[other], self.glance[other]))
            outside = coverage[:, 0] == 0
            self.assertTrue(torch.equal(result[camera][outside], self.glance[camera][outside]))
            self.assertFalse(torch.equal(result[camera], self.glance[camera]))

    def test_repeat_accumulates_without_tokens(self):
        update = (FUSION.Focus(0), torch.ones(64, 2560))
        once = self.adapter(self.glance, [update])
        twice = self.adapter(self.glance, [update, update])
        torch.testing.assert_close(twice, self.adapter(once, [update]))
        self.assertEqual(twice.shape, (4, 16, 2560))
        self.assertFalse(torch.equal(once, twice))
        with self.assertRaises(ValueError):
            self.adapter(self.glance, [update] * 5)

    def test_teacher_only_visible_cells(self):
        focus = FUSION.Focus(2)
        teacher = torch.randn(256, 2560)
        target, mask = FUSION.calibration_target(self.glance, teacher, focus)
        outside = ~(mask > 0).expand_as(target)
        self.assertTrue(torch.equal(target[outside], self.glance[outside]))
        modified = teacher.reshape(16, 16, 2560).clone()
        modified[:5] += 100
        other, _ = FUSION.calibration_target(self.glance, modified.reshape(256, 2560), focus)
        self.assertTrue(torch.equal(target, other))

    def test_freeze_and_gradient(self):
        output = self.adapter(self.glance, [(FUSION.Focus(0), torch.ones(64, 2560))])
        output.square().mean().backward()
        self.assertIsNotNone(self.adapter.gate_logit.grad)
        self.assertIsNone(self.glance.grad)
        self.adapter.freeze()
        self.assertFalse(self.adapter.gate_logit.requires_grad)

    def test_bfloat16_preservation(self):
        glance = self.glance.bfloat16()
        output = self.adapter(glance, [(FUSION.Focus(0), torch.ones(64, 2560).bfloat16())])
        self.assertEqual(output.dtype, glance.dtype)
        self.assertTrue(torch.equal(output[1:], glance[1:]))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            self.adapter(torch.zeros(64, 2560))
        with self.assertRaises(ValueError):
            self.adapter(self.glance, [(FUSION.Focus(0), torch.zeros(16, 2560))])

    def test_runtime_rejects_uncalibrated_focus(self):
        from Experiment.SCOPE.Stage2.Task21.runtime import Focus, FusionPolicy
        from Experiment.SCOPE.Stage2.Task21.fusion import Focus as PublicFocus
        self.assertIs(Focus, PublicFocus)
        policy = object.__new__(FusionPolicy)
        policy.calibrated = False
        with self.assertRaises(RuntimeError):
            policy.predict_session({}, frames=[], instruction="test", focus=[Focus(0)])

    def test_runtime_rejects_appended_focus_mode(self):
        from Experiment.SCOPE.Stage2.Task21.runtime import FusionPolicy
        policy = object.__new__(FusionPolicy)
        policy.calibrated = True
        with self.assertRaises(ValueError):
            policy.predict_session({}, frames=[{"policy_mode": "fixed_2"}], instruction="test")


if __name__ == "__main__":
    unittest.main()
