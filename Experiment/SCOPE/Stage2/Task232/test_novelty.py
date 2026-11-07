import copy
import unittest

import torch

from Experiment.SCOPE.Stage2.Task232.novelty import augment_transitions, coverage_novelty
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, gae
from Experiment.SCOPE.Stage2.Task232.report_c import exploration_gate


def focus(camera=0, left=0):
    return {"action": {"stop": False, "camera": camera}, "roi": (left, 0, left + 192, 192)}


class NoveltyTests(unittest.TestCase):
    def test_first_and_repeated(self):
        result = coverage_novelty([focus()] * 4, .01)
        self.assertEqual([row["novelty"] for row in result["focus"]], [1., 0., 0., 0.])
        self.assertEqual(result["reward"], .01)

    def test_different_cameras(self):
        self.assertEqual(coverage_novelty([focus(camera) for camera in range(4)], .01)["reward"], .04)

    def test_partial_overlap(self):
        result = coverage_novelty([focus(), focus(left=96), focus(left=192)], .01)
        self.assertEqual([row["novelty"] for row in result["focus"]], [1., .5, .5])

    def test_stop_zero_and_reset(self):
        self.assertEqual(coverage_novelty([{"action": {"stop": True}}], .01)["reward"], 0.)
        self.assertEqual(coverage_novelty([focus()], .01), coverage_novelty([focus()], .01))
        self.assertEqual(coverage_novelty([focus()], 0.)["reward"], 0.)

    def test_invalid_input(self):
        for beta in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                coverage_novelty([], beta)
        with self.assertRaises(ValueError):
            coverage_novelty([focus(camera=4)], .01)

    def test_reward_reaches_gae_and_restore(self):
        trace = {"decisions": [focus()] * 4, "navigation_reward": -.04, "terminated": True, "truncated": False}
        raw = NavigationTransition(trace, 0., 0., -.04, "episode")
        augmented = augment_transitions([copy.deepcopy(raw)], .01)
        self.assertAlmostEqual(augmented[0].reward, -.03)
        self.assertEqual(augmented[0].trace["navigation_reward"], augmented[0].reward)
        original_adv, _ = gae([raw])
        shaped_adv, _ = gae(augmented)
        self.assertAlmostEqual(float(shaped_adv[0] - original_adv[0]), .01, places=6)
        restored = copy.deepcopy(augmented[0])
        restored.reward = -.04
        restored.trace["navigation_reward"] = -.04
        self.assertEqual(augment_transitions([restored], .01)[0].reward, augmented[0].reward)

    def test_zero_beta_preserves_targets(self):
        trace = {"decisions": [focus()], "navigation_reward": -.02, "terminated": True, "truncated": False}
        original = NavigationTransition(trace, .1, 0., -.02, "episode")
        modified = augment_transitions([copy.deepcopy(original)], 0.)
        self.assertEqual(original.reward, modified[0].reward)
        for expected, actual in zip(gae([original]), gae(modified), strict=True):
            self.assertTrue(expected.equal(actual))

    def test_exploration_separate_from_navigation(self):
        reference = {"four_focus_coverage": .04, "four_focus_repeat_fraction": 1., "SR": .8}
        candidate = {"four_focus_coverage": .06, "four_focus_repeat_fraction": .5, "SR": .1}
        self.assertTrue(all(exploration_gate(reference, candidate).values()))
        candidate.update(four_focus_coverage=None, four_focus_repeat_fraction=None)
        self.assertFalse(any(exploration_gate(reference, candidate).values()))

    def test_camera_bias_gradient_decomposition(self):
        logits = torch.tensor([[.2, .7, -.4, .1], [.1, .3, .2, -.2]], dtype=torch.float64)
        bias = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([1, 2])
        base = torch.tensor([.4, -.6], dtype=torch.float64)
        bonus = torch.tensor([-.1, .1], dtype=torch.float64)
        distribution = torch.distributions.Categorical(logits=logits + bias)
        loss = -((base + bonus) * distribution.log_prob(actions)).mean()
        actual = torch.autograd.grad(loss, bias)[0]
        score = torch.nn.functional.one_hot(actions, 4) - logits.softmax(-1)
        expected = -(base[:, None] * score).mean(0) - (bonus[:, None] * score).mean(0)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=0.))


if __name__ == "__main__":
    unittest.main()
