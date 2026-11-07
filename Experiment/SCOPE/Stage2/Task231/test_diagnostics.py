from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Experiment.SCOPE.Stage2.Task231.diagnose import interval, paired
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, gae, navigation_reward


class DiagnosticTests(unittest.TestCase):
    def test_paired_exact_and_identity(self):
        first = [{"episode_uid": str(index), "SR": 1, "SPL": 1., "nDTW": 1.} for index in range(8)]
        second = [row | {"SR": 0, "SPL": 0., "nDTW": 0.} for row in first]
        result = paired(first, second)
        self.assertEqual(result["success_to_failure"], 8)
        self.assertEqual(result["exact_mcnemar_p"], 2 / 256)
        self.assertEqual(result["delta_SR"]["ci95"], [-1., -1.])
        self.assertEqual(paired(first, first)["exact_mcnemar_p"], 1.)

    def test_pairing_rejects_wrong_order(self):
        with self.assertRaises(AssertionError):
            paired([{"episode_uid": "a"}], [{"episode_uid": "b"}])

    def test_bootstrap_reproducible(self):
        self.assertEqual(interval([0, 1, 0]), interval([0, 1, 0]))

    def test_focus_cost_counterfactual(self):
        teacher = {"distance_before": 5., "distance_after": 4.7, "terminal": False, "success": False}
        no_focus = navigation_reward(teacher, 0, terminated=False, truncated=False)
        four = navigation_reward(teacher, 4, terminated=False, truncated=False)
        self.assertAlmostEqual(no_focus - four, .04)
        self.assertLess(four, 0)

    def test_gae_terminal_and_truncation(self):
        terminal = NavigationTransition({"terminated": True, "truncated": False}, .3, 100., 2., "a")
        truncated = NavigationTransition({"terminated": False, "truncated": True}, .2, .7, -.1, "b")
        advantages, returns = gae([terminal, truncated])
        torch.testing.assert_close(advantages, torch.tensor([1.7, -.1 + .99 * .7 - .2]))
        torch.testing.assert_close(returns, torch.tensor([2., -.1 + .99 * .7]))


if __name__ == "__main__":
    unittest.main()
