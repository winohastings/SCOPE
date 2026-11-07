from __future__ import annotations

import json
import copy
from pathlib import Path
import unittest

import torch

from Experiment.SCOPE.Stage2.Task232.metrics import roi_metrics
from Experiment.SCOPE.Stage2.Task232.runtime import ORIGINAL_REWARD, repair_reward
from Experiment.SCOPE.Stage2.Task23.audit import compare_nested
from Experiment.SCOPE.Stage2.Task232.decision import extension_eligibility

ROOT = Path(__file__).resolve().parent


class RepairTests(unittest.TestCase):
    def test_unchanged_reward(self):
        for count in range(5):
            for terminated in (False, True):
                teacher = {"distance_before": 5., "distance_after": 2., "terminal": terminated, "success": True}
                expected = ORIGINAL_REWARD(teacher, count, terminated=terminated, truncated=False)
                self.assertEqual(repair_reward(teacher, count, terminated=terminated, truncated=False, focus_cost=.01), expected)

    def test_reduced_cost_only(self):
        teacher = {"distance_before": 8., "distance_after": 7., "terminal": False, "success": False}
        original = repair_reward(teacher, 4, terminated=False, truncated=False, focus_cost=.01)
        reduced = repair_reward(teacher, 4, terminated=False, truncated=False, focus_cost=.0025)
        self.assertAlmostEqual(reduced - original, .03)

    def test_identical_roi_repeats(self):
        result = roi_metrics([(1, (160, 160, 352, 352))] * 4)
        self.assertEqual(result["coverage"], 192 ** 2 / (4 * 512 ** 2))
        self.assertEqual(result["repeated_roi_fraction"], 1.)
        self.assertEqual(result["unique_cameras"], 1)

    def test_different_cameras(self):
        result = roi_metrics([(camera, (160, 160, 352, 352)) for camera in range(4)])
        self.assertEqual(result["coverage"], 192 ** 2 / 512 ** 2)
        self.assertEqual(result["repeated_roi_fraction"], 0.)

    def test_zero_focus(self):
        self.assertEqual(roi_metrics([])["coverage"], 0.)

    def test_independent_manifest(self):
        manifest = json.loads((ROOT / "frozen/DATA_MANIFEST.json").read_text())
        old, new = manifest["historical_val_seen"], manifest["independent_val_seen"]
        self.assertEqual((len(old), len(new)), (32, 32))
        self.assertFalse({row["source_episode_id"] for row in old} & {row["source_episode_id"] for row in new})
        self.assertFalse({row["payload"]["trajectory_id"] for row in old} & {row["payload"]["trajectory_id"] for row in new})

    def test_full_fork_parameter_and_moment_identity(self):
        unchanged = torch.load(ROOT / "frozen/fork_U.pt", weights_only=False, map_location="cpu")
        for branch in ("A", "B"):
            fork = torch.load(ROOT / "frozen" / f"fork_{branch}.pt", weights_only=False, map_location="cpu")
            for key in ("actor", "critic", "old_actor", "old_critic", "minibatch_rng", "torch_rng"):
                compare_nested(unchanged["trainer"][key], fork["trainer"][key], key)
            for key in ("actor_optimizer", "critic_optimizer"):
                compare_nested(unchanged["trainer"][key]["state"], fork["trainer"][key]["state"], key)
            self.assertEqual(fork["cursor"]["train_episode_index"], 70)
            self.assertFalse(fork["buffer_boundaries"])

    def test_branch_hyperparameters(self):
        configs = {branch: json.loads((ROOT / "frozen" / f"config_{branch}.json").read_text()) for branch in ("U", "A", "B")}
        ignore = {"task232_branch", "focus_cost"}
        self.assertEqual({key: value for key, value in configs["U"].items() if key not in ignore},
                         {key: value for key, value in configs["A"].items() if key not in ignore})
        self.assertEqual(configs["B"]["focus_cost"], .01)
        self.assertEqual(configs["B"]["rollout_minimum"], 128)
        self.assertEqual(configs["B"]["actor_learning_rate"], 3e-5)
        self.assertEqual(configs["B"]["critic_learning_rate"], 1e-4)

    def test_sr_alone_cannot_trigger_extension(self):
        metrics = {"four_focus_coverage": .04, "four_focus_repeat_fraction": 1., "zero_focus_fraction": 0.,
                   "SR": .5, "SPL": .4, "critic": {"branch_terminal_only": {"EV": 0.}}}
        base = {"subsets": {"independent": metrics}}
        branch = {"subsets": {"independent": copy.deepcopy(metrics), "historical": {"SR": .5625}}}
        branch["subsets"]["independent"]["SR"] = 1.
        probe = {"heads": {"val_seen": {"zero_focus_first_step_fraction": 0.}}}
        self.assertFalse(extension_eligibility(base, branch, probe, True)["eligible"])

    def test_extension_requires_all_frozen_conditions(self):
        metrics = {"four_focus_coverage": .04, "four_focus_repeat_fraction": 1., "zero_focus_fraction": 0.,
                   "SR": .5, "SPL": .4, "critic": {"branch_terminal_only": {"EV": 0.}}}
        base = {"subsets": {"independent": metrics}}
        branch = {"subsets": {"independent": copy.deepcopy(metrics), "historical": {"SR": .5625}}}
        branch["subsets"]["independent"].update(four_focus_coverage=.06, four_focus_repeat_fraction=.7)
        probe = {"heads": {"val_seen": {"zero_focus_first_step_fraction": 0.}}}
        self.assertTrue(extension_eligibility(base, branch, probe, True)["eligible"])
        self.assertFalse(extension_eligibility(base, branch, probe, False)["eligible"])


if __name__ == "__main__":
    unittest.main()
