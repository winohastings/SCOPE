import unittest
import numpy as np

from Experiment.SCOPE.Stage2.Task233.teacher_candidates import candidates, random_action, roi, geometry
from Experiment.SCOPE.Stage2.Task233.expert_scoring import sample_path, world_to_body, score
from Experiment.SCOPE.Stage2.Task233.analyze import gate_decision, paired_interval, rank_correlation


class Gate0Tests(unittest.TestCase):
    def test_calibration_hash_matches_persisted_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from Experiment.SCOPE.Stage2.Task233.gate0 import freeze_thresholds, digest
        records = [{"anchor_id": str(index), "variant": variant, "score_gt": -1., "progress": .2,
                    "navigation_prefix": []} for index in range(8)
                   for variant in ("stop", "stop_repeat", "candidate_0", "candidate_repeat")]
        with tempfile.TemporaryDirectory() as directory:
            frozen = freeze_thresholds(Path(directory), records)
            self.assertEqual(frozen["calibration_rows_sha256"], digest(json.loads(json.dumps(records))))
            self.assertEqual(frozen["min_progress_m"], .02)
            self.assertEqual(frozen["min_gt_margin"], .0001)

    def test_scalar_weight_fingerprint(self):
        import torch
        from Experiment.SCOPE.Stage2.Task233.server import weight_fingerprint
        module = torch.nn.Module()
        module.register_parameter("scalar", torch.nn.Parameter(torch.tensor(1.)))
        before = weight_fingerprint(module)
        self.assertEqual(before, weight_fingerprint(module))
        with torch.no_grad():
            module.scalar.add_(1.)
        self.assertNotEqual(before, weight_fingerprint(module))

    def test_candidates_are_reproducible_legal_and_continuous(self):
        records = candidates(233)
        self.assertEqual(records, candidates(233))
        self.assertEqual(len(records), 8)
        self.assertEqual([record["camera"] for record in records[:4]], list(range(4)))
        for record in records + [random_action(seed) for seed in range(50)]:
            bounds = roi(record)
            self.assertEqual(bounds[2] - bounds[0], 192)
            self.assertEqual(bounds[3] - bounds[1], 192)
            self.assertTrue(all(0 <= value <= 512 for value in bounds))
            self.assertTrue(np.isfinite(record["latent"]).all())

    def test_history_repetition_and_distinct_camera_geometry(self):
        records = candidates(1)
        repeated = geometry([records[0]] * 4)
        diverse = geometry(records[:4])
        self.assertEqual(repeated["repeat_count"], 3)
        self.assertEqual(diverse["repeat_count"], 0)
        self.assertEqual(diverse["union_coverage"], 4 * repeated["union_coverage"])

    def test_coordinate_convention(self):
        result = world_to_body([[0., 1., 0.], [-1., 0., 0.]], [0., 0., 0., 0.], [0., np.pi / 2])
        np.testing.assert_allclose(result[:, :2], [[1., 0.], [0., 1.]])
        for yaw in np.linspace(-np.pi, np.pi, 9):
            anchor = np.asarray([2., -3., 1., yaw])
            points = np.asarray([[3., 1., 2.], [-1., -5., 1.]])
            body = world_to_body(points, anchor, [yaw, yaw])
            recovered = np.column_stack((anchor[0] - np.sin(yaw) * body[:, 0] - np.cos(yaw) * body[:, 1],
                                         anchor[1] + np.cos(yaw) * body[:, 0] - np.sin(yaw) * body[:, 1], anchor[2] + body[:, 2]))
            np.testing.assert_allclose(recovered, points, atol=1e-6)

    def test_path_sampling_and_degenerate_path(self):
        sampled = sample_path([[0, 0, 0], [0, 0, 0], [2, 0, 0]], [.25, 1, 3])
        np.testing.assert_allclose(sampled[:, 0], [.25, 1, 2])
        with self.assertRaises(ValueError):
            sample_path([[0, 0, 0], [0, 0, 0]], [1])

    def test_score_uses_wrapped_yaw_and_navigation_stop(self):
        actions = np.zeros((8, 4))
        expert = {"actions": actions.tolist(), "stop": False}
        self.assertEqual(score(actions, False, expert)["score_gt"], 0.)
        self.assertEqual(score(actions, True, expert)["score_gt"], -1.)
        actions[:, 3] = 2 * np.pi
        self.assertAlmostEqual(score(actions, False, expert)["score_gt"], 0.)

    def test_bootstrap_uses_episode_not_branch_count(self):
        records = [{"episode_uid": "one", "gain": 1.}] * 8 + [{"episode_uid": "two", "gain": 3.}]
        result = paired_interval(records, "gain")
        self.assertEqual(result["mean"], 2.)
        self.assertEqual(result["episodes"], 2)

    def test_gate_fails_closed_on_missing_count_or_integrity(self):
        thresholds = {"minimum_evaluation_anchors": 64, "minimum_heldout_anchors": 32,
                      "minimum_confident_fraction": .25, "paired_ci_level": .9833333333333333,
                      "bootstrap_episode_replicates": 100, "min_progress_m": .02}
        rows = [{"partition": "teacher_heldout", "episode_uid": str(index // 2), "confident": True,
                 "teacher_minus_stop": .1, "teacher_minus_fixed": .1, "teacher_minus_random": .1} for index in range(64)]
        self.assertEqual(gate_decision(rows, thresholds, True)["status"], "PASS")
        self.assertEqual(gate_decision(rows[:32], thresholds, True)["status"], "INCONCLUSIVE")
        self.assertEqual(gate_decision(rows, thresholds, False)["status"], "FAIL")
        for row in rows:
            row["teacher_minus_random"] = 0.
        self.assertEqual(gate_decision(rows, thresholds, True)["status"], "INCONCLUSIVE")

    def test_rank_ties_not_arbitrarily_ranked(self):
        self.assertIsNone(rank_correlation([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(rank_correlation([1, 1, 3], [2, 2, 4]), 1.)


if __name__ == "__main__":
    unittest.main()
