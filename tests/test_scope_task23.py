import copy
import json
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Experiment.SCOPE.Stage2.Task21.fusion import SpatialFusion
from Experiment.SCOPE.Stage2.Task21.incremental import IncrementalPerception
from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import perception_loop, replay_decisions
from Experiment.SCOPE.Stage2.Task23.critic import NavigationCritic
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, PPOTrainer, gae, navigation_reward, sequence_objective
from Experiment.SCOPE.Stage2.Task23.audit import compare_nested

torch.set_num_threads(1)


def config():
    return json.loads((ROOT / "Experiment/SCOPE/Stage2/Task23/config.json").read_text())


def state():
    glance = torch.randn(4, 16, 2560)
    return ActorState(glance, torch.randn(2560), glance, torch.zeros(4, 16, 1), torch.zeros(4, 7), 0)


class Task23Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(233)

    def test_critic_pre_focus_only(self):
        critic = NavigationCritic()
        observation = state()
        self.assertEqual(critic(observation).shape, ())
        self.assertTrue(torch.isfinite(critic(observation)))
        with self.assertRaises(ValueError):
            critic(ActorState(observation.glance, observation.language, observation.fused + 1,
                              observation.coverage, observation.history, 0))

    def test_reward_exact_formula(self):
        teacher = {"distance_before": 8., "distance_after": 2., "terminal": True, "success": True}
        self.assertAlmostEqual(navigation_reward(teacher, 4, terminated=True, truncated=False), 2.15)
        teacher.update(distance_after=10., success=False)
        self.assertAlmostEqual(navigation_reward(teacher, 0, terminated=True, truncated=False), -.2 * 2 / 3 - .01 - .5)
        teacher["terminal"] = False
        self.assertAlmostEqual(navigation_reward(teacher, 1, terminated=False, truncated=True), -.2 * 2 / 3 - .02)

    def test_reward_rejects_missing_oracle_and_wrong_boundary(self):
        with self.assertRaises(ValueError):
            navigation_reward(None, 0, terminated=False, truncated=False)
        teacher = {"distance_before": float("inf"), "distance_after": 2., "terminal": False, "success": True}
        with self.assertRaises(ValueError):
            navigation_reward(teacher, 0, terminated=False, truncated=False)
        teacher["distance_before"] = 3.
        with self.assertRaises(ValueError):
            navigation_reward(teacher, 0, terminated=True, truncated=False)

    def test_gae_terminal_truncation_and_episode_isolation(self):
        first = NavigationTransition({"terminated": False, "truncated": False}, .5, .7, .1, "episode_a")
        second = NavigationTransition({"terminated": True, "truncated": False}, .7, 999., 2., "episode_a")
        third = NavigationTransition({"terminated": False, "truncated": True}, .2, .8, .3, "episode_b")
        advantages, returns = gae([first, second, third])
        expected = torch.tensor([.1 + .99 * .7 - .5 + .99 * .95 * (2 - .7), 2 - .7, .3 + .99 * .8 - .2])
        torch.testing.assert_close(advantages, expected)
        self.assertFalse(advantages.requires_grad or returns.requires_grad)

    def test_incomplete_episode_rejected(self):
        transition = NavigationTransition({"terminated": False, "truncated": False}, 0., 0., 0., "one")
        with self.assertRaises(ValueError):
            gae([transition])

    def traces(self, trainer, count=4):
        result = []
        for index in range(count):
            observation = state()
            perception = IncrementalPerception(observation.glance, SpatialFusion().freeze(), lambda focus: torch.ones(64, 2560))
            trace = perception_loop(trainer.old_actor, perception, observation.language, mode="training-sampling",
                                    generator=torch.Generator().manual_seed(300 + index), transition_id=f"run{index}:1:0")
            with torch.no_grad():
                value = float(trainer.old_critic(observation))
            trace.update(purpose="train", policy_version=trainer.updates, old_value=value, feedback={}, episode_id="1",
                         terminated=True, truncated=False, next_state=None,
                         teacher_for_reward_only={"distance_before": 4., "distance_after": 2. if index % 2 else 5.,
                                                  "terminal": True, "success": bool(index % 2)})
            result.append(trace)
        return result

    def test_real_optimizer_and_exact_restore(self):
        trainer = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        transitions = trainer.prepare_episode(self.traces(trainer))
        before = copy.deepcopy(trainer.payload())
        report, targets = trainer.update(transitions)
        after = copy.deepcopy(trainer.payload())
        self.assertGreater(report["actor_parameter_l2_change"], 0)
        self.assertGreater(report["critic_parameter_l2_change"], 0)
        self.assertLess(report["initial_ratio_max_error"], 1e-4)
        self.assertGreater(trainer.optimizer_steps, 0)
        self.assertFalse(targets["advantages"].requires_grad)
        trainer.restore(before)
        trainer.update(transitions)
        compare_nested(after, trainer.payload())
        fresh = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        fresh.restore(before)
        fresh.update(transitions)
        compare_nested(after, fresh.payload())

    def test_stale_and_validation_rollouts_rejected(self):
        trainer = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        traces = self.traces(trainer, 1)
        traces[0]["purpose"] = "val_seen"
        with self.assertRaises(ValueError):
            trainer.prepare_episode(traces)
        traces[0]["purpose"] = "train"
        traces[0]["policy_version"] = 99
        with self.assertRaises(ValueError):
            trainer.prepare_episode(traces)

    def test_snapshots_are_frozen_and_independent(self):
        trainer = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.old_actor.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.old_critic.parameters()))
        self.assertFalse({id(parameter) for parameter in trainer.actor.parameters()} & {id(parameter) for parameter in trainer.critic.parameters()})
        old = next(trainer.old_actor.parameters()).clone()
        with torch.no_grad():
            next(trainer.actor.parameters()).add_(1)
        self.assertTrue(torch.equal(old, next(trainer.old_actor.parameters())))

    def test_per_decision_ratio_and_sequence_sum(self):
        trainer = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        trace = self.traces(trainer, 1)[0]
        decisions = replay_decisions(trace, "cpu")
        settings = config()
        settings["entropy_weights"] = {key: 0. for key in settings["entropy_weights"]}
        loss, ratios, _, _ = sequence_objective(trainer.actor, decisions, torch.tensor(2.), settings)
        torch.testing.assert_close(ratios, torch.ones(len(decisions)))
        self.assertAlmostEqual(float(loss), -2. * len(decisions), places=5)

    def test_nonunit_clipped_ratios_not_sequence_product(self):
        trainer = PPOTrainer(SpatialQueryActor(), config(), "cpu")
        decisions = replay_decisions(self.traces(trainer, 1)[0], "cpu")
        settings = config()
        settings["entropy_weights"] = {key: 0. for key in settings["entropy_weights"]}

        class ShiftedActor:
            def evaluate(self, observation, action):
                probability, entropy = trainer.actor.evaluate(observation, action)
                return probability + torch.tensor(1.5).log(), entropy

        loss, ratios, _, _ = sequence_objective(ShiftedActor(), decisions, torch.tensor(2.), settings)
        torch.testing.assert_close(ratios, torch.full_like(ratios, 1.5))
        self.assertAlmostEqual(float(loss), -2.4 * len(decisions), places=5)
        negative, _, _, _ = sequence_objective(ShiftedActor(), decisions, torch.tensor(-2.), settings)
        self.assertAlmostEqual(float(negative), 3. * len(decisions), places=5)

    def test_truncation_bootstraps_without_cross_episode_carry(self):
        transitions = [NavigationTransition({"terminated": False, "truncated": True}, .2, .8, .3, "same_data_id"),
                       NavigationTransition({"terminated": True, "truncated": False}, .1, 0., 100., "same_data_id")]
        advantages, returns = gae(transitions)
        self.assertAlmostEqual(float(advantages[0]), .3 + .99 * .8 - .2, places=6)
        self.assertAlmostEqual(float(returns[0]), .3 + .99 * .8, places=6)


if __name__ == "__main__":
    unittest.main()
