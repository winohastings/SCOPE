import dataclasses
import itertools
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/Stage2"
sys.path.insert(0, str(ROOT))
from data_guard import check_episode, episodes, partition
from features import CurrentFeature, CurrentFeatureBank, Geometry, SpatialAdapter, candidate_summaries
from policy import ActorConfig, FEEDBACK_SIZE, HIDDEN, IndependentCritic, PolicyObservation, RecurrentSession, ROIActor
from training import Decision, PPOTrainer, Transition, build_trainer, gae_targets, joint_log_probability
from trajectory import ExecutionFeedback, body_displacement, cost_aware_reward

torch.set_num_threads(1)


def observation(selected=(), has_feedback=False):
    return PolicyObservation(torch.randn(4, 16, 2560), torch.randn(2560),
                             tuple(torch.randn(64, 2560) for candidate in selected), tuple(selected),
                             torch.zeros(HIDDEN), torch.randn(2560), torch.randn(FEEDBACK_SIZE), has_feedback)


def transition(actor, critic, reward=0.5, terminated=False, identifier="train:0"):
    initial = observation()
    first = actor(initial)
    candidate = int(first.probs[:16].argmax())
    decision = Decision(initial.snapshot(), candidate, actor.action_mask((), "cpu"), first.log_prob(torch.tensor(candidate)).detach())
    subsequent = dataclasses.replace(initial, selected=(candidate,), focus=(torch.randn(64, 2560),))
    second = actor(subsequent)
    action = int(second.probs.argmax())
    other = Decision(subsequent.snapshot(), action, actor.action_mask((candidate,), "cpu"), second.log_prob(torch.tensor(action)).detach())
    return Transition([decision, other], initial.snapshot(), critic(initial).detach(), reward, terminated, False, 0.2, identifier)


class FeaturesTest(unittest.TestCase):
    def test_native_shapes_and_cache_exactly_once(self):
        calls = []

        def encode(images):
            calls.append(list(images))
            return [(torch.ones(16 if image == "g" else 64, 2560), (1, 8, 8) if image == "g" else (1, 16, 16)) for image in images]

        bank = CurrentFeatureBank("pose-A", "encoder-A", encode)
        bank.acquire(["g"] * 4, [Geometry.glance(camera) for camera in range(4)])
        self.assertEqual(tuple(bank.glance().shape), (4, 16, 2560))
        bank.acquire(["g"], [Geometry.glance(0)])
        bank.select(5, "f")
        self.assertEqual(tuple(bank.read_focus(5).tokens.shape), (64, 2560))
        self.assertEqual(bank.encoder_images, 5)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(PermissionError):
            bank.read_focus(6)
        with self.assertRaises(ValueError):
            bank.select(5, "f")

    def test_bad_grid_and_trainable_features_rejected(self):
        with self.assertRaises(ValueError):
            CurrentFeature("state", "hash", Geometry.glance(0), (1, 16, 16), torch.zeros(16, 2560))
        with self.assertRaises(ValueError):
            CurrentFeature("state", "hash", Geometry.glance(0), (1, 8, 8), torch.zeros(16, 2560, requires_grad=True))

    def test_quadrant_spatial_order(self):
        glance = torch.arange(64).view(4, 16, 1).expand(4, 16, 2560).float()
        torch.testing.assert_close(candidate_summaries(glance)[:4, 0], torch.tensor([2.5, 4.5, 10.5, 12.5]))
        self.assertEqual(Geometry.focus(15).roi, (256, 256, 512, 512))

    def test_camera_rays_and_crop_projection(self):
        for camera in range(4):
            rays = Geometry.glance(camera).rays()
            self.assertEqual(tuple(rays.shape), (16, 3))
            torch.testing.assert_close(rays.norm(dim=-1), torch.ones(16))
        self.assertLess(float(Geometry.glance(1).rays()[:, 1].mean()), 0)
        self.assertGreater(float(Geometry.glance(2).rays()[:, 1].mean()), 0)
        self.assertTrue((Geometry.focus(0).rays()[:, 1:] < 0).all())
        self.assertTrue((Geometry.focus(3).rays()[:, 1:] > 0).all())

    def test_zero_initialized_spatial_adapter_identity(self):
        feature = CurrentFeature("state", "hash", Geometry.glance(0), (1, 8, 8), torch.randn(16, 2560))
        torch.testing.assert_close(SpatialAdapter()(feature), feature.tokens, rtol=0, atol=0)
        feature = dataclasses.replace(feature, tokens=feature.tokens.bfloat16())
        torch.testing.assert_close(SpatialAdapter()(feature), feature.tokens, rtol=0, atol=0)


class ActorTest(unittest.TestCase):
    def test_all_switches_share_core_and_backward(self):
        for flags in itertools.product((False, True), repeat=5):
            actor = ROIActor(ActorConfig(*flags))
            distribution = actor(observation(has_feedback=True))
            action = int(distribution.probs.argmax())
            (-distribution.log_prob(torch.tensor(action))).backward()
            self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters() if parameter.grad is not None))
            self.assertEqual(float(distribution.probs[16]) > 0, flags[-1])

    def test_language_and_focus_change_logits(self):
        actor = ROIActor()
        original = observation((3,))
        logits = actor(original).logits
        changed = dataclasses.replace(original, language=original.language + 10)
        self.assertGreater(float((actor(changed).logits[:3] - logits[:3]).abs().sum()), 1e-6)
        changed = dataclasses.replace(original, focus=(original.focus[0] + 10,))
        self.assertGreater(float((actor(changed).logits[:3] - logits[:3]).abs().sum()), 1e-6)
        actor.config = dataclasses.replace(actor.config, use_language=False, use_focus_memory=False)
        torch.testing.assert_close(actor(original).logits, actor(changed).logits)

    def test_no_unselected_focus_and_masks(self):
        actor = ROIActor()
        initial = observation((3,))
        self.assertEqual(float(actor(initial).probs[3]), 0)
        with self.assertRaises(ValueError):
            actor(dataclasses.replace(initial, selected=()))
        with self.assertRaises(ValueError):
            actor(initial, torch.ones(17, dtype=torch.bool))
        fixed = ROIActor(ActorConfig.regime("A"))
        self.assertEqual(float(fixed(observation()).probs[16]), 0)

    def test_zero_one_two_focus_stop_sequences(self):
        actor = ROIActor()
        for selected in ((), (0,), (0, 1)):
            distribution = actor(observation(selected))
            self.assertGreater(float(distribution.probs[16]), 0)
            if len(selected) == 2:
                self.assertEqual(float(distribution.probs[:16].sum()), 0)

    def test_navigation_and_feedback_switches_remove_dependence(self):
        initial = observation(has_feedback=True)
        changed = dataclasses.replace(initial, feedback=initial.feedback + 100)
        actor = ROIActor(ActorConfig(use_action_feedback=False))
        torch.testing.assert_close(actor(initial).probs, actor(changed).probs)
        actor = ROIActor(ActorConfig(use_navigation_state=False))
        changed = dataclasses.replace(changed, navigation_previous=torch.randn(HIDDEN), navigation_visual=torch.randn(2560))
        torch.testing.assert_close(actor(initial).probs, actor(changed).probs)

    def test_actor_critic_independent(self):
        actor, critic = ROIActor(), IndependentCritic()
        self.assertFalse({parameter.data_ptr() for parameter in actor.parameters()} & {parameter.data_ptr() for parameter in critic.parameters()})
        with self.assertRaises(ValueError):
            critic(observation((2,)))

    def test_gru_actual_feedback_reset_and_gradients(self):
        trainer = build_trainer()
        actor, critic = trainer.actor, trainer.critic
        initial = observation(has_feedback=True)
        changed = dataclasses.replace(initial, feedback=initial.feedback + 1)
        self.assertFalse(torch.allclose(actor(initial).probs, actor(changed).probs))
        loss = -actor(initial).log_prob(torch.tensor(1))
        loss.backward()
        self.assertGreater(float(actor.encoder.state_gru.weight_ih.grad.norm()), 0)
        session = RecurrentSession("cpu")
        session.complete(actor, critic, observation(), observation(), torch.randn(2560), torch.randn(FEEDBACK_SIZE))
        self.assertEqual(session.completed_steps, 1)
        self.assertTrue(session.has_feedback)
        session.reset()
        self.assertEqual(session.completed_steps, 0)
        self.assertFalse(session.has_feedback)
        self.assertEqual(float(session.actor_previous.norm()), 0)

    def test_observation_no_ground_truth_fields(self):
        names = {field.name for field in dataclasses.fields(PolicyObservation)}
        self.assertFalse(names & {"reward", "success", "spl", "ndtw", "distance", "goal", "collision"})
        with self.assertRaises(TypeError):
            PolicyObservation(**dataclasses.asdict(observation()), distance=1)


class PPOTest(unittest.TestCase):
    def test_joint_ratio_initially_one(self):
        trainer = build_trainer()
        item = transition(trainer.actor, trainer.critic)
        new_log, entropy = joint_log_probability(trainer.actor, item.decisions)
        self.assertAlmostEqual(float((new_log - item.old_log_probability).exp()), 1, places=5)
        self.assertGreater(float(entropy), 0)

    def test_gae_terminal_truncation_and_episode_boundaries(self):
        trainer = build_trainer()
        first = transition(trainer.actor, trainer.critic, reward=1, terminated=True)
        first.old_value = torch.tensor(0.5)
        first.next_value = 999
        second = transition(trainer.actor, trainer.critic, reward=2, identifier="train:1")
        second.old_value = torch.tensor(0.2)
        second.next_value = 0.4
        second.truncated = True
        targets = gae_targets([first, second])
        torch.testing.assert_close(targets.returns, torch.tensor([1.0, 2.396]))
        self.assertFalse(targets.returns.requires_grad)

    def test_ppo_targets_fixed_and_parameters_change(self):
        trainer = build_trainer()
        rollout = [transition(trainer.actor, trainer.critic, reward=reward) for reward in (1, -1)]
        targets = gae_targets(rollout)
        frozen = targets.returns.clone()
        previous = trainer.actor.query[0].weight.detach().clone()
        statistics = trainer.update(rollout, epochs=2, targets=targets)
        self.assertFalse(torch.equal(previous, trainer.actor.query[0].weight))
        torch.testing.assert_close(targets.returns, frozen, rtol=0, atol=0)
        self.assertGreater(statistics[0]["actor_grad_norm"], 0)
        self.assertGreater(statistics[0]["critic_grad_norm"], 0)

    def test_checkpoint_resume_exact(self):
        trainer = build_trainer()
        rollout = [transition(trainer.actor, trainer.critic, reward=reward) for reward in (1, -1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            trainer.save(path, provenance={"hash": "frozen"}, cursor={"episode": 1})
            trainer.update(rollout, epochs=2, minibatch_size=1)
            expected_rng = (random.random(), np.random.rand(), torch.rand(1))
            expected = {name: parameter.detach().clone() for name, parameter in trainer.actor.named_parameters()}
            resumed = build_trainer(seed=999)
            cursor = resumed.restore(path, provenance={"hash": "frozen"})
            self.assertEqual(cursor, {"episode": 1})
            resumed.update(rollout, epochs=2, minibatch_size=1)
            actual_rng = (random.random(), np.random.rand(), torch.rand(1))
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)
            for name, parameter in resumed.actor.named_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                resumed.restore(path, provenance={"hash": "wrong"})
            with self.assertRaises(ValueError):
                trainer.save(path, provenance={}, cursor={}, episode_boundary=False)


class FeedbackGuardTest(unittest.TestCase):
    def test_feedback_body_frame(self):
        actual = body_displacement([0, 0, 0, np.pi / 2], [0, 1, 0, np.pi / 2])
        np.testing.assert_allclose(actual, [1, 0, 0, 0], atol=1e-6)
        feedback = ExecutionFeedback([0, 0, 0, 0], [1, 0, 0, 0], [[1, 0, 0, 0]], [[1, 0, 0, 0]], 128, 10, 20, False, "running", True)
        self.assertEqual(feedback.vector().shape, (18,))
        np.testing.assert_allclose(feedback.vector()[8:12], 0)
        with self.assertRaises(ValueError):
            dataclasses.replace(feedback, executed_poses=[])

    def test_reward_cost_and_missing_teacher(self):
        arguments = dict(distance_before=4, distance_after=3, collision=False, terminal=False, success=False, spl=0, latency_ms=10)
        self.assertGreater(cost_aware_reward(**arguments, focus_count=0), cost_aware_reward(**arguments, focus_count=2))
        with self.assertRaises(ValueError):
            cost_aware_reward(**(arguments | {"distance_before": None}), focus_count=0)

    def test_train_dev_heldout_guard(self):
        payload = partition()
        self.assertFalse(set(payload["training_scenes"]) & set(payload["development_scenes"]))
        for purpose in ("train", "dev"):
            record = episodes(purpose, 1)[0]
            check_episode(record["source_episode_id"], record["scene_id"], purpose)
        with self.assertRaises(PermissionError):
            check_episode("1", "zsNo4HB9uLZ", "val_unseen")
        record = episodes("dev", 1)[0]
        with self.assertRaises(PermissionError):
            check_episode(record["source_episode_id"], record["scene_id"], "train")


if __name__ == "__main__":
    unittest.main()
