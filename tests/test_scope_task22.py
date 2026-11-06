from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Experiment.SCOPE.Stage2.Task21.fusion import Focus, SpatialFusion
from Experiment.SCOPE.Stage2.Task21.incremental import IncrementalPerception
from Experiment.SCOPE.Stage2.Task22.actor import ActorState, HierarchicalDistribution, LatentAction, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save, joint_log_probability, load_episode_traces, navigation_transition, perception_loop, replay_decisions

torch.set_num_threads(1)


def make_state(count=0):
    glance = torch.randn(4, 16, 2560)
    history = torch.zeros(4, 7)
    history[:count, 0] = 1
    history[:count, 4:6] = .5
    history[:count, 6] = 1
    return ActorState(glance, torch.randn(2560), glance.clone(), torch.zeros(4, 16, 1), history, count)


class ScriptedActor:
    def __init__(self, focus_count):
        self.focus_count = focus_count

    def __call__(self, state):
        return HierarchicalDistribution(torch.tensor(100. if state.count == self.focus_count else -100.),
                                        torch.arange(4).float(), torch.zeros(4, 2), torch.full((4, 2), .8))


class Task22Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(222)
        self.actor = SpatialQueryActor().eval()
        self.state = make_state()

    def test_initial_distribution(self):
        distribution = self.actor(self.state)
        self.assertAlmostEqual(float(distribution.stop.probs), .25)
        torch.testing.assert_close(distribution.position.loc, torch.zeros(4, 2))
        torch.testing.assert_close(distribution.position.scale, torch.full((4, 2), .8))

    def test_center_roi_mapping(self):
        for latent in (-100., -10., 0., 10., 100.):
            action = LatentAction(False, 2, torch.tensor([latent, -latent]))
            center = action.center()
            self.assertTrue(((center >= 96) & (center <= 416)).all())
            focus = action.focus()
            left, top, right, bottom = focus.roi
            self.assertEqual((right - left, bottom - top), (192, 192))
            self.assertLessEqual(abs(left + 96 - float(center[0])), .5)
            self.assertLessEqual(abs(top + 96 - float(center[1])), .5)
        self.assertEqual(LatentAction(False, 0, torch.zeros(2)).focus().roi, (160, 160, 352, 352))

    def test_stop_probability_ignores_other_heads(self):
        distribution = self.actor(self.state)
        probability = distribution.log_prob(LatentAction(True))
        torch.testing.assert_close(probability, torch.log(distribution.stop.probs))
        probability.backward()
        self.assertIsNone(self.actor.camera_head[-1].weight.grad)
        self.assertIsNone(self.actor.position_head[-1].weight.grad)

    def test_joint_probability_formula(self):
        distribution = self.actor(self.state)
        action = LatentAction(False, 1, torch.tensor([.37, -.29]))
        expected = torch.log1p(-distribution.stop.probs) + torch.log(distribution.camera.probs[1])
        expected += distribution.position.log_prob(action.latent)[1].sum()
        torch.testing.assert_close(distribution.log_prob(action), expected)

    def test_all_heads_receive_gradients(self):
        losses = []
        for camera in range(4):
            logprob, entropy = self.actor.evaluate(self.state, LatentAction(False, camera, torch.tensor([.2 + camera, -.4])))
            losses.append(-logprob - .01 * entropy["joint_latent"])
        torch.stack(losses).sum().backward()
        for parameter in (self.actor.stop_head[-1].weight, self.actor.camera_head[-1].weight,
                          self.actor.position_head[-1].weight, self.actor.log_std,
                          self.actor.visual_projection[0].weight, self.actor.language_projection[0].weight,
                          self.actor.camera_embedding.weight, self.actor.position_projection.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_replay_in_train_mode(self):
        with torch.no_grad():
            distribution = self.actor(self.state)
            action = distribution.choose("training-sampling", torch.Generator().manual_seed(12))
            old = distribution.log_prob(action)
        self.actor.train()
        new, _ = self.actor.evaluate(self.state, action)
        torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-6)

    def test_camera_identity(self):
        with torch.no_grad():
            self.actor.position_head[-1].weight.normal_(0, .02)
        distribution = self.actor(self.state)
        self.assertFalse(torch.equal(distribution.position.loc[0], distribution.position.loc[1]))
        swapped = ActorState(self.state.glance.flip(0), self.state.language, self.state.fused.flip(0),
                             self.state.coverage, self.state.history, 0)
        self.assertFalse(torch.equal(distribution.camera.probs, self.actor(swapped).camera.probs))

    def test_incremental_union_and_repeat(self):
        calls = []
        def encode(focus):
            calls.append(focus)
            return torch.ones(64, 2560) * (focus.camera + 1)
        perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), encode)
        self.assertIs(perception.current, self.state.glance)
        perception.acquire(Focus(1))
        first = perception.current.clone()
        coverage = perception.coverage.clone()
        perception.acquire(Focus(1))
        self.assertTrue(torch.equal(coverage, perception.coverage))
        self.assertFalse(torch.equal(first, perception.current))
        self.assertTrue(torch.equal(first[0], self.state.glance[0]))
        self.assertEqual(len(calls), 2)
        perception.finish()
        with self.assertRaises(RuntimeError):
            perception.acquire(Focus(0))

    def test_zero_through_four_focus_trace(self):
        for count in range(5):
            perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), lambda focus: torch.ones(64, 2560))
            trace = perception_loop(ScriptedActor(count), perception, self.state.language, mode="deterministic",
                                    generator=torch.Generator().manual_seed(1), transition_id="episode:0")
            self.assertEqual(trace["focus_count"], count)
            self.assertEqual(len(trace["decisions"]), min(count + 1, 4))
            self.assertFalse(trace["forced_stop_has_log_probability"])
            self.assertEqual(len(replay_decisions(trace, "cpu")), min(count + 1, 4))
            for index, record in enumerate(trace["decisions"]):
                self.assertEqual(record["state"]["count"], index)
                if index:
                    self.assertFalse(torch.equal(record["state"]["glance"], record["state"]["fused"]))
            if count == 0:
                self.assertIs(perception.current, perception.glance)

    def test_serialize_and_gae_bridge(self):
        perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), lambda focus: torch.ones(64, 2560))
        trace = perception_loop(self.actor, perception, self.state.language, mode="training-sampling",
                                generator=torch.Generator().manual_seed(4), transition_id="episode:0")
        trace.update(episode_id="1", feedback={"navigation_done": True})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt"
            atomic_save(path, trace)
            restored = torch.load(path, weights_only=True)
        new, _ = joint_log_probability(self.actor, replay_decisions(restored, "cpu"))
        torch.testing.assert_close(new, restored["old_sequence_log_probability"], atol=1e-5, rtol=1e-5)
        transition = navigation_transition(restored, old_value=.2, reward=1., terminated=True,
                                           truncated=False, next_value=0., device="cpu")
        from Experiment.SCOPE.Stage2.training import gae_targets
        targets = gae_targets([transition])
        torch.testing.assert_close(targets.returns, torch.tensor([1.]))
        self.assertFalse(targets.old_log_probabilities.requires_grad)

    def test_sampling_reproducibility_and_branch_support(self):
        distribution = self.actor(self.state)
        generator = torch.Generator().manual_seed(32)
        actions = [distribution.choose("training-sampling", generator) for _ in range(256)]
        self.assertTrue(any(action.stop for action in actions))
        self.assertEqual({action.camera for action in actions if not action.stop}, set(range(4)))
        repeated = distribution.choose("training-sampling", torch.Generator().manual_seed(32))
        self.assertEqual(actions[0].stop, repeated.stop)
        self.assertEqual(actions[0].camera, repeated.camera)
        if not repeated.stop:
            self.assertTrue(torch.equal(actions[0].latent, repeated.latent))

    def test_std_bounds_and_forced_stop_rejected(self):
        with torch.no_grad():
            self.actor.log_std[0].fill_(-100)
            self.actor.log_std[1].fill_(100)
        std = self.actor(self.state).position.scale
        self.assertTrue((std >= .1 - 1e-7).all() and (std <= 1.5 + 1e-7).all())
        with self.assertRaises(ValueError):
            self.actor(make_state(4))

    def test_bad_actions_and_oracle_state(self):
        for action in (LatentAction(True, 0), LatentAction(False, 4, torch.zeros(2)),
                       LatentAction(False, 0, torch.tensor([float("nan"), 0.]))):
            with self.assertRaises(ValueError):
                action.validate()
        packed = self.state.packed()
        packed["oracle_distance"] = 1.
        with self.assertRaises(TypeError):
            ActorState.unpack(packed, "cpu")

    def test_deterministic_not_ppo_eligible(self):
        with self.assertRaises(ValueError):
            navigation_transition({"ppo_eligible": False}, old_value=0., reward=0., terminated=False,
                                  truncated=True, next_value=0., device="cpu")

    def test_actor_mode_is_separate_from_navigation_mode(self):
        from Experiment.SCOPE.Stage2.Task22.actor_runtime import ActorPolicy
        policy = object.__new__(ActorPolicy)
        policy.actor_mode = "training-sampling"
        policy.mode = "glance_only"
        self.assertEqual(policy.actor_mode, "training-sampling")
        with self.assertRaises(PermissionError):
            policy.finish({}, update=True)

    def test_incremental_frozen_and_budget_contract(self):
        with self.assertRaises(ValueError):
            IncrementalPerception(self.state.glance, SpatialFusion(), lambda focus: torch.zeros(64, 2560))
        perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), lambda focus: torch.zeros(64, 2560))
        for camera in range(4):
            perception.acquire(Focus(camera))
        with self.assertRaises(RuntimeError):
            perception.acquire(Focus(0))

    def test_tampered_old_probability_rejected(self):
        perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), lambda focus: torch.ones(64, 2560))
        trace = perception_loop(self.actor, perception, self.state.language, mode="training-sampling",
                                generator=torch.Generator().manual_seed(4), transition_id="episode:0")
        trace["old_sequence_log_probability"] += 1
        with self.assertRaises(ValueError):
            replay_decisions(trace, "cpu")

    def test_episode_loader_requires_bootstrap_on_truncation(self):
        perception = IncrementalPerception(self.state.glance, SpatialFusion().freeze(), lambda focus: torch.ones(64, 2560))
        trace = perception_loop(self.actor, perception, self.state.language, mode="training-sampling",
                                generator=torch.Generator().manual_seed(4), transition_id="run:1:0")
        trace.update(episode_id="1", feedback={}, provenance={}, actor_checkpoint_sha256="test",
                     terminated=False, truncated=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.pt"
            boundary = Path(directory) / "boundary.pt"
            atomic_save(path, trace)
            atomic_save(boundary, {"episode_boundary": True, "provenance": {}, "actor_checkpoint_sha256": "test",
                                   "report": {"episode_id": "1", "trace_paths": [str(path)]}})
            with self.assertRaises(ValueError):
                load_episode_traces(boundary)
            trace["bootstrap_state"] = self.state.packed()
            atomic_save(path, trace)
            loaded = load_episode_traces(boundary)
            self.assertTrue(torch.equal(loaded[0]["next_state"]["glance"], self.state.glance))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_inference_and_gradient_replay(self):
        actor = self.actor.cuda().eval()
        state = ActorState.unpack(self.state.packed(), "cuda")
        action = LatentAction(False, 2, torch.tensor([1.23, -.71], device="cuda"))
        with torch.inference_mode():
            old = actor(state).log_prob(action)
        actor.train()
        actual = actor(state).log_prob(action)
        torch.testing.assert_close(old, actual, atol=1e-6, rtol=1e-6)
        actual.backward()
        perception = IncrementalPerception(state.glance, SpatialFusion().cuda().freeze(),
                                           lambda focus: torch.ones(64, 2560, device="cuda"))
        trace = perception_loop(actor, perception, state.language, mode="training-sampling",
                                generator=torch.Generator(device="cuda").manual_seed(223), transition_id="gpu:0")
        self.assertGreater(trace["focus_count"], 0)
        replay_decisions(trace, "cpu")


if __name__ == "__main__":
    unittest.main()
