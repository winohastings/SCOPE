import unittest
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from Experiment.SCOPE.Stage2.Task22.actor import HierarchicalDistribution, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task233.spatial_warmup import reset_spatial, spatial_log_prob


class SpatialWarmupTests(unittest.TestCase):
    def test_existing_run_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            sentinel = output / 'PROGRESS_GATE.json'
            sentinel.write_text('original evidence')
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('spatial_warmup.py')),
                                     '--dataset', str(output / 'missing'), '--output', str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), 'original evidence')
            self.assertFalse((output / 'INCOMPLETE.json').exists())

    def test_stop_not_in_loss(self):
        stop = torch.tensor(2., requires_grad=True)
        camera = torch.zeros(4, requires_grad=True)
        mean = torch.zeros(4, 2, requires_grad=True)
        distribution = HierarchicalDistribution(stop, camera, mean, torch.ones(4, 2))
        loss = -spatial_log_prob(distribution, 2, torch.tensor([.4, -.2]))
        loss.backward()
        self.assertIsNone(stop.grad)
        self.assertIsNotNone(camera.grad)
        self.assertGreater(float(mean.grad.abs().sum()), 0.)

    def test_initialization_resets_only_spatial_outputs(self):
        actor = SpatialQueryActor()
        stop = {name: value.clone() for name, value in actor.stop_head.state_dict().items()}
        reset_spatial(actor)
        self.assertTrue(all(not parameter.requires_grad for parameter in actor.stop_head.parameters()))
        self.assertEqual(int(actor.camera_head[-1].weight.count_nonzero()), 0)
        self.assertTrue(all(torch.equal(value, stop[name]) for name, value in actor.stop_head.state_dict().items()))

    def test_spatial_logprob_matches_components(self):
        distribution = HierarchicalDistribution(torch.tensor(0.), torch.arange(4.).float(), torch.zeros(4, 2), torch.ones(4, 2))
        latent = torch.tensor([.3, .2])
        expected = distribution.camera.log_prob(torch.tensor(1)) + distribution.position.log_prob(latent)[1].sum()
        self.assertTrue(torch.equal(spatial_log_prob(distribution, 1, latent), expected))


if __name__ == '__main__':
    unittest.main()
