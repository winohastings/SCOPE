import importlib.util
import unittest
from pathlib import Path

import torch

MODULE = Path(__file__).resolve().parents[1] / "Experiment/SCOPE/legacy_runtime/starVLA/model/modules/qwen35_vision.py"
SPEC = importlib.util.spec_from_file_location("scope_vision", MODULE)
VISION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISION)


class ScopeVisionTest(unittest.TestCase):
    def test_mixed_grid_budgets(self):
        grids = torch.tensor([[1, 8, 8]] * 4 + [[1, 16, 16]] * 2)
        tokens = VISION.validate_scope_grids(grids, [16] * 4 + [64] * 2)
        self.assertEqual(sum(tokens), 192)
        with self.assertRaises(ValueError):
            VISION.validate_scope_grids(grids, [64] * 6)

    def test_history_cache_contract(self):
        for grid, count in (([1, 8, 8], 16), ([1, 16, 16], 64)):
            tokens = torch.randn(count, 2560, dtype=torch.bfloat16)
            pooled = VISION.pool_qwen35_postmerge(tokens, grid, target_tokens=4)
            self.assertEqual(tuple(pooled.shape), (4, 2560))
            bits = VISION.bf16_to_numpy_bits(pooled)
            restored = VISION.numpy_bits_to_bf16(bits, torch.device("cpu"))
            torch.testing.assert_close(restored, pooled.to(torch.bfloat16), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
