from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.engine.single_fragment import load_model_initialization


class InitCheckpointSemanticsTest(unittest.TestCase):
    def test_init_loads_model_only(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model": source.state_dict(), "optimizer": {"sentinel": 1}}, path)
            report = load_model_initialization(target, path, strict=True)
        self.assertTrue(torch.equal(source.weight, target.weight))
        self.assertFalse(report["optimizer_loaded"])
        self.assertFalse(report["counters_loaded"])

    def test_k1_state_cannot_strictly_initialize_k8_queries(self):
        class Queries(torch.nn.Module):
            def __init__(self, count):
                super().__init__()
                self.query_embedding = torch.nn.Embedding(count, 4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "k1.pth"
            torch.save({"model": Queries(1).state_dict()}, path)
            with self.assertRaises(RuntimeError):
                load_model_initialization(Queries(8), path, strict=True)


if __name__ == "__main__":
    unittest.main()
