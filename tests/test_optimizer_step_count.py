from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.engine.overfit_trainer import _build_scheduler, _unique_run_directory
from symm_template_reg.engine.single_fragment import TrainingCounters


class OptimizerStepCountTest(unittest.TestCase):
    def test_accumulation_counts_batches_updates_and_samples_separately(self):
        counters = TrainingCounters()
        for batch in range(5):
            counters.record_batch(2)
            if (batch + 1) % 2 == 0 or batch == 4:
                counters.record_optimizer_step()
        self.assertEqual(counters.batch_step, 5)
        self.assertEqual(counters.optimizer_step, 3)
        self.assertEqual(counters.samples_seen, 10)
        self.assertEqual(counters.to_dict()["global_step"], 3)

    def test_constant_scheduler_keeps_learning_rate(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        scheduler = _build_scheduler(optimizer, {"type": "constant"}, 1000)
        values = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            values.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(values, [3e-4] * 5)

    def test_run_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            first = _unique_run_directory(Path(directory), "stage")
            second = _unique_run_directory(Path(directory), "stage")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
