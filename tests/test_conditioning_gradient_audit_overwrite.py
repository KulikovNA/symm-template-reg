import csv
import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.engine.overfit_trainer import _write_conditioning_gradient_audit
from symm_template_reg.engine.single_fragment import TrainingCounters


class ConditioningGradientAuditOverwriteTest(unittest.TestCase):
    def _evaluation(self, root: Path, epoch: int, distance: float) -> None:
        directory = root / "evaluations" / f"epoch_{epoch:04d}"
        directory.mkdir(parents=True)
        (directory / "context_conditioning_diagnostics.json").write_text(
            json.dumps({"q_aux_summary_pairwise_distance_matrix": [[0.0, distance], [distance, 0.0]]})
        )
        with (directory / "per_sample_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_id", "fragment_id", "frame_id"))
            writer.writeheader()
            writer.writerows((
                {"sample_id": "a", "fragment_id": 0, "frame_id": 2},
                {"sample_id": "b", "fragment_id": 1, "frame_id": 4},
            ))

    def test_new_best_atomically_replaces_existing_pairwise_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._evaluation(root, 0, 1.0)
            self._evaluation(root, 1, 2.0)
            counters = TrainingCounters()
            for epoch in (0, 1):
                _write_conditioning_gradient_audit(
                    root, label="best", source_epoch=epoch, gradient_norm=1.0,
                    module_gradient_norms={}, nonzero_gradient_parameter_fraction=1.0,
                    counters=counters, status="best_after_real_backward",
                )
            pairwise = root / "conditioning_gradient_audits" / "best" / "fragment_conditioning_pairwise.csv"
            with pairwise.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["q_aux_summary_distance"]), 2.0)
            audit = json.loads((pairwise.parent / "audit.json").read_text())
            self.assertEqual(audit["source_epoch"], 1)
            self.assertFalse(pairwise.with_suffix(".csv.tmp").exists())


if __name__ == "__main__":
    unittest.main()
