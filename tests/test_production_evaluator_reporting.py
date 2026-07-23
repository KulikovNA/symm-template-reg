import csv
import json
import tempfile
import unittest
from pathlib import Path

from symm_template_reg.engine.production_evaluator import (
    aggregate_production_metrics,
    format_validation_report,
    write_evaluation_report,
    write_validation_tracking,
)


def _rows() -> list[dict]:
    rows = []
    index = 0
    for frame_id in (2, 4, 5, 8):
        for fragment_id in (0, 1, 2, 3):
            index += 1
            rows.append(
                {
                    "sample_id": (
                        "train/scene_000000/"
                        f"frame_{frame_id:06d}/fragment_{fragment_id:04d}"
                    ),
                    "source_dataset_split": "train",
                    "evaluation_role": "overfit_validation",
                    "scene_id": "scene_000000",
                    "frame_id": frame_id,
                    "fragment_id": fragment_id,
                    "aux_coordinate_p95_mm": float(index),
                    "exact_global_projected_correspondence_p95_mm": (
                        1.0 if index <= 8 else 3.0
                    ),
                    "exact_global_projection_alignment_p95_mm": (
                        1.0 if index <= 8 else 3.0
                    ),
                    "exact_global_projection_rotation_error_deg": (
                        0.5 if index <= 4 else 2.0
                    ),
                    "exact_global_projection_translation_error_mm": (
                        0.25 if index <= 4 else 1.0
                    ),
                    "physical_score": float(index),
                    "k16_exact_global_triangle_recall": index / 16.0,
                    "k16_fallback_fraction": (16 - index) / 16.0,
                }
            )
    return rows


class ProductionEvaluatorReportingTest(unittest.TestCase):
    def test_aggregates_all_samples_frames_fragments_and_metrics(self):
        summary = aggregate_production_metrics(_rows())
        self.assertEqual(summary["source_dataset_split"], "train")
        self.assertEqual(summary["evaluation_role"], "overfit_validation")
        self.assertEqual(summary["num_samples"], 16)
        self.assertEqual(summary["num_frames"], 4)
        self.assertEqual(summary["num_physical_fragments"], 4)
        self.assertEqual(summary["frame_ids"], [2, 4, 5, 8])
        self.assertEqual(summary["fragment_ids"], [0, 1, 2, 3])
        self.assertEqual(len(summary["per_frame"]), 4)
        self.assertEqual(len(summary["per_physical_fragment"]), 4)
        self.assertTrue(
            all(row["num_observations"] == 4 for row in summary["per_frame"])
        )
        self.assertTrue(
            all(
                row["num_observations"] == 4
                for row in summary["per_physical_fragment"]
            )
        )
        physical = summary["sample_metric_statistics"]["physical_score"]
        self.assertEqual(physical["mean"], 8.5)
        self.assertEqual(physical["median"], 8.5)
        self.assertEqual(physical["p90"], 14.5)
        self.assertEqual(physical["max"], 16.0)
        self.assertEqual(summary["success_counts"]["pose_success_count"], 4)
        self.assertEqual(
            summary["success_counts"]["practical_surface_success_count"], 8
        )
        self.assertEqual(summary["success_counts"]["joint_success_count"], 4)
        self.assertEqual(
            [row["physical_score"] for row in summary["worst_samples"]],
            [16.0, 15.0, 14.0],
        )

    def test_writes_extended_reports_and_validation_histories(self):
        rows = _rows()
        summary = aggregate_production_metrics(rows)
        summary.update(
            {
                "runtime_seconds": 1.25,
                "max_batches": 4,
                "test_results_must_not_be_used_for_model_selection": False,
            }
        )
        state = {
            "epoch": 100,
            "batch_in_epoch": 4,
            "batch_step": 400,
            "optimizer_step": 100,
            "samples_seen": 1600,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "evaluations" / "step_000100"
            write_evaluation_report(evaluation, summary, rows)
            combined = root / "history.jsonl"
            write_validation_tracking(
                root,
                summary,
                state,
                combined_history_path=combined,
            )
            expected = {
                "summary.json",
                "metrics.json",
                "per_sample_metrics.csv",
                "per_frame.csv",
                "per_physical_fragment.csv",
                "per_scene.csv",
                "fragment_frame_matrix.csv",
            }
            self.assertEqual(
                {path.name for path in evaluation.iterdir()}, expected
            )
            metrics = json.loads((evaluation / "metrics.json").read_text())
            self.assertIn("sample_metric_statistics", metrics)
            latest = json.loads(
                (root / "latest_validation_metrics.json").read_text()
            )
            self.assertEqual(latest["training_state"]["optimizer_step"], 100)
            self.assertEqual(latest["num_samples"], 16)
            history = [
                json.loads(line)
                for line in (root / "validation_history.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(history[0]["record_type"], "validation")
            self.assertEqual(
                json.loads(combined.read_text())["record_type"], "validation"
            )
            with (root / "validation_history.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["evaluation_role"], "overfit_validation")
            with (evaluation / "fragment_frame_matrix.csv").open(
                newline=""
            ) as stream:
                matrix = list(csv.DictReader(stream))
            self.assertEqual(len(matrix), 4)
            self.assertIn("frame_000008_physical_score", matrix[0])

    def test_terminal_report_contains_requested_sections(self):
        summary = aggregate_production_metrics(_rows())
        rendered = format_validation_report(
            summary, {"optimizer_step": 100}, max_group_rows=4
        )
        self.assertIn("[VALIDATION step 000100]", rendered)
        self.assertIn("source=train", rendered)
        self.assertIn("role=overfit_validation", rendered)
        self.assertIn("frame_ids=[2, 4, 5, 8]", rendered)
        self.assertIn("worst samples:", rendered)
        self.assertIn("BY FRAME", rendered)
        self.assertIn("BY FRAGMENT", rendered)


if __name__ == "__main__":
    unittest.main()
