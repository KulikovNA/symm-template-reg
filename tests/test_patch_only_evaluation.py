import types
import unittest
import json
import tempfile
from pathlib import Path

import torch

from symm_template_reg.engine.metrics import batch_pose_metric_rows
from symm_template_reg.engine.overfit_trainer import _write_evaluation
from symm_template_reg.config import load_config
from tests.test_fragment_symmetry_targets import metadata


class PatchOnlyEvaluationTest(unittest.TestCase):
    def test_stage_a_selects_best_checkpoint_by_patch_metrics(self):
        for frame in ("04", "08"):
            config = load_config(
                Path(
                    f"configs/debug/correspondence_head_v4/"
                    f"00_patch_classifier_frame{frame}.py"
                )
            )
            self.assertEqual(
                config["train"]["best_metric"],
                "eval/valid_patch_set_top1_accuracy",
            )
            self.assertEqual(config["train"]["best_metric_mode"], "max")
            self.assertEqual(
                config["train"]["best_metric_tie_breaker"],
                "eval/valid_patch_set_top4_recall",
            )

    def test_zero_pose_weights_do_not_instantiate_legacy_joint_loss(self):
        pose = torch.eye(4)[None]
        points = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
        )
        faces = torch.tensor([[0, 1, 2]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        barycentric = torch.eye(3)[None]
        coarse_logits = torch.zeros((1, 3, 1))
        auxiliary = {
            "selected_triangle_ids": torch.zeros((1, 3), dtype=torch.long),
            "predicted_barycentric": barycentric,
            "patch_points_O": points[:, :1],
            "coarse_patch_logits": coarse_logits,
            "fine_local_logits": torch.zeros((1, 3, 1)),
            "candidate_triangle_ids": torch.zeros((1, 3, 1), dtype=torch.long),
            "all_candidate_triangle_ids": torch.zeros((1, 1, 1), dtype=torch.long),
            "selected_topk_patch_ids": torch.zeros((1, 3, 1), dtype=torch.long),
        }
        prediction = types.SimpleNamespace(
            pose_hypotheses=pose[:, None],
            pose_logits=torch.zeros((1, 1)),
            active_region_logits=None,
            observed_region_logits=None,
            base_pose=pose,
            context_diagnostics=None,
            residual_transforms=None,
            base_correction_transform=None,
            correspondence_pose=pose,
            correspondence_points_O=points,
            correspondence_confidence=torch.full((1, 3), 1.0 / 3.0),
            correspondence_logits=coarse_logits,
            correspondence_auxiliary=auxiliary,
            correspondence_pose_diagnostics={
                "rank": torch.tensor([2]),
                "source_rank": torch.tensor([2]),
                "target_rank": torch.tensor([2]),
                "rank_valid": torch.tensor([False]),
                "valid_solution": torch.tensor([False]),
                "determinant": torch.ones(1),
                "orthogonality_error": torch.zeros(1),
                "reflection_corrected": torch.zeros(1, dtype=torch.bool),
            },
            observed_valid_mask=valid,
            weighting_mode="uniform",
        )
        batch = {
            "sample_id": ["patch-only"],
            "scene_id": ["scene"],
            "fragment_id": torch.tensor([0]),
            "frame_id": torch.tensor([4]),
            "template_symmetry_metadata": [metadata()],
            "template": {"points_O": points, "valid_mask": valid},
            "template_mesh_vertices_O": [points[0]],
            "template_mesh_faces": [faces],
            "observed": {"points_C": points, "valid_mask": valid},
            "gt": {
                "T_C_from_O": pose,
                "effective_symmetry_group": [{"type": "C", "order": 1}],
                "active_symmetry_regions": None,
                "active_symmetry_regions_valid_mask": None,
                "points_O_corresponding": points,
            },
            "meta": [
                {
                    "fragment_mesh": {
                        "num_faces": 1,
                        "surface_area_m2": 0.5,
                        "bbox_diagonal_m": 2.0**0.5,
                    }
                }
            ],
        }
        config = {
            "enabled": True,
            "lambda_patch_ce": 1.0,
            "lambda_local_fine": 0.0,
            "lambda_corr_mean": 0.0,
            "lambda_corr_tail": 0.0,
            "lambda_rot": 0.0,
            "lambda_trans": 0.0,
            "lambda_align_mean": 0.0,
            "lambda_align_tail": 0.0,
            "lambda_surface": 0.0,
            "lambda_local_rigidity": 0.0,
            "lambda_covariance": 0.0,
            "lambda_min_eigenvalue": 0.0,
            "lambda_patch_diversity": 0.0,
        }
        row = batch_pose_metric_rows(
            prediction, batch, joint_loss_config=config
        )[0]
        self.assertEqual(row["coarse_patch_top1_accuracy"], 1.0)
        self.assertEqual(row["coarse_patch_top4_recall"], 1.0)

    def test_patch_confusion_matrix_is_serialized_as_a_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write_evaluation(
                run_dir,
                0,
                {"eval/coarse_patch_top1_accuracy": 0.5},
                [
                    {
                        "sample_id": "sample",
                        "patch_confusion_matrix": [[1, 2], [3, 4]],
                        "observed_region_confusion_true_0_pred_0": 7,
                    }
                ],
            )
            patch_payload = json.loads(
                (run_dir / "patch_confusion_matrix.json").read_text()
            )
            region_payload = json.loads(
                (run_dir / "region_confusion_matrix.json").read_text()
            )
        self.assertEqual(
            patch_payload["metrics"]["sample"], [[1, 2], [3, 4]]
        )
        self.assertEqual(
            region_payload["metrics"][
                "observed_region_confusion_true_0_pred_0"
            ],
            7.0,
        )


if __name__ == "__main__":
    unittest.main()
