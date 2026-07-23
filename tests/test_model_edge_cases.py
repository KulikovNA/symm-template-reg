from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace

import torch

from symm_template_reg.models import build_model
from symm_template_reg.models.backbones import SimplePointEncoder
from symm_template_reg.models.geometry import (
    GeometricStructureEmbedding,
    PointPairFeatures,
)
from symm_template_reg.models.geometry.point_ops import knn_indices
from symm_template_reg.models.losses import SymmetryPoseLoss
from symm_template_reg.models.matching import HungarianPoseAssigner
from symm_template_reg.models.pose import (
    PoseHypotheses,
    axis_angle_to_matrix,
    symmetry_aware_pose_errors,
)
from symm_template_reg.models.structures import RegistrationPrediction
from symm_template_reg.models.symmetry import (
    CyclicGroup,
    SO2Group,
    equivalent_gt_poses,
    symmetry_transforms,
)
from tests.test_model_forward import tiny_model_config, variable_batch


def _prediction_contract() -> RegistrationPrediction:
    batch, queries, observed, template = 2, 3, 4, 5
    poses = torch.eye(4).repeat(batch, queries, 1, 1)
    return RegistrationPrediction(
        pose_hypotheses=poses,
        pose_logits=torch.zeros(batch, queries),
        pose_uncertainty=torch.zeros(batch, queries, 6),
        observed_overlap_logits=torch.zeros(batch, observed),
        template_visibility_logits=torch.zeros(batch, template),
        correspondence_points_O=torch.zeros(batch, observed, 3),
        correspondence_confidence=torch.zeros(batch, observed),
        observed_region_logits=torch.zeros(batch, observed, 2),
        active_region_logits=torch.zeros(batch, 2),
        insufficient_information_logit=torch.zeros(batch, 1),
        observed_valid_mask=torch.ones(batch, observed, dtype=torch.bool),
        template_valid_mask=torch.ones(batch, template, dtype=torch.bool),
        auxiliary_outputs=[
            {
                "pose_hypotheses": poses.clone(),
                "pose_logits": torch.zeros(batch, queries),
                "pose_uncertainty": torch.zeros(batch, queries, 6),
                "valid_mask": torch.ones(batch, queries, dtype=torch.bool),
            }
        ],
        symmetry_available=torch.tensor([False, True]),
    )


class AssignmentEdgeCasesTest(unittest.TestCase):
    def test_rectangular_8_by_36_assignment_has_known_non_greedy_optimum(self) -> None:
        cost = torch.full((8, 36), 100.0)
        cost[0, 0] = 1.0
        cost[0, 1] = 2.0
        cost[1, 0] = 1.0
        cost[1, 1] = 100.0
        for row in range(2, 8):
            cost[row, row] = 0.0

        prediction, target = HungarianPoseAssigner()(cost)

        self.assertEqual(prediction.tolist(), list(range(8)))
        self.assertEqual(target.tolist(), [1, 0, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(set(target.tolist())), 8)
        self.assertAlmostEqual(float(cost[prediction, target].sum()), 3.0)


class PointGeometryEdgeCasesTest(unittest.TestCase):
    def test_single_point_knn_encoder_and_geometric_embedding_are_finite(self) -> None:
        points = torch.tensor([[[0.2, -0.1, 0.7]]])
        mask = torch.ones(1, 1, dtype=torch.bool)

        indices = knn_indices(points, points, mask, k=12)
        encoded = SimplePointEncoder(
            embed_dim=16, hidden_dim=8, num_neighbors=12
        ).eval()(points, mask)
        embedded = GeometricStructureEmbedding(
            embed_dim=16, num_neighbors=8
        ).eval()(points, mask)

        self.assertEqual(indices.tolist(), [[[0]]])
        self.assertEqual(tuple(encoded.point_features.shape), (1, 1, 16))
        self.assertEqual(tuple(embedded.shape), (1, 1, 16))
        self.assertTrue(torch.isfinite(encoded.point_features).all())
        self.assertTrue(torch.isfinite(encoded.global_feature).all())
        self.assertTrue(torch.isfinite(embedded).all())

    def test_padding_coordinates_do_not_change_valid_local_features(self) -> None:
        torch.manual_seed(31)
        points = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.1, -0.2, 0.3],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [-1.0, -2.0, -3.0],
                ]
            ]
        )
        changed_padding = points.clone()
        changed_padding[:, 2:] = torch.tensor(
            [[[1000.0, -2000.0, 3000.0], [-4000.0, 5000.0, -6000.0], [7.0, 70.0, 700.0]]]
        )
        mask = torch.tensor([[True, True, False, False, False]])
        encoder = SimplePointEncoder(
            embed_dim=16, hidden_dim=8, num_neighbors=4
        ).eval()
        embedding = GeometricStructureEmbedding(
            embed_dim=16, num_neighbors=4
        ).eval()

        with torch.no_grad():
            first = encoder(points, mask)
            second = encoder(changed_padding, mask)
            first_geometry = embedding(points, mask)
            second_geometry = embedding(changed_padding, mask)

        self.assertTrue(
            torch.allclose(first.point_features[:, :2], second.point_features[:, :2], atol=1e-6)
        )
        self.assertTrue(torch.allclose(first.global_feature, second.global_feature, atol=1e-6))
        self.assertTrue(
            torch.allclose(first_geometry[:, :2], second_geometry[:, :2], atol=1e-6)
        )

    def test_ppf_parallel_and_antiparallel_backward_is_finite(self) -> None:
        source_points = torch.zeros(1, 2, 3, requires_grad=True)
        target_points = torch.tensor(
            [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], requires_grad=True
        )
        source_normals = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], requires_grad=True
        )
        target_normals = torch.tensor(
            [[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], requires_grad=True
        )

        features = PointPairFeatures()(
            source_points, target_points, source_normals, target_normals
        )
        features.sum().backward()

        self.assertTrue(torch.isfinite(features).all())
        for value in (source_points, target_points, source_normals, target_normals):
            self.assertIsNotNone(value.grad)
            assert value.grad is not None
            self.assertTrue(torch.isfinite(value.grad).all())


class PoseEdgeCasesTest(unittest.TestCase):
    def test_zero_axis_angle_backward_is_finite(self) -> None:
        axis_angle = torch.zeros(4, 3, requires_grad=True)
        weights = torch.arange(9, dtype=axis_angle.dtype).reshape(3, 3)
        loss = (axis_angle_to_matrix(axis_angle) * weights).sum()
        loss.backward()

        self.assertIsNotNone(axis_angle.grad)
        assert axis_angle.grad is not None
        self.assertTrue(torch.isfinite(axis_angle.grad).all())

    def test_pose_query_probabilities_are_independent_sigmoids(self) -> None:
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        logits = torch.tensor([[-1.0, 0.0, 1.0]])
        hypotheses = PoseHypotheses(poses, logits)

        self.assertTrue(torch.allclose(hypotheses.probabilities, torch.sigmoid(logits)))
        self.assertAlmostEqual(float(hypotheses.probabilities[0, 1]), 0.5)
        self.assertGreater(float(hypotheses.probabilities.sum()), 1.0)

    def test_prediction_to_dtype_preserves_boolean_contract_fields(self) -> None:
        prediction = _prediction_contract()
        prediction.validate()

        converted = prediction.to(dtype=torch.float64)

        self.assertEqual(converted.pose_hypotheses.dtype, torch.float64)
        self.assertEqual(converted.observed_valid_mask.dtype, torch.bool)
        self.assertEqual(converted.template_valid_mask.dtype, torch.bool)
        self.assertEqual(converted.symmetry_available.dtype, torch.bool)
        assert converted.auxiliary_outputs is not None
        self.assertEqual(converted.auxiliary_outputs[0]["pose_logits"].dtype, torch.float64)
        self.assertEqual(converted.auxiliary_outputs[0]["valid_mask"].dtype, torch.bool)
        converted.validate()

    def test_prediction_validate_rejects_disconnected_observed_batch(self) -> None:
        prediction = _prediction_contract()
        malformed = replace(
            prediction,
            observed_overlap_logits=torch.zeros(1, 4),
            correspondence_points_O=torch.zeros(1, 4, 3),
            correspondence_confidence=torch.zeros(1, 4),
            observed_region_logits=torch.zeros(1, 4, 2),
            observed_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        )

        with self.assertRaises(ValueError):
            malformed.validate()


class SymmetryQueryBatchEdgeCasesTest(unittest.TestCase):
    def test_c4_metrics_and_loss_broadcast_gt_over_pose_queries(self) -> None:
        target = torch.eye(4).repeat(2, 1, 1)
        target[1, :3, 3] = torch.tensor([0.2, -0.1, 0.4])
        equivalents = equivalent_gt_poses(
            target, CyclicGroup(4), axis=[0.0, 0.0, 1.0]
        )
        predicted = torch.stack(
            (equivalents[:, 0], equivalents[:, 1], equivalents[:, 2]), dim=1
        )

        metrics = symmetry_aware_pose_errors(
            predicted,
            target,
            CyclicGroup(4),
            axis=[0.0, 0.0, 1.0],
        )
        loss = SymmetryPoseLoss()(
            predicted, target, equivalent_gt_poses=equivalents
        )

        self.assertEqual(tuple(metrics["combined"].shape), (2, 3))
        self.assertEqual(tuple(metrics["matched_index"].shape), (2, 3))
        self.assertTrue(torch.isfinite(metrics["combined"]).all())
        self.assertLess(float(metrics["combined"].max()), 1e-4)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)

    def test_so2_metrics_and_loss_broadcast_gt_over_pose_queries(self) -> None:
        axis = torch.tensor([0.0, 1.0, 0.0])
        origin = torch.tensor([0.3, -0.2, 0.1])
        target = torch.eye(4).repeat(2, 1, 1)
        target[1, :3, 3] = torch.tensor([-0.1, 0.25, 0.6])
        twists = symmetry_transforms(
            SO2Group(), axis, origin, so2_num_samples=4
        )[:3]
        predicted = torch.matmul(target.unsqueeze(1), twists.unsqueeze(0))

        metrics = symmetry_aware_pose_errors(
            predicted,
            target,
            SO2Group(),
            axis=axis,
            origin=origin,
        )
        loss = SymmetryPoseLoss()(
            predicted,
            target,
            continuous_axis_O=axis,
            continuous_origin_O=origin,
        )

        self.assertEqual(tuple(metrics["combined"].shape), (2, 3))
        self.assertEqual(tuple(metrics["matched_index"].shape), (2, 3))
        self.assertTrue(torch.isfinite(metrics["combined"]).all())
        self.assertLess(float(metrics["combined"].max()), 1e-4)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 1e-3)


class OptionalSymmetryHeadEdgeCasesTest(unittest.TestCase):
    def test_model_accepts_no_symmetry_head_and_false_boolean_metadata(self) -> None:
        config = deepcopy(tiny_model_config())
        config["symmetry_head"] = None
        model = build_model(config).eval()
        batch = variable_batch()
        batch["meta"] = [False, False]

        with torch.no_grad():
            prediction = model(batch)

        self.assertIsNone(prediction.observed_region_logits)
        self.assertIsNone(prediction.active_region_logits)
        self.assertEqual(prediction.symmetry_available.tolist(), [False, False])


if __name__ == "__main__":
    unittest.main()
