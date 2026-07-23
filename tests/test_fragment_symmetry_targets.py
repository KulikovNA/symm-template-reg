from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch

from symm_template_reg.datasets import fragment_template_dataset
from symm_template_reg.models.losses import SymmetryPoseLoss
from symm_template_reg.models.symmetry.groups import CyclicGroup, SO2Group
from symm_template_reg.models.symmetry.hypothesis_expander import (
    place_fragment_for_hypothesis,
)
from symm_template_reg.models.symmetry.metadata import SymmetryMetadata
from symm_template_reg.models.symmetry.targets import (
    build_fragment_symmetry_targets,
)
from symm_template_reg.visualization.symmetry_debug import audit_fragment_annotations


DATASET_ROOT = Path(
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)
TEMPLATE = DATASET_ROOT / "models" / "object_000004__scale_0p05.ply"


def metadata() -> SymmetryMetadata:
    return SymmetryMetadata.from_dict(
        {
            "version": 1,
            "object_model_id": "object_000004",
            "coordinate_frame": "O",
            "axis": {"name": "y", "origin": [0, 0, 0], "direction": [0, 1, 0]},
            "regions": [
                {
                    "region_id": "band_00",
                    "y_min_m": -0.05,
                    "y_max_m": -0.036,
                    "rotation_group": {"type": "SO2"},
                },
                {
                    "region_id": "band_01",
                    "y_min_m": -0.036,
                    "y_max_m": 0.004,
                    "rotation_group": {"type": "C", "order": 10},
                },
                {
                    "region_id": "band_02",
                    "y_min_m": 0.004,
                    "y_max_m": 0.019,
                    "rotation_group": {"type": "SO2"},
                },
                {
                    "region_id": "band_03",
                    "y_min_m": 0.019,
                    "y_max_m": 0.05,
                    "rotation_group": {"type": "C", "order": 4},
                },
            ],
        }
    )


def triangle_at_y(y: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[-0.01, y, 0.0], [0.01, y, 0.0], [0.0, y, 0.02]]),
        torch.tensor([[0, 1, 2]]),
    )


class FragmentSymmetryTargetTest(unittest.TestCase):
    def test_exact_c10_c4_and_intersection_hypothesis_counts(self) -> None:
        sidecar = metadata()
        points10, faces10 = triangle_at_y(-0.02)
        target10 = build_fragment_symmetry_targets(
            points10, sidecar, fragment_faces=faces10
        )
        self.assertEqual(target10.effective_group, CyclicGroup(10))
        self.assertEqual(target10.equivalent_pose_set.num_hypotheses, 10)

        points4, faces4 = triangle_at_y(0.03)
        target4 = build_fragment_symmetry_targets(
            points4, sidecar, fragment_faces=faces4
        )
        self.assertEqual(target4.effective_group, CyclicGroup(4))
        self.assertEqual(target4.equivalent_pose_set.num_hypotheses, 4)

        mixed_points = torch.cat((points10, points4), dim=0)
        mixed_faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        mixed = build_fragment_symmetry_targets(
            mixed_points, sidecar, fragment_faces=mixed_faces
        )
        self.assertEqual(mixed.effective_group, CyclicGroup(2))
        self.assertEqual(mixed.equivalent_pose_set.num_hypotheses, 2)

    def test_unused_boundary_vertex_does_not_activate_mesh_region(self) -> None:
        points, faces = triangle_at_y(-0.02)
        # This vertex lies exactly on the following band boundary but belongs to
        # no face, so it must remain diagnostic-only for mesh activation.
        points = torch.cat((points, torch.tensor([[0.0, 0.019, 0.0]])), dim=0)
        targets = build_fragment_symmetry_targets(
            points, metadata(), fragment_faces=faces
        )
        self.assertEqual(targets.active_regions.tolist(), [False, True, False, False])
        self.assertEqual(targets.region_point_counts.tolist(), [0, 3, 0, 1])

    def test_so2_remains_continuous_and_loss_consumes_shared_target(self) -> None:
        points, faces = triangle_at_y(-0.043)
        targets = build_fragment_symmetry_targets(
            points, metadata(), fragment_faces=faces, so2_num_samples=7
        )
        self.assertEqual(targets.effective_group, SO2Group())
        self.assertEqual(targets.training_target_type, "continuous_analytic")
        self.assertFalse(targets.equivalent_pose_set.exhaustive)
        self.assertEqual(targets.equivalent_pose_set.num_hypotheses, 7)
        loss = SymmetryPoseLoss()(
            torch.eye(4), torch.eye(4), symmetry_targets=targets
        )
        self.assertTrue(torch.isfinite(loss))
        built_inside_loss = SymmetryPoseLoss()(
            torch.eye(4),
            torch.eye(4),
            fragment_points_O=points,
            fragment_faces=faces,
            symmetry_metadata=metadata(),
            symmetry_target_kwargs={"so2_num_samples": 7},
        )
        self.assertTrue(torch.isfinite(built_inside_loss))

    def test_identity_placement_and_rigid_hypotheses(self) -> None:
        points, faces = triangle_at_y(0.03)
        targets = build_fragment_symmetry_targets(
            points, metadata(), fragment_faces=faces
        )
        placed = place_fragment_for_hypothesis(
            points, torch.eye(4), targets.equivalent_poses
        )
        torch.testing.assert_close(placed[0], points)
        determinants = torch.linalg.det(targets.equivalent_poses[:, :3, :3])
        torch.testing.assert_close(determinants, torch.ones_like(determinants))
        self.assertFalse(torch.allclose(targets.equivalent_poses[-1], targets.equivalent_poses[0]))
        reference = torch.cdist(points, points)
        for hypothesis in placed:
            torch.testing.assert_close(torch.cdist(hypothesis, hypothesis), reference)

    def test_dataset_imports_the_shared_fragment_builder(self) -> None:
        source = inspect.getsource(fragment_template_dataset._symmetry_targets)
        self.assertIn("build_fragment_symmetry_targets", source)


@unittest.skipUnless(TEMPLATE.is_file(), "real annotated fragment fixture unavailable")
class RealFragmentAnnotationContractTest(unittest.TestCase):
    def test_three_scenes_resolve_twelve_annotated_meshes_in_F_to_O(self) -> None:
        audit, fragments = audit_fragment_annotations(
            DATASET_ROOT,
            scene_ids=("scene_000000", "scene_000001", "scene_000002"),
            object_model_id="object_000004",
            template_path=TEMPLATE,
        )
        self.assertEqual(audit["totals"]["selected_entries"], 12)
        self.assertEqual(audit["totals"]["contract_valid"], 12)
        self.assertEqual(audit["totals"]["annotation_contract_errors"], 0)
        self.assertEqual(len(fragments), 12)
        for fragment in fragments:
            self.assertEqual(fragment.coordinate_frame, "F")
            self.assertEqual(fragment.face_labels.shape, (len(fragment.faces),))
            self.assertGreater(int((fragment.face_labels == 0).sum()), 0)
            expected = (
                fragment.points_F.astype("float64") @ fragment.T_O_from_F[:3, :3].T
                + fragment.T_O_from_F[:3, 3]
            )
            torch.testing.assert_close(
                torch.from_numpy(fragment.points_O),
                torch.from_numpy(expected).to(torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
