from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.models.symmetry.groups import CyclicGroup, SO2Group
from symm_template_reg.models.symmetry.metadata import (
    SymmetryMetadata,
    load_symmetry_metadata,
    object_model_ids_match,
)
from symm_template_reg.models.symmetry.region_assignment import (
    active_symmetry_regions,
    assign_symmetry_regions,
    effective_group_from_regions,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "object_000004__scale_0p05.symmetry.example.json"


class TestSymmetryMetadata(unittest.TestCase):
    def test_optional_missing_sidecar_is_none_not_c1(self) -> None:
        self.assertIsNone(load_symmetry_metadata(None))
        self.assertIsNone(load_symmetry_metadata(ROOT / "does_not_exist.symmetry.json"))

    def test_example_loads_and_scaled_template_id_matches(self) -> None:
        metadata = load_symmetry_metadata(
            EXAMPLE,
            expected_object_model_id="object_000004__scale_0p05",
        )
        self.assertIsInstance(metadata, SymmetryMetadata)
        assert metadata is not None
        self.assertEqual(metadata.version, 1)
        self.assertEqual(metadata.coordinate_frame, "O")
        self.assertEqual(metadata.region_ids, ("band_00", "body_continuous"))
        self.assertIsInstance(metadata.regions[0].rotation_group, CyclicGroup)
        self.assertEqual(metadata.regions[0].rotation_group.order, 4)
        self.assertIsInstance(metadata.regions[1].rotation_group, SO2Group)
        self.assertTrue(metadata.has_continuous_symmetry)
        self.assertTrue(object_model_ids_match(metadata.object_model_id, "object_000004__scale_0p05"))
        self.assertEqual(SymmetryMetadata.from_dict(metadata.to_dict()).to_dict(), metadata.to_dict())

    def test_region_assignment_uses_configured_axis_and_intersection(self) -> None:
        metadata = load_symmetry_metadata(EXAMPLE)
        assert metadata is not None
        points = torch.tensor(
            [
                [100.0, -0.03, -7.0],
                [-20.0, 0.00, 5.0],
                [0.0, 0.20, 0.0],
            ]
        )
        membership = assign_symmetry_regions(points, metadata)
        self.assertEqual(tuple(membership.shape), (3, 2))
        self.assertTrue(torch.equal(membership[0], torch.tensor([True, False])))
        self.assertTrue(torch.equal(membership[1], torch.tensor([False, True])))
        self.assertFalse(bool(membership[2].any()))
        active = active_symmetry_regions(points, metadata)
        self.assertTrue(torch.equal(active, torch.tensor([True, True])))
        effective = effective_group_from_regions(metadata, active)
        self.assertIsInstance(effective, CyclicGroup)
        self.assertEqual(effective.order, 4)  # C4 ∩ SO2 = C4

    def test_present_metadata_with_no_active_regions_is_known_c1(self) -> None:
        metadata = load_symmetry_metadata(EXAMPLE)
        assert metadata is not None
        effective = effective_group_from_regions(metadata, torch.tensor([False, False]))
        self.assertEqual(effective, CyclicGroup(1))

    def test_existing_invalid_sidecar_raises_instead_of_disabling_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.symmetry.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "object_model_id": "object_bad",
                        "coordinate_frame": "O",
                        "axis": {
                            "name": "y",
                            "origin": [0, 0, 0],
                            "direction": [0, 0, 0],
                        },
                        "regions": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "direction"):
                load_symmetry_metadata(path)


if __name__ == "__main__":
    unittest.main()
