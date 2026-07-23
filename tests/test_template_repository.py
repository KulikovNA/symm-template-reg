from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.datasets.template_repository import TemplateRepository, load_ply

from tests.dataset_test_utils import write_ascii_tetrahedron


class TemplateRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.models = Path(self.temporary.name) / "models"
        self.models.mkdir()
        write_ascii_tetrahedron(self.models / "object_000004__scale_0p05.ply")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cache_reuses_one_mesh_load_and_computes_normals(self) -> None:
        repository = TemplateRepository(self.models, fine_points=3, coarse_points=2)
        first = repository.get("object_000004__scale_0p05")
        second = repository.get("object_000004__scale_0p05")
        base_alias = repository.get("object_000004")
        self.assertIs(first, second)
        self.assertIs(first, base_alias)
        self.assertEqual(repository.load_count("object_000004__scale_0p05"), 1)
        self.assertEqual(repository.cache_size, 1)  # aliases share one tensor dictionary
        self.assertEqual(len(repository), 1)
        self.assertEqual(tuple(first["normals_O"].shape), (4, 3))
        self.assertTrue(torch.isfinite(first["normals_O"]).all())
        self.assertEqual(tuple(first["fine_points_O"].shape), (3, 3))
        self.assertEqual(tuple(first["coarse_points_O"].shape), (2, 3))
        self.assertIsNone(first["symmetry_metadata"])

    def test_template_feature_cache_is_separate(self) -> None:
        repository = TemplateRepository(self.models, fine_points=4, coarse_points=2)
        feature = torch.randn(4, 8)
        repository.cache_feature("object_000004__scale_0p05", "encoder-v1", feature)
        self.assertIs(
            repository.get_cached_feature("object_000004__scale_0p05", "encoder-v1"),
            feature,
        )
        self.assertIs(repository.get_cached_feature("object_000004", "encoder-v1"), feature)

    def test_binary_little_endian_ply_fallback(self) -> None:
        path = self.models / "binary.ply"
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            "element vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        ).encode("ascii")
        payload = b"".join(
            struct.pack("<fff", *point)
            for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        ) + struct.pack("<Biii", 3, 0, 1, 2)
        path.write_bytes(header + payload)
        mesh = load_ply(path)
        self.assertEqual(mesh["format"], "binary_little_endian")
        self.assertEqual(mesh["points"].shape, (3, 3))
        self.assertEqual(mesh["faces"].tolist(), [[0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
