from __future__ import annotations

import unittest

from symm_template_reg.engine.single_fragment import (
    manifest_content_sha256,
    validate_single_fragment_manifest_payload,
)


class SingleFragmentManifestTest(unittest.TestCase):
    def payload(self):
        samples = [
            {
                "sample_id": f"scene_000000/frame_{frame:06d}/fragment_0002",
                "scene_id": "scene_000000",
                "frame_id": frame,
                "fragment_id": 2,
                "fragment_mesh_sha256": "mesh",
                "fragment_num_faces": 2318,
                "T_W_from_C_available": True,
            }
            for frame in range(10)
        ]
        ids = [sample["sample_id"] for sample in samples]
        return {"samples": samples, "train_sample_ids": ids, "validation_sample_ids": ids}

    def test_one_physical_fragment_ten_different_frames(self):
        result = validate_single_fragment_manifest_payload(self.payload())
        self.assertEqual(result["fragment_id"], 2)
        self.assertEqual(result["frame_ids"], list(range(10)))
        self.assertEqual(result["fragment_mesh_sha256"], "mesh")

    def test_content_hash_ignores_hash_field(self):
        payload = self.payload()
        first = manifest_content_sha256(payload)
        payload["manifest_sha256"] = "ignored"
        self.assertEqual(first, manifest_content_sha256(payload))


if __name__ == "__main__":
    unittest.main()
