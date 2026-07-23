import tempfile
import unittest
from pathlib import Path

from symm_template_reg.engine.frozen_feature_cache import build_frozen_feature_cache_key


class EightViewCacheKeyTest(unittest.TestCase):
    def test_key_contains_every_provenance_field(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, manifest = Path(directory) / "c.pth", Path(directory) / "m.json"
            checkpoint.write_bytes(b"checkpoint"); manifest.write_bytes(b"manifest")
            key, payload = build_frozen_feature_cache_key(
                frozen_module_state_sha256_value="frozen", initialization_checkpoint=checkpoint,
                manifest=manifest, template_sha256="template", sidecar_sha256="sidecar",
                point_selection_policy="shell_only", model_config={"frozen": True},
                dtype="float32", tensor_shapes={"features": (8, 32)},
                point_order_sha256_value="order",
            )
        self.assertEqual(len(key), 64)
        self.assertEqual(payload["template_sha256"], "template")
        self.assertEqual(payload["sidecar_sha256"], "sidecar")
        self.assertEqual(payload["tensor_shapes"]["features"], [8, 32])
        self.assertIn("initialization_checkpoint_sha256", payload)
        self.assertIn("manifest_file_sha256", payload)


if __name__ == "__main__":
    unittest.main()
