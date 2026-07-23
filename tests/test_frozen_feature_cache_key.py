import tempfile
import unittest
from pathlib import Path

from symm_template_reg.engine.frozen_feature_cache import build_frozen_feature_cache_key


class FrozenFeatureCacheKeyTest(unittest.TestCase):
    def test_manifest_or_order_changes_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / "a.pth"; manifest = root / "m.json"
            checkpoint.write_bytes(b"checkpoint"); manifest.write_bytes(b"manifest")
            kwargs = dict(
                frozen_module_state_sha256_value="f", initialization_checkpoint=checkpoint,
                manifest=manifest, template_sha256="t", sidecar_sha256="s",
                point_selection_policy="shell", model_config={}, dtype="float32",
                tensor_shapes={"x": (1, 2, 3)},
            )
            first, _ = build_frozen_feature_cache_key(**kwargs, point_order_sha256_value="one")
            second, _ = build_frozen_feature_cache_key(**kwargs, point_order_sha256_value="two")
            self.assertNotEqual(first, second)


if __name__ == "__main__": unittest.main()

