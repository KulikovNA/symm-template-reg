import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.engine.frozen_feature_cache import FrozenFeatureCache


class EightViewCacheEquivalenceTest(unittest.TestCase):
    def test_all_eight_payloads_round_trip_exactly(self):
        payload = {f"frame_{frame}": {"features": torch.randn(frame + 2, 3)} for frame in range(8)}
        with tempfile.TemporaryDirectory() as directory:
            cache = FrozenFeatureCache(Path(directory), "key")
            cache.store(payload, {"views": 8})
            loaded = cache.load()["payload"]
        self.assertEqual(set(loaded), set(payload))
        for key in payload:
            torch.testing.assert_close(loaded[key]["features"], payload[key]["features"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
