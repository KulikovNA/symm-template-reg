import tempfile
import unittest
from pathlib import Path

import torch

from symm_template_reg.models import build_model
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import LEGACY_MODULE_TOKENS
from tests.clean_v3_test_utils import tiny_clean_v3_config


class CleanV3CheckpointTest(unittest.TestCase):
    def test_checkpoint_has_no_legacy_keys(self):
        model = build_model(tiny_clean_v3_config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scratch.pth"
            torch.save({"model": model.state_dict()}, path)
            state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        forbidden = [key for key in state if any(token in key.lower() for token in LEGACY_MODULE_TOKENS)]
        self.assertEqual(forbidden, [])


if __name__ == "__main__": unittest.main()
