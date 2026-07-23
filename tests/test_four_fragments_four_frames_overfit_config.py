import os
import unittest
from pathlib import Path

from symm_template_reg.config import load_config
from symm_template_reg.datasets import SplitDirectoryFragmentDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/debug/four_fragments_four_frames_overfit.py"


class FourByFourOverfitConfigTest(unittest.TestCase):
    def test_config_is_portable_and_uses_one_update_per_epoch(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["model"]["type"], "CoordinateGuidedSurfaceRegistrationV3"
        )
        self.assertEqual(config["data"]["validation_split"], "train")
        self.assertEqual(config["data"]["train_batch_size"], 4)
        self.assertEqual(config["train"]["gradient_accumulation_steps"], 4)
        self.assertEqual(
            config["validation"]["evaluation_role"], "overfit_validation"
        )
        self.assertEqual(config["data"]["train"]["selector"]["frame_ids"], (2, 4, 5, 8))
        self.assertEqual(config["data"]["train"]["selector"]["fragment_ids"], (0, 1, 2, 3))
        self.assertNotIn("/home/", CONFIG.read_text(encoding="utf-8"))

    def test_real_selector_contains_exactly_sixteen_observations(self):
        root = os.environ.get("FRAG_DATASET_ROOT")
        if not root:
            self.skipTest("FRAG_DATASET_ROOT is not set")
        config = load_config(CONFIG)
        for role in ("train", "validation"):
            dataset_config = config["data"][role]
            dataset = SplitDirectoryFragmentDataset(
                root,
                split="train",
                min_num_faces=dataset_config["min_num_faces"],
                selector=dataset_config["selector"],
                point_sampling=dataset_config.get("point_sampling"),
                boundary_augmentation={"enabled": False},
            )
            self.assertEqual(len(dataset), 16)


if __name__ == "__main__":
    unittest.main()
