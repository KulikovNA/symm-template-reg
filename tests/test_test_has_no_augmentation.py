import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets import SplitDirectoryFragmentDataset
from tests.dataset_test_utils import build_dataset


class TestAugmentationIsolationTest(unittest.TestCase):
    def test_test_split_is_unaugmented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val", "test"):
                build_dataset(root / split, with_sidecar=True)
            dataset = SplitDirectoryFragmentDataset(
                root,
                split="test",
                min_num_faces=0,
                min_observed_shell_points=0,
                max_observed_shell_points=16,
                template_fine_points=4,
                template_coarse_points=2,
                max_samples=1,
            )
            self.assertEqual(
                dataset[0]["augmentation_metadata"]["augmentation_mode"], "none"
            )
            with self.assertRaisesRegex(ValueError, "forbidden"):
                SplitDirectoryFragmentDataset(
                    root,
                    split="test",
                    min_num_faces=0,
                    min_observed_shell_points=0,
                    boundary_augmentation={"enabled": True},
                )


if __name__ == "__main__":
    unittest.main()
