import tempfile
import unittest
from pathlib import Path

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
from tests.dataset_test_utils import build_dataset


class ShellOnlySelectionTest(unittest.TestCase):
    def test_fracture_rows_are_excluded_before_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = build_dataset(Path(directory) / "test")
            dataset = FragmentTemplateRegistrationDataset(
                root, min_observed_points=0, observed_policy="all_points",
                template_fine_points=4, template_coarse_points=2,
                registration_point_selection="shell_only",
            )
            sample = dataset[1]
            self.assertEqual(len(sample["observed"]["points_C"]), 4)
            self.assertTrue(sample["observed"]["surface_labels"].eq(0).all())
            self.assertEqual(len(sample["gt"]["points_O_corresponding"]), 4)

