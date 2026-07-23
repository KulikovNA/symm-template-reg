import unittest

from symm_template_reg.engine.view_ladder import subset_view_manifest
from tests.test_single_frame_manifest import source_manifest


class EightViewManifestNestedTest(unittest.TestCase):
    def test_four_view_ids_are_a_strict_subset(self):
        source = source_manifest()
        four = subset_view_manifest(source, [4, 5, 2, 8])
        eight = subset_view_manifest(source, [4, 5, 2, 8, 0, 1, 6, 9])
        self.assertLess(set(four["train_sample_ids"]), set(eight["train_sample_ids"]))
        self.assertEqual(eight["train_sample_ids"][:4], four["train_sample_ids"])


if __name__ == "__main__":
    unittest.main()
