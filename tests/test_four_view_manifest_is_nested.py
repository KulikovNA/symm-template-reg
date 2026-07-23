import unittest

from symm_template_reg.engine.view_ladder import subset_view_manifest


class FourViewNestedTest(unittest.TestCase):
    def test_frame_sets_are_strictly_nested(self):
        samples = [{"frame_id": value, "sample_id": str(value)} for value in (4, 5, 2, 8)]
        source = {"samples": samples, "manifest_sha256": "source"}
        two = subset_view_manifest(source, (4, 8))
        four = subset_view_manifest(source, (4, 5, 2, 8))
        self.assertLess(
            {sample["frame_id"] for sample in two["samples"]},
            {sample["frame_id"] for sample in four["samples"]},
        )


if __name__ == "__main__": unittest.main()

