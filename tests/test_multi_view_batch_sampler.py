import unittest

from symm_template_reg.datasets.multi_view_batch_sampler import MultiViewBatchSampler


class MultiViewBatchSamplerTest(unittest.TestCase):
    def test_groups_same_fragment_and_chunks_views(self):
        samples = [
            {"scene_id": "s", "fragment_id": 2, "fragment_mesh_sha256": "mesh", "frame_id": index}
            for index in range(10)
        ]
        sampler = MultiViewBatchSampler(samples, views_per_group=4, shuffle=False)
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])


if __name__ == "__main__": unittest.main()
