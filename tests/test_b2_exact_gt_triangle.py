import unittest
from symm_template_reg.config import load_config


class B2ExactTriangleTest(unittest.TestCase):
    def test_b2_forces_exact_triangle_and_only_bary_head_trains(self):
        cfg=load_config('configs/debug/correspondence_head_v4_local/02_barycentric_gt_triangle_frame04.py')
        self.assertTrue(cfg['model']['correspondence_head']['teacher_force_exact_triangle'])
        self.assertEqual(tuple(cfg['stage']['trainable_module_prefixes']), ('correspondence_head.barycentric_head',))


if __name__ == "__main__": unittest.main()
