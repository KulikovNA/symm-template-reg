import unittest
from symm_template_reg.config import load_config


class B1LossConfigTest(unittest.TestCase):
    def test_only_local_triangle_loss_is_enabled(self):
        cfg=load_config('configs/debug/correspondence_head_v4_local/01_triangle_classifier_gt_patch_frame04.py')
        loss=cfg['loss']['joint_surface_correspondence_pose_v3']
        enabled={k:v for k,v in loss.items() if k.startswith('lambda_') and float(v)!=0}
        self.assertEqual(enabled, {'lambda_local_fine':1.0})
        head=cfg['model']['correspondence_head']
        self.assertEqual(head['max_local_candidate_total'], 32)
        self.assertTrue(head['deduplicate_local_candidates'])
        self.assertTrue(head['inject_all_valid_triangles'])
        self.assertTrue(head['teacher_forcing_select_shared_symmetry_element'])


if __name__ == "__main__": unittest.main()
