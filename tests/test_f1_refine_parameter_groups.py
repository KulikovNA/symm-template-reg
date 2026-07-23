import unittest
from symm_template_reg.config import load_config


class F1RefineGroupsTest(unittest.TestCase):
    def test_only_adapter_and_coordinate_head_train(self):
        cfg=load_config('configs/debug/fine_correspondence_v1/01_fine_adapter_coordinate_control_frame04_refine.py')
        self.assertEqual(set(cfg['stage']['trainable_module_prefixes']), {'correspondence_head.fine_feature_adapter','correspondence_head.fine_coordinate_auxiliary_head'})
        self.assertEqual(cfg['train']['optimizer']['lr'],1e-4); self.assertEqual(cfg['train']['max_epochs'],500)


if __name__ == "__main__": unittest.main()

