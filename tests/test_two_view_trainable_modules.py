import unittest
from symm_template_reg.config import load_config
class TrainableTest(unittest.TestCase):
    def test_fine_only(self):
        c=load_config('configs/debug/coordinate_guided_surface_v2/views02.py');self.assertEqual(set(c['stage']['trainable_module_prefixes']),{'correspondence_head.fine_feature_adapter','correspondence_head.fine_coordinate_auxiliary_head'})
