import unittest
from symm_template_reg.config import load_config
class TwoViewConfigTest(unittest.TestCase):
    def test_shared_batch_and_primary_modes(self):
        c=load_config('configs/debug/coordinate_guided_surface_v2/views02.py');self.assertEqual(c['data']['train_batch_size'],2);self.assertEqual(c['data']['validation_batch_size'],2);self.assertEqual(c['coordinate_guided_surface_v2']['evaluate_modes'],('exact_global','aux_guided_global_topk'));self.assertFalse(c['train']['amp'])
