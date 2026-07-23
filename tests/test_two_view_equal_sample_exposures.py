import unittest
from symm_template_reg.config import load_config
class ExposureTest(unittest.TestCase):
    def test_both_samples_each_epoch_1500_times(self):
        c=load_config('configs/debug/coordinate_guided_surface_v2/views02.py');self.assertEqual(c['train_budget'],{'mode':'epochs','epochs':1500});self.assertEqual(c['data']['expected_selected_samples'],2);self.assertEqual(c['data']['train_batch_size'],2)
