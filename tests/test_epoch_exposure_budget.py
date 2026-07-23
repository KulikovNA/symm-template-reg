import unittest
from symm_template_reg.engine.training_budget import resolve_training_budget, sample_exposure_statistics

class EpochExposureTest(unittest.TestCase):
    def test_nested_sizes_all_reach_1500(self):
        for n in (2,4,8):
            b=resolve_training_budget({"mode":"epochs","epochs":1500},selected_samples=n,batch_size=1,gradient_accumulation_steps=1,drop_last=False,configured_max_optimizer_steps=None,configured_max_epochs=1500)
            stats=sample_exposure_statistics({str(i):1500 for i in range(n)},target=b.target_sample_exposures)
            self.assertEqual((stats["sample_exposures_min"],stats["sample_exposures_max"],b.target_sample_exposures),(1500,1500,1500))
