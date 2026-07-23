import inspect,unittest
from symm_template_reg.visualization.prediction_debug import _export_joint_prediction_visualizations

class TrainingGalleryGroupTest(unittest.TestCase):
    def test_gt_group_contract_is_explicit(self):
        source=inspect.getsource(_export_joint_prediction_visualizations)
        self.assertIn('group_source=gt_training_symmetry_target',source)
        self.assertIn('effective_group=group',source)
