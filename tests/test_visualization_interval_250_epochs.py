import unittest
from symm_template_reg.visualization.prediction_debug import joint_visualization_epochs

class VisualizationScheduleTest(unittest.TestCase):
    def test_exact_schedule(self):
        self.assertEqual(joint_visualization_epochs(),[0,250,500,750,1000,1250,1500])
