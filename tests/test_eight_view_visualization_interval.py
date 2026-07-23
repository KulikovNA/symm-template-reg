import unittest
from pathlib import Path

from symm_template_reg.config import load_config
from symm_template_reg.visualization.prediction_debug import joint_visualization_epochs


class EightViewVisualizationIntervalTest(unittest.TestCase):
    def test_zero_through_5000_every_250(self):
        expected = list(range(0, 5001, 250))
        self.assertEqual(joint_visualization_epochs(5000, 250), expected)
        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/debug/coordinate_guided_surface_v2/views08.py")
        self.assertEqual(list(cfg["debug_visualization"]["required_epochs"]), expected)
        self.assertEqual(cfg["debug_visualization"]["num_samples"], 8)


if __name__ == "__main__":
    unittest.main()
