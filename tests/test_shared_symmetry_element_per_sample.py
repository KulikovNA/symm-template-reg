import inspect, unittest
from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import CleanCoordinatePoseLossV3
class TestSharedSymmetry(unittest.TestCase):
    def test_one_argmin_selects_all_components(self):
        source=inspect.getsource(CleanCoordinatePoseLossV3.forward); self.assertEqual(source.count("totals.detach().argmin"),1); self.assertIn("selected_normalized[name].append(normalized[name][selected])",source)

