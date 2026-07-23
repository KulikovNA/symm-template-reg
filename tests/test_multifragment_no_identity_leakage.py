import inspect, unittest
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import CoordinateGuidedSurfaceRegistrationV3
class TestIdentity(unittest.TestCase):
    def test_forward_does_not_read_ids(self):
        source=inspect.getsource(CoordinateGuidedSurfaceRegistrationV3.forward); self.assertNotIn('batch["fragment_id"]', source); self.assertNotIn('batch["frame_id"]', source)

