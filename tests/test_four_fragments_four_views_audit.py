import unittest
from symm_template_reg.engine.multifragment_overfit import validate_multifragment_manifest_payload
from tests.multifragment_test_utils import manifest
class TestAudit(unittest.TestCase):
    def test_exact_cartesian_product_passes(self): self.assertEqual(validate_multifragment_manifest_payload(manifest())["sample_count"], 16)

