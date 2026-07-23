import unittest
from symm_template_reg.models.symmetry.groups import parse_rotation_group
class TestGroups(unittest.TestCase):
    def test_supported_together(self):
        groups=[{"type":"C","order":1},{"type":"C","order":2},{"type":"C","order":4},{"type":"C","order":10},{"type":"SO2"}]
        self.assertEqual([parse_rotation_group(x).type for x in groups],["C","C","C","C","SO2"])

