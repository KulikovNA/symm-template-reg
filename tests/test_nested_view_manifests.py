import unittest
from tools.build_joint_view_manifests import LADDER

class NestedManifestTest(unittest.TestCase):
    def test_frames_are_strictly_nested(self):
        values=[set(v) for v in LADDER.values()]
        self.assertTrue(values[0] < values[1] < values[2])
        self.assertEqual(tuple(LADDER.values())[-1],(4,5,2,8,0,1,6,9))
