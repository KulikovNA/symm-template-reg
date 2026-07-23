import unittest
from pathlib import Path
class FrameLabelsTest(unittest.TestCase):
    def test_generic_fields_are_primary_and_old_are_deprecated(self):
        text=Path('tools/recheck_coordinate_guided_surface.py').read_text();self.assertIn('"frame_correctness_solved"',text);self.assertIn('"frame_id"',text);self.assertIn('deprecated_compatibility_aliases',text);self.assertIn('two_view_coordinate_training',text)
