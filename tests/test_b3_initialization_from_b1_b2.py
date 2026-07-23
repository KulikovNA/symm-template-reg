import unittest
from symm_template_reg.config import load_config


class B3InitializationTest(unittest.TestCase):
    def test_config_declares_both_sources(self):
        cfg=load_config('configs/debug/correspondence_head_v4_local/03_triangle_plus_barycentric_frame04.py')
        sources=cfg['stage']['required_initialization']
        self.assertIn('B1 fine_query', sources); self.assertIn('B2 barycentric_head', sources)


if __name__ == "__main__": unittest.main()
