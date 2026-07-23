import unittest
from pathlib import Path

class MetricNamesTest(unittest.TestCase):
    def test_pipeline_declares_distinct_recall_names(self):
        text=Path('tools/audit_triangle_candidate_pipeline.py').read_text()
        for name in ('predicted_patch_valid_set_top4_recall','predicted_patch_union_triangle_recall','post_dedup_triangle_recall','post_truncation_triangle_recall','qaux_shortlist_triangle_recall'):
            self.assertIn(name,text)
