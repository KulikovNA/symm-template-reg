import unittest
import torch
from tools.audit_triangle_candidate_pipeline import _recall

class CandidatePipelineRecallTest(unittest.TestCase):
    def test_valid_set_recall_is_any_valid_triangle(self):
        ids=torch.tensor([[3,5]]); mask=torch.ones_like(ids,dtype=torch.bool)
        valid=torch.zeros(1,8,dtype=torch.bool); valid[0,5]=True
        self.assertTrue(bool(_recall(ids,mask,valid)[0]))
