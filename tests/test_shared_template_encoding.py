import unittest
import torch
from symm_template_reg.models.backbones.simple_point_encoder import SimplePointEncoder

class SharedTemplateEncodingTest(unittest.TestCase):
    def test_shared_output_equals_repeated_samplewise_output(self):
        torch.manual_seed(2); model=SimplePointEncoder(embed_dim=16,hidden_dim=8,num_neighbors=3)
        one=torch.randn(1,12,3); repeated=one.expand(4,-1,-1); mask=torch.ones(4,12,dtype=torch.bool)
        old=model(repeated,mask,samplewise_learned_ops=True)
        new=model(one,mask[:1],samplewise_learned_ops=True)
        self.assertTrue(torch.equal(old.point_features,new.point_features.expand_as(old.point_features)))

