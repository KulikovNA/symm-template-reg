import copy, unittest
import torch
from symm_template_reg.models.backbones.simple_point_encoder import SimplePointEncoder

class SharedTemplateGradientTest(unittest.TestCase):
    def test_expand_sums_all_sample_gradients(self):
        torch.manual_seed(2); a=SimplePointEncoder(embed_dim=16,hidden_dim=8,num_neighbors=3); b=copy.deepcopy(a)
        one=torch.randn(1,12,3); repeated=one.expand(4,-1,-1); mask=torch.ones(4,12,dtype=torch.bool)
        a(repeated,mask,samplewise_learned_ops=True).point_features.sum().backward()
        b(one,mask[:1],samplewise_learned_ops=True).point_features.expand(4,-1,-1).sum().backward()
        for left,right in zip(a.parameters(),b.parameters()):
            self.assertEqual(left.grad is None,right.grad is None)
            if left.grad is not None: self.assertTrue(torch.allclose(left.grad,right.grad,atol=1e-6,rtol=1e-6))
