import unittest, torch
from symm_template_reg.models.pose import WeightedProcrustes

class PoseGradientTest(unittest.TestCase):
    def test_gradient_crosses_procrustes(self):
        q=torch.randn(1,16,3,requires_grad=True); angle=torch.tensor(.2); r=torch.tensor([[torch.cos(angle),-torch.sin(angle),0],[torch.sin(angle),torch.cos(angle),0],[0,0,1.]])
        p=q.detach()@r.T+torch.tensor([.1,.2,.3]); m=torch.ones(1,16,dtype=torch.bool); t=WeightedProcrustes()(q,p,torch.ones(1,16),m)
        loss=(t[:,:3,:3]-torch.eye(3)).square().sum()+t[:,:3,3].square().sum(); loss.backward()
        self.assertGreater(float(q.grad.norm()),0)
