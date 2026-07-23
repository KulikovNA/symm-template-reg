import unittest
import torch
from symm_template_reg.engine.static_geometry_cache import build_static_geometry

class StaticGeometryEquivalenceTest(unittest.TestCase):
    def test_online_and_cached_structures_match(self):
        torch.manual_seed(1); points=torch.randn(2,20,3); mask=torch.ones(2,20,dtype=torch.bool)
        a=build_static_geometry(points,mask,points.clone(),mask.clone(),observed_tokens=8,template_tokens=8)
        b=build_static_geometry(points,mask,points.clone(),mask.clone(),observed_tokens=8,template_tokens=8)
        for name in a: self.assertTrue(torch.equal(a[name],b[name]),name)

