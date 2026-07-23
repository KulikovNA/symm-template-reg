import unittest
import torch
from symm_template_reg.engine.static_geometry_cache import static_geometry_cache_key

class StaticGeometryKeyTest(unittest.TestCase):
    def test_coordinate_order_is_in_key(self):
        points=torch.arange(18,dtype=torch.float32).reshape(1,6,3); mask=torch.ones(1,6,dtype=torch.bool)
        def key(value): return static_geometry_cache_key(manifest_sha256="m",observed_points=value,observed_mask=mask,template_points=points,template_mask=mask,template_mesh_sha256="t",geometry_config={"k":8},point_selection_policy="shell_only")
        self.assertNotEqual(key(points),key(points.flip(1)))

