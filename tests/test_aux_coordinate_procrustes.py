import unittest, torch
from symm_template_reg.models.pose import WeightedProcrustes


class AuxCoordinateProcrustesTest(unittest.TestCase):
    def test_perfect_aux_coordinates_recover_pose(self):
        torch.manual_seed(0); q = torch.randn(1, 20, 3); r = torch.eye(3); t = torch.tensor([.1, -.2, .3]); p = q @ r.T + t
        out = WeightedProcrustes().solve(q, p, torch.ones(1, 20), torch.ones(1, 20, dtype=torch.bool))
        self.assertTrue(torch.allclose(out["transform"][0, :3, 3], t, atol=1e-5)); self.assertTrue(bool(out["rank_valid"][0]))


if __name__ == "__main__": unittest.main()

