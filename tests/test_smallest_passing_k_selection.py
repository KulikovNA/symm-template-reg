import unittest
from tools.recheck_coordinate_guided_surface import select_smallest_passing_candidate

class SmallestKTest(unittest.TestCase):
    def test_smallest_k_wins_not_fastest(self):
        gates={'aux_k64':{'passed':True},'aux_k16':{'passed':True},'aux_k32':{'passed':False}}
        self.assertEqual(select_smallest_passing_candidate(gates),(16,'aux_k16'))
