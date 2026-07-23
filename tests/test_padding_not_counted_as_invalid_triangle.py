import unittest, torch

class PaddingSemanticsTest(unittest.TestCase):
    def test_padding_and_invalid_id_are_separate(self):
        ids=torch.tensor([[1,-1]]); mask=torch.tensor([[True,False]])
        padded=float((~mask).float().mean()); invalid=float(((ids<0)&mask).float().mean())
        self.assertEqual(padded,.5); self.assertEqual(invalid,0.)
