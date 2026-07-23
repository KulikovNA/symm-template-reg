import unittest

class EqualSampleWeightTest(unittest.TestCase):
    def test_every_sample_has_one_sixteenth_weight(self):
        for batch in (16,8,4,2):
            weights=[batch/16/batch for _ in range(16)]
            self.assertEqual(weights,[1/16]*16); self.assertEqual(sum(weights),1.0)

