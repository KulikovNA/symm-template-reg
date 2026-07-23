import unittest
class TestEffectiveBatch(unittest.TestCase):
    def test_modes(self):
        for batch,accumulation in ((16,1),(8,2),(4,4),(2,8)): self.assertEqual(batch*accumulation,16)

