import unittest
from pathlib import Path
from symm_template_reg.visualization.multifragment_debug import required_multifragment_visualization_names
from symm_template_reg.visualization.prediction_debug import _joint_sample_directory
class TestViz(unittest.TestCase):
    def test_contract_names(self):
        names=required_multifragment_visualization_names(); self.assertEqual(len(names["per_frame"]),4); self.assertEqual(len(names["per_fragment"]),4); self.assertIn("q_aux_vs_global_projection.ply",names["worst_sample"])
    def test_same_frame_fragments_have_distinct_directories(self):
        root=Path("debug")
        first=_joint_sample_directory(root,{"frame_id":5,"fragment_id":0},{"multifragment_layout":True})
        second=_joint_sample_directory(root,{"frame_id":5,"fragment_id":1},{"multifragment_layout":True})
        self.assertNotEqual(first,second)
        self.assertEqual(first,Path("debug/per_view/frame_000005/fragment_0000"))
