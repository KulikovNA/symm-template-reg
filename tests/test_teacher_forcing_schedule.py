import unittest
from symm_template_reg.models.heads.surface_constrained_correspondence_head_v2 import SurfaceConstrainedCorrespondenceHeadV2


class TeacherScheduleTest(unittest.TestCase):
    def test_linear_one_to_zero(self):
        head = SurfaceConstrainedCorrespondenceHeadV2(embed_dim=8, num_patches=4, teacher_forcing_initial_probability=1., teacher_forcing_final_probability=0., teacher_forcing_start_epoch=10, teacher_forcing_anneal_epochs=20)
        head.set_epoch(10); self.assertEqual(head.teacher_forcing_probability, 1.)
        head.set_epoch(20); self.assertAlmostEqual(head.teacher_forcing_probability, .5)
        head.set_epoch(30); self.assertEqual(head.teacher_forcing_probability, 0.)

    def test_decay_waits_for_top4_gate(self):
        head = SurfaceConstrainedCorrespondenceHeadV2(embed_dim=8, num_patches=4, teacher_forcing_initial_probability=1., teacher_forcing_final_probability=0., teacher_forcing_start_epoch=0, teacher_forcing_anneal_epochs=10, teacher_forcing_decay_min_top4_recall=.995)
        head.set_epoch(10); self.assertEqual(head.teacher_forcing_probability, 1.)
        head.set_patch_recall(.995); head.set_epoch(5)
        self.assertEqual(head.teacher_forcing_probability, 1.)
        head.set_epoch(10); self.assertAlmostEqual(head.teacher_forcing_probability, .5)
