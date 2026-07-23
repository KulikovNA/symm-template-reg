import unittest

from symm_template_reg.evaluation.active_coordinate import ten_view_stage_gates


def row(frame, *, correspondence=1.5, alignment=1.5, rotation=.4, translation=.4):
    return {
        "sample_id": f"frame_{frame}", "frame_id": frame,
        "exact_global_projected_correspondence_p95_mm": correspondence,
        "exact_global_projection_alignment_p95_mm": alignment,
        "exact_global_projection_rotation_error_deg": rotation,
        "exact_global_projection_translation_error_mm": translation,
        "exact_global_projection_rank": 3,
        "exact_global_surface_membership_p95_mm": .05,
        "k16_exact_global_triangle_recall": .995,
        "k16_fallback_fraction": 0.0,
    }


class ThreeGateTest(unittest.TestCase):
    def test_gates_are_independent_and_one_bad_frame_blocks(self):
        rows = [row(frame) for frame in range(10)]
        gates = ten_view_stage_gates(rows)
        self.assertFalse(gates["strict_surface_gate"]["stage_passed"])
        self.assertTrue(gates["practical_surface_gate"]["stage_passed"])
        self.assertTrue(gates["pose_placement_gate"]["stage_passed"])
        rows[-1] = row(9, rotation=.6)
        gates = ten_view_stage_gates(rows)
        self.assertFalse(gates["practical_surface_gate"]["stage_passed"])
        self.assertFalse(gates["pose_placement_gate"]["stage_passed"])
        self.assertFalse(gates["stage_passed"])


if __name__ == "__main__": unittest.main()
