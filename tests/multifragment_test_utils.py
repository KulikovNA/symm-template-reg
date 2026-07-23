from __future__ import annotations

from symm_template_reg.engine.multifragment_overfit import manifest_content_sha256


def samples():
    return [
        {
            "sample_id": f"scene_000000/frame_{frame:06d}/fragment_{fragment:04d}",
            "scene_id": "scene_000000", "fragment_id": fragment, "frame_id": frame,
            "fragment_mesh_sha256": f"mesh-{fragment}", "fragment_num_faces": 840,
            "shell_point_count": 128 + fragment + frame, "fracture_point_count": 0,
            "registration_point_selection": "shell_only", "T_W_from_C_available": True,
            "data_contract_errors": [],
        }
        for fragment in (0, 1, 2, 3) for frame in (2, 4, 5, 8)
    ]


def manifest():
    rows = samples(); ids = [row["sample_id"] for row in rows]
    value = {
        "debug_training_on_test_split": True,
        "train_and_validation_use_same_samples": True,
        "results_are_not_final_evaluation": True,
        "manifest_type": "four_fragments_four_views_overfit",
        "experiment_type": "four_fragments_four_views_overfit",
        "initialization_mode": "scratch", "pretrained_checkpoint": None,
        "train_sample_ids": ids, "validation_sample_ids": ids, "samples": rows,
    }
    value["manifest_sha256"] = manifest_content_sha256(value); return value


def metric_rows(bad_sample=None):
    output = []
    for row in samples():
        bad = row["sample_id"] == bad_sample
        output.append({
            **row,
            "exact_global_projected_correspondence_p95_mm": 3.0 if bad else 0.5,
            "exact_global_projection_alignment_p95_mm": 3.0 if bad else 0.5,
            "exact_global_projection_rotation_error_deg": 1.5 if bad else 0.1,
            "exact_global_projection_translation_error_mm": 0.7 if bad else 0.05,
            "exact_global_projection_rank": 3,
            "exact_global_surface_membership_p95_mm": 0.01,
            "k16_exact_global_triangle_recall": 1.0,
            "k16_fallback_fraction": 0.0,
        })
    return output

