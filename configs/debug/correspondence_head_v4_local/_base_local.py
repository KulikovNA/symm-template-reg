"""Shared exact-target contract for isolated local correspondence substages."""

_base_ = ["../correspondence_head_v4/_base.py"]

experiment = dict(name="v4_local_substage_DO_NOT_RUN")

model = dict(
    correspondence_head=dict(
        teacher_forcing_initial_probability=1.0,
        teacher_forcing_final_probability=1.0,
        teacher_forcing_during_evaluation=True,
        teacher_forcing_select_shared_symmetry_element=True,
        deduplicate_local_candidates=True,
        inject_all_valid_triangles=True,
        teacher_force_exact_triangle=False,
        triangle_target_tolerance_m=0.00015,
        candidate_geometry_weight=1.0,
        max_local_candidate_total=32,
        sort_owned_faces_by_distance=True,
    )
)

loss = dict(
    joint_surface_correspondence_pose_v3=dict(
        patch_target_mode="multi_valid_patch_set",
        triangle_target_mode="multi_valid_patch_set",
        triangle_target_tolerance_m=0.00015,
        require_exact_triangle_candidate=True,
        use_teacher_forcing_shared_symmetry_element=True,
        conditional_covariance=True,
        covariance_minimum_eigenvalue_ratio=0.10,
    )
)

train = dict(
    rank_collapse_patience_evals=0,
    max_epochs=1000,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
)

stage_gate_dependencies = dict(
    require_patch_gate=False,
    require_no_correspondence_collapse=False,
    require_no_excessive_attention_diffusion=False,
    local_triangle_target_tolerance_m=0.00015,
    parameterization_capacity_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame04_final8/surface_parameterization_capacity_summary.json",
    parameterization_capacity_required_field="gt_injected_predicted_patch_capacity_passed",
)
