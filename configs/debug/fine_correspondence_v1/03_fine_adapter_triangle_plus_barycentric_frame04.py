_base_ = ["02_fine_adapter_triangle_frame04.py"]
experiment = dict(name="fine_adapter_triangle_plus_barycentric_frame04")
stage = dict(
    name="F3_fine_adapter_triangle_plus_barycentric_frame04",
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_candidate_triangle_head",
        "correspondence_head.fine_coordinate_auxiliary_head",
        "correspondence_head.barycentric_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 3e-4,
        "correspondence_head.fine_candidate_triangle_head": 3e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 3e-4,
        "correspondence_head.barycentric_head": 3e-4,
    },
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_local_fine=1.0, lambda_barycentric=1.0, lambda_corr_mean=1.0,
    fine_coordinate_aux_weight=0.1,
))
fine_stage_gate = dict(
    correspondence_p95_mm=0.5, barycentric_p95_mm=0.5,
    correspondence_rank=3, require_procrustes_valid=True,
)
train = dict(
    best_metric="eval/correspondence_p95_mm",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/barycentric_reconstruction_p95_mm",
    best_metric_tie_breaker_mode="min",
)
