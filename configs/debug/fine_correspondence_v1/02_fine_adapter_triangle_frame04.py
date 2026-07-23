_base_ = ["01_fine_adapter_coordinate_control_frame04.py"]
experiment = dict(name="fine_adapter_triangle_frame04")
stage = dict(
    name="F2_fine_adapter_triangle_frame04",
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_candidate_triangle_head",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 3e-4,
        "correspondence_head.fine_candidate_triangle_head": 3e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 3e-4,
    },
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_local_fine=1.0, fine_coordinate_aux_weight=0.25,
))
fine_stage_gate = dict(
    valid_triangle_set_top1=0.95, valid_triangle_set_top4=0.995,
    candidate_recall=1.0, require_no_target_index_mismatch=True,
    require_no_collapse=True,
)
train = dict(
    best_metric="eval/valid_triangle_set_top1",
    best_metric_mode="max",
    best_metric_tie_breaker="eval/valid_triangle_set_top4",
    best_metric_tie_breaker_mode="max",
)
