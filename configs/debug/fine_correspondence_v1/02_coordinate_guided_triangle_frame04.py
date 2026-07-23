"""Learned fallback only if coordinate projection still fails after F1 refine."""

_base_ = ["01_fine_adapter_coordinate_control_frame04_refine.py"]
experiment = dict(name="coordinate_guided_triangle_fallback_frame04")
model = dict(correspondence_head=dict(
    fine_candidate_triangle_head=None,
    coordinate_guided_triangle_head=dict(
        type="CoordinateGuidedTriangleHead", embed_dim=256, hidden_dim=256,
    ),
    analytic_barycentric_projection=True,
    learned_barycentric_status="failed_frozen_feature_barycentric_capacity_on_frame04",
))
stage = dict(
    name="coordinate_guided_triangle_fallback_frame04",
    # The F1 checkpoint intentionally has no coordinate-guided triangle head.
    # Load all shared weights while leaving this new fallback head initialized.
    strict_initialization=False,
    trainable_module_prefixes=(
        "correspondence_head.coordinate_guided_triangle_head",
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.coordinate_guided_triangle_head": 3e-4,
        "correspondence_head.fine_feature_adapter": 3e-5,
        "correspondence_head.fine_coordinate_auxiliary_head": 3e-5,
    },
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_local_fine=1.0,
    lambda_barycentric=0.0,
    fine_coordinate_aux_weight=0.25,
    fine_coordinate_tail_weight=0.0,
))
train = dict(
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    best_metric="eval/valid_triangle_set_top1",
    best_metric_mode="max",
    best_metric_tie_breaker="eval/valid_triangle_set_top4",
    best_metric_tie_breaker_mode="max",
)
