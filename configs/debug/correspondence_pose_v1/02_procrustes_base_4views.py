_base_ = ["_base.py"]

experiment = dict(name="correspondence_pose_v1_02_procrustes_base_4views")
model = dict(correspondence_only=False, base_pose_source="weighted_procrustes")
loss = dict(
    conditioned_pose_loss=dict(
        base_pose_weight=1.0,
        best_residual_pose_weight=0.0,
        residual_regularization_weight=0.0,
    ),
    correspondence_pose_loss_weight=0.0,
    direct_vs_correspondence_pose_consistency_weight=0.0,
)
stage = dict(
    name="procrustes_base_4views",
    checkpoint_filename="best_procrustes_base.pth",
    readiness_thresholds=dict(
        top1_pose_success_5deg_5mm=0.9,
        correspondence_pose_success_5deg_5mm=0.9,
        rotation_response_ratio=0.5,
        base_pose_static_fraction=0.0,
        world_axis_spread_deg=10.0,
        require_no_target_leakage=True,
    ),
)
