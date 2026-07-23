_base_ = ["01_k1_direct_equal_exposure.py"]

experiment = dict(name="conditioned_v2_05_k8_bounded_residual")
model = dict(
    residual_pose_head=dict(
        num_hypotheses=8,
        residual_bounds=dict(max_rotation_deg=15.0, max_translation_m=0.01),
    )
)
loss = dict(
    conditioned_pose_loss=dict(
        base_pose_weight=1.0,
        best_residual_pose_weight=1.0,
        residual_regularization_weight=0.1,
    )
)
stage = dict(
    name="conditioned_v2_k8_bounded",
    checkpoint_filename="best_k8_bounded.pth",
    requires_k1_base_gate=True,
    readiness_thresholds=dict(
        base_top1_pose_success_5deg_5mm=0.9,
        base_rotation_response_ratio=0.5,
        base_pose_static_fraction=0.0,
        oracle_pose_success_5deg_5mm=0.9,
        residual_query_static_fraction=0.25,
        query_static_codebook_score=0.25,
        require_target_exposures_or_valid_early_stop=True,
    ),
)
