_base_ = ["02_procrustes_base_4views.py"]

experiment = dict(name="correspondence_pose_v1_03_hybrid_bounded_residual_4views")
model = dict(
    base_pose_source="procrustes_plus_direct_residual",
    base_pose_head=dict(
        output_mode="bounded_correction",
        max_rotation_correction_deg=5.0,
        max_translation_correction_m=0.003,
    ),
)
loss = dict(
    correspondence_pose_loss_weight=1.0,
    direct_vs_correspondence_pose_consistency_weight=0.1,
    hybrid_direct_residual=dict(
        enabled=True,
        regularization_weight=0.01,
        translation_scale=100.0,
    ),
)
diagnostic_gates = dict(
    residual_static_codebook=dict(
        enabled=True,
        max_static_fraction=0.25,
        minimum_nonidentity_rotation_deg=0.1,
        minimum_nonidentity_translation_mm=0.1,
    ),
    residual_bound_saturation=dict(enabled=True, max_saturation_fraction=0.25),
)
stage = dict(
    name="hybrid_bounded_residual_4views",
    checkpoint_filename="best_hybrid_bounded.pth",
    readiness_thresholds=dict(
        final_not_worse_than_correspondence=True,
        top1_pose_success_2deg_2mm_not_worse=True,
        max_rotation_correction_deg=5.0,
        max_translation_correction_m=0.003,
    ),
)
