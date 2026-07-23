"""SurfaceHeadV2 with exact triangle-barycentric output and V3 loss."""

_base_ = ["_base.py"]

model = dict(
    correspondence_head=dict(
        _delete_=True,
        type="SurfaceConstrainedCorrespondenceHeadV2",
        embed_dim=256,
        num_patches=64,
        top_k_patches=4,
        local_candidates=32,
        fine_mode="triangle_barycentric",
        temperature=1.0,
        initial_temperature=1.0,
        final_temperature=1.0,
        anneal_epochs=0,
    ),
)

loss = dict(
    joint_surface_correspondence_pose_v3=dict(
        enabled=True,
        correspondence_scale_m=0.002,
        rotation_scale_deg=2.0,
        translation_scale_m=0.002,
        alignment_scale_m=0.002,
        template_surface_scale_m=0.001,
        tail_fraction=0.10,
        local_rigidity_k=8,
        lambda_patch_ce=1.0,
        lambda_local_fine=1.0,
        lambda_corr_mean=1.0,
        lambda_corr_tail=1.0,
        lambda_rot=1.0,
        lambda_trans=1.0,
        lambda_align_mean=0.5,
        lambda_align_tail=0.5,
        lambda_surface=0.5,
        lambda_local_rigidity=0.25,
        so2_samples=36,
    ),
)
