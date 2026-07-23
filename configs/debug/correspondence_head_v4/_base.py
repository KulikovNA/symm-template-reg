"""Common shell-only, one-view settings for the staged V4 diagnosis."""

_base_ = ["../joint_correspondence_pose_v3/_base.py"]

experiment = dict(name="correspondence_head_v4_DO_NOT_RUN")

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
    joint_correspondence_pose=dict(enabled=False),
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
        lambda_barycentric=1.0,
        lambda_corr_mean=1.0,
        lambda_corr_tail=1.0,
        lambda_rot=1.0,
        lambda_trans=1.0,
        lambda_align_mean=0.5,
        lambda_align_tail=0.5,
        lambda_surface=0.0,
        lambda_local_rigidity=0.25,
        lambda_covariance=0.5,
        lambda_min_eigenvalue=0.5,
        lambda_patch_diversity=0.25,
        min_eigenvalue_m2=1e-6,
        so2_samples=36,
        patch_target_mode="multi_valid_patch_set",
    ),
)

train = dict(
    max_epochs=1000,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    evaluate_before_training=True,
    visualize_before_training=True,
    rank_collapse_patience_evals=3,
)

stage_gate_dependencies = dict(
    require_parameterization_capacity=True,
    require_patch_gate=True,
    require_no_correspondence_collapse=True,
    require_no_excessive_attention_diffusion=False,
)
