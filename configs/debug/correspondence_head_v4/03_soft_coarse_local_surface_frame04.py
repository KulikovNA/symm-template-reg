_base_ = ["_base.py"]
experiment = dict(name="v4_soft_coarse_local_surface_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
model = dict(correspondence_head=dict(_delete_=True, type="SoftCoarseLocalSurfaceCorrespondenceHead", embed_dim=256, nearest_triangle_candidates=32, coarse_temperature=1.0, local_temperature=1.0, max_coarse_to_surface_distance_m=0.05))
stage = dict(name="soft_coarse_local_surface_frame04", trainable_module_prefixes=None)
stage_gate_dependencies = dict(parameterization_capacity_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame04_final8/surface_parameterization_capacity_summary.json", parameterization_capacity_required_field="legacy_soft_local_surface_capacity_passed")
loss = dict(joint_surface_correspondence_pose_v3=dict(lambda_patch_ce=0.25, lambda_local_fine=1.0, lambda_corr_mean=1.0, lambda_corr_tail=1.0, lambda_rot=1.0, lambda_trans=1.0, lambda_align_mean=0.5, lambda_align_tail=0.5, lambda_surface=0.0, lambda_local_rigidity=0.25, lambda_covariance=0.5, lambda_min_eigenvalue=0.5, lambda_patch_diversity=0.25))
