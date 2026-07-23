_base_ = ["_base_local.py"]

experiment = dict(name="v4_local_b4_full_correspondence_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
stage = dict(
    name="b4_full_local_correspondence_frame04",
    local_substage="B4",
    required_initialization="passed B3 frame 4 checkpoint",
    trainable_module_prefixes=(
        "correspondence_head.fine_query",
        "correspondence_head.barycentric_head",
    ),
)
train = dict(best_metric="eval/physical_normalized_score", best_metric_mode="min")
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_patch_ce=0.0,
    lambda_local_fine=1.0,
    lambda_barycentric=1.0,
    lambda_corr_mean=1.0,
    lambda_corr_tail=0.5,
    lambda_rot=0.25,
    lambda_trans=0.25,
    lambda_align_mean=0.25,
    lambda_align_tail=0.25,
    lambda_surface=0.0,
    lambda_local_rigidity=0.10,
    lambda_covariance=0.25,
    lambda_min_eigenvalue=0.25,
    lambda_patch_diversity=0.0,
))
stage_gate_dependencies = dict(
    local_substage="B4",
    physical_thresholds=dict(
        correspondence_p95_mm=0.5,
        visible_alignment_p95_mm=2.0,
        rotation_error_deg=0.5,
        translation_total_mm=0.5,
    ),
)
