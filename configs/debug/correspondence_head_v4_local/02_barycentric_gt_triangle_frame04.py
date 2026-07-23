_base_ = ["_base_local.py"]

experiment = dict(name="v4_local_b2_barycentric_gt_triangle_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
model = dict(correspondence_head=dict(teacher_force_exact_triangle=True))
stage = dict(
    name="b2_barycentric_gt_triangle_frame04",
    local_substage="B2",
    required_initialization="Stage A frame 4 best checkpoint via --init-checkpoint",
    trainable_module_prefixes=("correspondence_head.barycentric_head",),
)
train = dict(
    best_metric="eval/barycentric_reconstruction_p95_mm",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/correspondence_p95_mm",
    best_metric_tie_breaker_mode="min",
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_patch_ce=0.0,
    lambda_local_fine=0.0,
    lambda_barycentric=1.0,
    lambda_corr_mean=1.0,
    lambda_corr_tail=0.0,
    lambda_rot=0.0,
    lambda_trans=0.0,
    lambda_align_mean=0.0,
    lambda_align_tail=0.0,
    lambda_surface=0.0,
    lambda_local_rigidity=0.0,
    lambda_covariance=0.0,
    lambda_min_eigenvalue=0.0,
    lambda_patch_diversity=0.0,
))
stage_gate_dependencies = dict(local_substage="B2")
