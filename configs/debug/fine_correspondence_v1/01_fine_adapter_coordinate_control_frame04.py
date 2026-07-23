_base_ = ["_base.py"]
experiment = dict(name="fine_adapter_coordinate_control_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
stage = dict(
    name="F1_fine_adapter_coordinate_control_frame04",
    policy="fine_adapter_only",
    strict_initialization=False,
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 3e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 3e-4,
    },
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    lambda_patch_ce=0.0, lambda_local_fine=0.0, lambda_barycentric=0.0,
    lambda_corr_mean=0.0, lambda_corr_tail=0.0, lambda_rot=0.0,
    lambda_trans=0.0, lambda_align_mean=0.0, lambda_align_tail=0.0,
    lambda_surface=0.0, lambda_local_rigidity=0.0, lambda_covariance=0.0,
    lambda_min_eigenvalue=0.0, lambda_patch_diversity=0.0,
    fine_coordinate_aux_weight=1.0,
))
fine_stage_gate = dict(
    coordinate_p95_mm=1.0, coordinate_rmse_mm=0.5,
    require_no_target_leakage=True, require_noncollapsed_feature_variance=True,
)
train = dict(
    best_metric="eval/aux_coordinate_p95_mm",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/aux_coordinate_rmse_mm",
    best_metric_tie_breaker_mode="min",
    rank_collapse_patience_evals=0,
)
