_base_ = ["_base.py"]
experiment = dict(name="v4_patch_classifier_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
stage = dict(name="patch_classifier_frame04", trainable_module_prefixes=("observed_encoder", "template_encoder", "interaction_transformer", "dual_stream_geometry_encoder", "correspondence_head.observed_query", "correspondence_head.template_key"))
train = dict(
    best_metric="eval/valid_patch_set_top1_accuracy",
    best_metric_mode="max",
    best_metric_tie_breaker="eval/valid_patch_set_top4_recall",
    best_metric_tie_breaker_mode="max",
    # Stage A does not train the local triangle/barycentric head.  Its
    # provisional correspondences may therefore be rank deficient and must
    # never trigger the full-correspondence diagnostic early stop.
    rank_collapse_patience_evals=0,
)
stage_gate_dependencies = dict(parameterization_capacity_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame04_final8/surface_parameterization_capacity_summary.json", parameterization_capacity_required_field="gt_injected_predicted_patch_capacity_passed", patch_only_gate=True, require_no_correspondence_collapse=False)
loss = dict(joint_surface_correspondence_pose_v3=dict(lambda_patch_ce=1.0, lambda_local_fine=0.0, lambda_barycentric=0.0, lambda_corr_mean=0.0, lambda_corr_tail=0.0, lambda_rot=0.0, lambda_trans=0.0, lambda_align_mean=0.0, lambda_align_tail=0.0, lambda_surface=0.0, lambda_local_rigidity=0.0, lambda_covariance=0.0, lambda_min_eigenvalue=0.0, lambda_patch_diversity=0.0))
