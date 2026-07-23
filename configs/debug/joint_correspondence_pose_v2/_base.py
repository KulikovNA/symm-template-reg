"""Shared baseline: surface correspondences -> uniform Procrustes -> one pose."""

_base_ = ["../single_fragment/_base.py"]

experiment = dict(name="single_pose_uniform_correspondence_base_DO_NOT_RUN")

data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02.json",
    validation_manifest="same_as_train",
    expected_selected_samples=2,
    train_batch_size=1,
    validation_batch_size=1,
    drop_last=False,
    shuffle_train=True,
    shuffle_validation=False,
    num_workers=0,
    persistent_workers=False,
)

dataset = dict(random_seed=0)
augmentation = dict(enabled=False)

model = dict(
    _delete_=True,
    type="UniformCorrespondenceProcrustesReg",
    max_observed_tokens=256,
    max_template_tokens=512,
    weighting_mode="uniform",
    observed_encoder=dict(type="SimplePointEncoder", embed_dim=256, hidden_dim=128, num_neighbors=12, dropout=0.0),
    template_encoder=dict(type="SimplePointEncoder", embed_dim=256, hidden_dim=128, num_neighbors=12, dropout=0.0),
    interaction_transformer=dict(type="RegTRInteractionTransformer", embed_dim=256, num_heads=8, num_layers=4, feedforward_dim=512, dropout=0.0),
    dual_stream_geometry_encoder=dict(
        type="DualStreamGeometryEncoder",
        embed_dim=256,
        matching_geometric_embedding=dict(type="GeometricStructureEmbedding", embed_dim=256, num_neighbors=8, distance_scale_m=0.01),
        matching_ppf_embedding=dict(type="LocalPointPairFeatureEmbedding", embed_dim=256, num_neighbors=8, distance_scale_m=0.01),
    ),
    correspondence_head=dict(type="CorrespondenceHead", embed_dim=256, output_mode="soft_template_surface_matching", residual_scale_m=0.0),
    weighted_procrustes=dict(type="WeightedProcrustes", minimum_effective_points=3.0, rank_tolerance=1e-7, fail_on_degenerate=False),
    point_weight_head=None,
)

loss = dict(
    _delete_=True,
    joint_correspondence_pose=dict(
        enabled=True,
        correspondence_scale_m=0.002,
        rotation_scale_deg=2.0,
        translation_scale_m=0.002,
        alignment_scale_m=0.002,
        template_surface_scale_m=0.001,
        lambda_corr=1.0,
        lambda_rot=1.0,
        lambda_trans=1.0,
        lambda_align=0.5,
        lambda_surface=0.25,
        so2_samples=36,
    ),
)

train_budget = dict(mode="epochs", epochs=1500)
train = dict(
    max_optimizer_steps=None,
    max_epochs=1500,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    gradient_clip_norm=None,
    amp=False,
    eval_interval_optimizer_steps=0,
    debug_visualization_interval_optimizer_steps=0,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    log_interval_optimizer_steps=10,
    evaluate_before_training=True,
    visualize_before_training=True,
    early_stopping_patience_evals=0,
    save_best_only=True,
    save_periodic_checkpoints=False,
    save_final_checkpoint=False,
    best_metric="eval/physical_normalized_score",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/all_samples_pose_success_2deg_2mm",
    best_metric_tie_breaker_mode="max",
)

debug_visualization = dict(num_samples=2, single_fragment_layout=True, joint_correspondence_pose=True)
diagnostic_gates = dict(enabled=False)
plateau_detection = dict(enabled=True, action="warning_only")
target_leakage_policy = dict(forbid_detected=True, audit_path=None)

stage = dict(
    name="single_pose_uniform_correspondence",
    checkpoint_filename="best.pth",
    trainable_module_prefixes=None,
    readiness_thresholds=dict(),
)

debug_training_on_test_split = True
train_and_validation_use_same_samples = True
results_are_not_final_evaluation = True
