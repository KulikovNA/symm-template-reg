"""Clean ten-view active-only registration trained deterministically from scratch."""

_base_ = ["../coordinate_guided_surface_v2/views08.py"]

debug_training_on_test_split = True
train_and_validation_use_same_samples = True
results_are_not_final_evaluation = True
initialization_mode = "scratch"
pretrained_checkpoint = None
seed = 0

experiment = dict(
    name="coordinate_guided_surface_v3_views10_scratch_full",
    debug_training_on_test_split=True,
    train_and_validation_use_same_samples=True,
    results_are_not_final_evaluation=True,
    initialization_mode="scratch",
    pretrained_checkpoint=None,
)

data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/coordinate_guided_surface/fragment0002_views10_shell_only.json",
    validation_manifest="same_as_train",
    expected_selected_samples=10,
    train_batch_size=10,
    # Exact/K16 evaluation is independent per sample.  Micro-batching it avoids
    # retaining a large projection allocator reserve before the batch-10 train step.
    validation_batch_size=2,
    effective_views_per_optimizer_step=10,
    shuffle_train=False,
    shuffle_validation=False,
)
dataset = dict(random_seed=0, registration_point_selection="shell_only")
augmentations = dict(enabled=False)

model = dict(
    _delete_=True,
    type="CoordinateGuidedSurfaceRegistrationV3",
    embed_dim=256,
    max_observed_tokens=256,
    max_template_tokens=512,
    final_coordinate_initialization_std=1e-3,
    observed_encoder=dict(
        type="SimplePointEncoder", embed_dim=256, hidden_dim=128,
        num_neighbors=12, dropout=0.0,
    ),
    template_encoder=dict(
        type="SimplePointEncoder", embed_dim=256, hidden_dim=128,
        num_neighbors=12, dropout=0.0,
    ),
    interaction_transformer=dict(
        type="RegTRInteractionTransformer", embed_dim=256, num_heads=8,
        num_layers=4, feedforward_dim=512, dropout=0.0,
    ),
    dual_stream_geometry_encoder=dict(
        type="DualStreamGeometryEncoder", embed_dim=256, matching_only=True,
        matching_geometric_embedding=dict(
            type="GeometricStructureEmbedding", embed_dim=256,
            num_neighbors=8, distance_scale_m=0.01,
        ),
        matching_ppf_embedding=dict(
            type="LocalPointPairFeatureEmbedding", embed_dim=256,
            num_neighbors=8, distance_scale_m=0.01,
        ),
    ),
    fine_feature_adapter=dict(
        type="FineLocalCorrespondenceFeatureAdapter", embed_dim=256,
        knn_scales=(8, 16, 32), observed_only=True,
    ),
    canonical_coordinate_head=dict(
        type="FineCanonicalCoordinateAuxiliaryHead", embed_dim=256,
        hidden_dim=256,
    ),
    weighted_procrustes=dict(
        type="WeightedProcrustes", minimum_effective_points=3.0,
        rank_tolerance=1e-7, fail_on_degenerate=False,
    ),
)

loss = dict(
    joint_correspondence_pose=dict(enabled=False),
    joint_surface_correspondence_pose_v3=dict(
        _delete_=True,
        enabled=True,
        clean_active_only=True,
        coordinate_mean_weight=1.0,
        coordinate_tail_weight=0.5,
        pose_rotation_weight=0.25,
        pose_translation_weight=0.25,
        alignment_weight=0.25,
        rotation_scale_deg=1.0,
        translation_scale_m=0.001,
        alignment_scale_m=0.001,
        warmup_epochs=250,
        tail_fraction=0.10,
        so2_samples=36,
        loss_reduction="per_sample_mean_then_batch_mean",
    ),
)

train_budget = dict(mode="epochs", epochs=6000)
train = dict(
    max_epochs=6000,
    max_optimizer_steps=6000,
    gradient_accumulation_steps=1,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="linear_warmup_constant", warmup_optimizer_steps=100),
    gradient_clip_norm=1.0,
    amp=False,
    eval_interval_epochs=50,
    debug_visualization_interval_epochs=250,
    evaluate_before_training=True,
    visualize_before_training=True,
    save_best_only=True,
    save_periodic_checkpoints=False,
    save_final_checkpoint=False,
    early_stopping_patience_evals=0,
    min_sample_exposures_before_early_stop=3000,
    best_metric="eval/active/worst_sample_score",
    best_metric_mode="min",
    best_metric_tie_breaker=None,
    best_metric_tie_breakers=(
        dict(metric="eval/active/pose_ready_sample_count", mode="max"),
        dict(metric="eval/active/practical_surface_passed_sample_count", mode="max"),
        dict(metric="eval/active/strict_surface_passed_sample_count", mode="max"),
        dict(metric="eval/active/exact_global/worst_correspondence_p95_mm", mode="min"),
    ),
)

stage = dict(
    _delete_=True,
    name="TEN_VIEW_clean_v3_scratch_full",
    initialization="scratch",
    strict_initialization=True,
    trainable_module_prefixes=None,
    prefix_learning_rates={
        "observed_encoder": 1e-4,
        "template_encoder": 1e-4,
        "interaction_transformer": 1e-4,
        "dual_stream_geometry_encoder": 1e-4,
        "dense_observed_fine_projection": 3e-4,
        "fine_template_projection": 3e-4,
        "template_context_projection": 3e-4,
        "fine_feature_adapter": 3e-4,
        "canonical_coordinate_head": 3e-4,
    },
)

frozen_feature_cache = dict(enabled=False, require_passing_audit=False)

active_coordinate_path = dict(
    enabled=True,
    clean_active_only=True,
    exact_global=True,
    k16=True,
    candidate_k=16,
    projection_chunk_size=256,
    pose_solver="uniform_procrustes",
    expected_frames=tuple(range(10)),
    ten_view_gates=True,
)

debug_visualization = dict(
    num_samples=10,
    single_fragment_layout=True,
    joint_correspondence_pose=True,
    coordinate_guided_primary=True,
    active_projected_pose_only=True,
    required_epochs=tuple(range(0, 6001, 250)),
    combined_world_pose_comparison=True,
    legacy_outputs=False,
)

conditioning_audit = dict(required_epochs=(0, 500, 1000, 3000, "best", "final"))
plateau_detection = None
