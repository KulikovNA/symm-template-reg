"""Eight-view fine-only coordinate overfit; initialize model-only via CLI."""

_base_ = ["views04.py"]

experiment = dict(name="coordinate_guided_surface_v2_views08")

data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/coordinate_guided_surface/fragment0002_views08_shell_only.json",
    validation_manifest="same_as_train",
    expected_selected_samples=8,
    train_batch_size=8,
    validation_batch_size=8,
    effective_views_per_optimizer_step=8,
    shuffle_train=False,
    shuffle_validation=False,
)

train_budget = dict(mode="epochs", epochs=5000)

stage = dict(
    name="EIGHT_VIEW_coordinate_guided_surface_v2_fine_only",
    initialization="model_only_via_cli_from_four_view_best",
    strict_initialization=True,
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 1e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 1e-4,
    },
)

loss = dict(
    joint_surface_correspondence_pose_v3=dict(
        loss_reduction="per_sample_mean_then_batch_mean",
    ),
)

train = dict(
    max_epochs=5000,
    max_optimizer_steps=5000,
    gradient_accumulation_steps=1,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
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
    best_metric="eval/active/worst_sample_practical_score",
    best_metric_mode="min",
    best_metric_tie_breaker=None,
    best_metric_tie_breakers=(
        dict(metric="eval/active/practical_passed_sample_count", mode="max"),
        dict(metric="eval/active/strict_passed_sample_count", mode="max"),
        dict(metric="eval/active/exact_global/worst_correspondence_p95_mm", mode="min"),
    ),
)

active_coordinate_path = dict(
    enabled=True,
    exact_global=True,
    k16=True,
    candidate_k=16,
    projection_chunk_size=256,
    pose_solver="uniform_procrustes",
    expected_frames=(4, 5, 2, 8, 0, 1, 6, 9),
    dual_gates=True,
    inactive_namespaces=(
        "legacy_triangle",
        "legacy_barycentric",
        "legacy_pose_query",
        "regions",
        "ranking",
    ),
)

frozen_feature_cache = dict(
    enabled=True,
    require_passing_audit=True,
    verify_provenance=True,
    audit_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/eight_view_cache_audit_20260722_codex/frozen_feature_cache_audit.json",
    fallback_to_online=True,
    cache_dtype="float32",
)

debug_visualization = dict(
    num_samples=8,
    coordinate_guided_primary=True,
    active_projected_pose_only=True,
    required_epochs=tuple(range(0, 5001, 250)),
    combined_world_pose_comparison=True,
)

# Plateau is reported from the history after training; it must not stop this
# controlled first eight-view run or launch another architecture.
plateau_detection = None
