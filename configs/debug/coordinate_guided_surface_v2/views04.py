"""Four-view fine-only coordinate training; initialized model-only via CLI."""

_base_ = ["views02.py"]

experiment = dict(name="coordinate_guided_surface_v2_views04")
data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/coordinate_guided_surface/fragment0002_views04_shell_only.json",
    validation_manifest="same_as_train",
    expected_selected_samples=4,
    train_batch_size=4,
    validation_batch_size=4,
    shuffle_train=False,
    shuffle_validation=False,
)
train_budget = dict(mode="epochs", epochs=2500)
stage = dict(
    name="FOUR_VIEW_coordinate_guided_surface_v2_fine_only",
    initialization="model_only_via_cli_from_two_view_best",
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
train = dict(
    max_epochs=2500,
    gradient_accumulation_steps=1,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    amp=False,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    evaluate_before_training=True,
    visualize_before_training=True,
    best_metric="eval/active/worst_sample_projection_score",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/active/all_samples_gate_passed",
    best_metric_tie_breaker_mode="max",
)
active_coordinate_path = dict(
    enabled=True,
    exact_global=True,
    k16=True,
    candidate_k=16,
    projection_chunk_size=256,
    pose_solver="uniform_procrustes",
    expected_frames=(4, 5, 2, 8),
    inactive_namespaces=(
        "legacy_triangle",
        "legacy_pose_query",
        "regions",
        "ranking",
    ),
)
frozen_feature_cache = dict(
    enabled=True,
    require_passing_audit=True,
    audit_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/frozen_feature_cache_audit_20260721_codex_v4/frozen_feature_cache_audit.json",
    fallback_to_online=True,
    cache_dtype="float32",
)
four_view_stage_gate = dict(
    expected_frames=(4, 5, 2, 8),
    projected_correspondence_p95_mm=1.0,
    alignment_p95_mm=1.0,
    rotation_error_deg=1.0,
    translation_error_mm=1.0,
    rank=3,
    surface_membership_p95_mm=0.1,
    k16_exact_global_triangle_recall=0.995,
    k16_fallback_fraction=0.0,
    require_no_target_leakage=True,
    require_active_finite=True,
)
debug_visualization = dict(
    num_samples=4,
    coordinate_guided_primary=True,
    active_projected_pose_only=True,
    required_epochs=tuple(range(0, 2501, 250)),
    combined_world_pose_comparison=True,
)
