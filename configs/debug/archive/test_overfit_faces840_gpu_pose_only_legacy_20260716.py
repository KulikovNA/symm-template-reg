# Legacy configuration preserved to reproduce the pose-only run from 2026-07-16.
# The active/observed region losses are disabled and query classification uses
# the old one-positive/K-1-negative scheme. Do not use this as the new baseline.

_base_ = ["../../symm_template_reg_baseline.py"]

debug_training_on_test_split = True
train_and_validation_use_same_samples = True
results_are_not_final_evaluation = True

_work_dir_root = "/home/nikita/disser/fragment-template-registration-lab/work_dirs"
_dataset_root = (
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)

experiment = dict(
    name="test_overfit_faces840_gpu",
    debug_only=True,
    debug_training_on_test_split=True,
    train_and_validation_use_same_samples=True,
    results_are_not_final_evaluation=True,
    work_dir_root=_work_dir_root,
)

fragment_mesh_filter = dict(
    enabled=True,
    min_num_faces=840,
    max_num_faces=None,
    min_num_vertices=None,
    min_surface_area_m2=None,
    min_bbox_diagonal_m=None,
    exclude_entire_fragment=True,
    missing_mesh_policy="error",
    manifest_mismatch_policy="error",
    cache_metadata=True,
    train_policy="exclude",
    debug_eval_policy="exclude",
    validation_policy="report_only",
)

observed_filter = dict(
    min_observed_points=128,
    max_observed_points=4096,
    point_policy="deterministic_all_or_geometric_cap",
)

_train_manifest = (
    _work_dir_root + "/manifests/test_faces840_all_9e91dfb58d07.json"
)

data = dict(
    dataset_root=_dataset_root,
    train_manifest=_train_manifest,
    validation_manifest="same_as_train",
    fragment_mesh_filter=fragment_mesh_filter,
    observed_filter=observed_filter,
    train_batch_size=2,
    validation_batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    packed_collate=True,
    shuffle_train=True,
    shuffle_validation=False,
    max_train_samples=None,
    max_validation_samples=None,
)

dataset = dict(
    dataset_root=_dataset_root,
    fragment_mesh_filter=fragment_mesh_filter,
    observed_filter=observed_filter,
    fragment_mesh_cache_dir=_work_dir_root + "/cache",
    observed_policy="farthest_point_up_to_max",
    min_observed_points=128,
    max_observed_points=4096,
    voxel_size_m=0.002,
    template_fine_points=2048,
    template_coarse_points=512,
)

# The current heads still execute, but only these two pose terms contribute.
loss = dict(
    _delete_=True,
    symmetry_pose_weight=1.0,
    translation_cost_weight=10.0,
    rotation_cost_weight=1.0,
    pose_query_classification_weight=0.2,
    correspondence_weight=0.0,
    overlap_weight=0.0,
    template_visibility_weight=0.0,
    point_weight_weight=0.0,
    observed_region_weight=0.0,
    active_region_weight=0.0,
    uncertainty_weight=0.0,
    insufficient_information_weight=0.0,
    pose_decoder_auxiliary_weight=0.0,
)

train = dict(
    max_epochs=100,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=1e-4),
    scheduler=dict(type="cosine", warmup_epochs=0, min_lr=1e-6),
    gradient_accumulation_steps=2,
    gradient_clip_norm=1.0,
    amp=True,
    amp_dtype="auto",
    eval_interval_epochs=2,
    debug_visualization_interval_epochs=10,
    evaluate_before_training=True,
    visualize_before_training=True,
    log_interval_steps=10,
    save_best_only=True,
    best_metric="eval/symmetry_pose_loss",
    best_metric_mode="min",
    best_metric_min_delta=1e-6,
    best_metric_tie_breaker="eval/top1_pose_success_5deg_5mm",
    save_periodic_checkpoints=False,
    save_final_checkpoint=False,
)

history = dict(
    enabled=True,
    filename="history/history.jsonl",
    flush_every_record=True,
    fsync=False,
    save_epoch_csv=True,
)

terminal_output = dict(
    show_model=True,
    progress_bars=True,
    leave_progress_bars=True,
    print_train_epoch_summary=True,
    print_eval_metrics=True,
)

debug_visualization = dict(
    num_samples=8,
    progress_bar=True,
    debug_num_base_queries=3,
    debug_active_region_threshold=0.5,
    so2_visualization_samples=12,
    gallery_columns=4,
    gallery_spacing_scale=1.5,
    include_gt_reference=True,
    # Color the predicted fragment footprint directly on adaptively split
    # template faces, matching tools/debug_symmetry_visualization.py.
    template_projection_distance_m=5e-4,
    template_boundary_resolution_m=1e-4,
    template_boundary_max_depth=2,
)

seed = 0
work_dir = _work_dir_root
sample_manifest = _train_manifest
