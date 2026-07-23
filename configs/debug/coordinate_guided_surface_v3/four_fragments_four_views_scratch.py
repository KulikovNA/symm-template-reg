"""Clean V3 scratch overfit: four physical fragments in four common views."""

_base_ = ["views10_scratch_full.py"]

debug_training_on_test_split = True
train_and_validation_use_same_samples = True
results_are_not_final_evaluation = True
experiment_type = "four_fragments_four_views_overfit"
initialization_mode = "scratch"
pretrained_checkpoint = None
seed = 0

experiment = dict(
    name="coordinate_guided_surface_v3_four_fragments_four_views_scratch",
    debug_training_on_test_split=True,
    train_and_validation_use_same_samples=True,
    results_are_not_final_evaluation=True,
    experiment_type="four_fragments_four_views_overfit",
    initialization_mode="scratch",
    pretrained_checkpoint=None,
)

data = dict(
    train_manifest=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
        "multifragment_overfit/scene000000_fragments0000_0003_"
        "frames0002_0004_0005_0008_shell_only.json"
    ),
    validation_manifest="same_as_train",
    single_fragment_contract=False,
    multifragment_contract=True,
    scene_id="scene_000000",
    fragment_id=None,
    expected_selected_samples=16,
    train_batch_size=16,
    validation_batch_size=2,
    effective_views_per_optimizer_step=16,
    shuffle_train=False,
    shuffle_validation=False,
)

dataset = dict(random_seed=0, registration_point_selection="shell_only")
augmentations = dict(enabled=False)

loss = dict(
    joint_correspondence_pose=dict(enabled=False),
    joint_surface_correspondence_pose_v3=dict(
        coordinate_mean_weight=1.0,
        coordinate_tail_weight=0.5,
        pose_rotation_weight=0.25,
        pose_translation_weight=0.25,
        alignment_weight=0.25,
        rotation_scale_deg=1.0,
        translation_scale_m=0.001,
        alignment_scale_m=0.001,
        warmup_epochs=250,
        loss_reduction="per_sample_mean_then_batch_mean",
    ),
)

train_budget = dict(mode="epochs", epochs=8000)
train = dict(
    max_epochs=8000,
    max_optimizer_steps=8000,
    gradient_accumulation_steps=1,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="linear_warmup_constant", warmup_optimizer_steps=100),
    gradient_clip_norm=1.0,
    amp=False,
    eval_interval_epochs=0,
    eval_interval_optimizer_steps=100,
    debug_visualization_interval_epochs=0,
    debug_visualization_interval_optimizer_steps=500,
    evaluate_before_training=True,
    visualize_before_training=True,
    save_best_only=True,
    save_periodic_checkpoints=False,
    save_final_checkpoint=False,
    early_stopping_patience_evals=0,
    min_sample_exposures_before_early_stop=5000,
    best_metric="eval/active/worst_sample_multifragment_score",
    best_metric_mode="min",
    best_metric_tie_breaker=None,
    best_metric_tie_breakers=(
        dict(metric="eval/active/pose_ready_sample_count", mode="max"),
        dict(metric="eval/active/practical_surface_passed_sample_count", mode="max"),
        dict(metric="eval/active/strict_surface_passed_sample_count", mode="max"),
        dict(metric="eval/active/exact_global/worst_correspondence_p95_mm", mode="min"),
    ),
)

stage = dict(name="FOUR_FRAGMENTS_FOUR_VIEWS_clean_v3_scratch")

active_coordinate_path = dict(
    enabled=True,
    clean_active_only=True,
    exact_global=True,
    k16=True,
    candidate_k=16,
    projection_chunk_size=256,
    pose_solver="uniform_procrustes",
    ten_view_gates=False,
    dual_gates=False,
    multifragment_gates=True,
    expected_fragments=(0, 1, 2, 3),
    expected_frames=(2, 4, 5, 8),
)

debug_visualization = dict(
    num_samples=16,
    single_fragment_layout=False,
    multifragment_layout=True,
    joint_correspondence_pose=True,
    coordinate_guided_primary=True,
    active_projected_pose_only=True,
    combined_world_pose_comparison=True,
    legacy_outputs=False,
)

conditioning_audit = dict(required_epochs=(0, 1000, 3000, 5000, "best", "final"))
target_leakage_policy = dict(
    forbid_detected=True,
    audit_path=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
        "multifragment_overfit/scene000000_fragments0000_0003_"
        "frames0002_0004_0005_0008_shell_only_identifiability.json"
    ),
)
plateau_detection = dict(enabled=True, warning_only=True, minimum_optimizer_step=5000)
