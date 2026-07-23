"""Shared strict schedule for one physical fragment observed in ten views."""

_base_ = ["../test_overfit_faces840_gpu.py"]

experiment = dict(
    name="single_fragment_base_DO_NOT_RUN",
)

data = dict(
    train_manifest=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
        "single_fragment_scene000000_fragment0002_de68591bf9a5.json"
    ),
    validation_manifest="same_as_train",
    single_fragment_contract=True,
    expected_selected_samples=10,
    scene_id="scene_000000",
    fragment_id=2,
    scene_ids=None,
    train_batch_size=2,
    validation_batch_size=2,
    num_workers=0,
    persistent_workers=False,
    shuffle_train=True,
    shuffle_validation=False,
    max_train_samples=None,
    max_validation_samples=None,
)

augmentation = dict(enabled=False)

model = dict(
    pose_head=dict(num_queries=8),
)

loss = dict(
    symmetry_pose_weight=0.0,
    pose_query_classification_weight=0.0,
    pose_query_ranking=dict(
        type="matched_categorical",
        weight=0.0,
        temperature=0.25,
        cost_normalization="minmax",
        detach_pose_cost=True,
    ),
    observed_region_weight=0.0,
    observed_region_loss=dict(
        class_balancing="inverse_sqrt_frequency",
        max_class_weight=5.0,
    ),
    active_region_weight=0.0,
    active_region_loss=dict(
        type="bce",
        focal_gamma=2.0,
        pos_weight_source="manifest",
    ),
    region_consistency_weight=0.0,
    correspondence_weight=0.0,
    overlap_weight=0.0,
    template_visibility_weight=0.0,
    point_weight_weight=0.0,
    uncertainty_weight=0.0,
    insufficient_information_weight=0.0,
    pose_decoder_auxiliary_weight=0.0,
    auxiliary_registration_losses=True,
    cross_view_consistency_weight=0.0,
)

train = dict(
    max_optimizer_steps=3000,
    max_epochs=1000,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    gradient_clip_norm=1.0,
    amp=True,
    amp_dtype="auto",
    eval_interval_optimizer_steps=100,
    debug_visualization_interval_optimizer_steps=250,
    log_interval_optimizer_steps=10,
    eval_interval_epochs=0,
    debug_visualization_interval_epochs=0,
    evaluate_before_training=True,
    visualize_before_training=True,
    early_stopping_patience_evals=10,
    save_best_only=True,
    save_periodic_checkpoints=False,
    save_final_checkpoint=False,
)

debug_visualization = dict(
    num_samples=10,
    single_fragment_layout=True,
    debug_num_base_queries=8,
)

stage = dict(
    name="base",
    trainable_module_prefixes=None,
    checkpoint_filename="best.pth",
    readiness_thresholds=dict(),
)
