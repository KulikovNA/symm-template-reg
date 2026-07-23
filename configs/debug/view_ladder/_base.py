"""Deterministic pose-only defaults for explicit view-ladder experiments."""

_base_ = ["../single_fragment/01_k8_pose_only.py"]

experiment = dict(name="view_ladder_base_DO_NOT_RUN")

data = dict(
    train_manifest=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/"
        "view_ladder/frame04_only.json"
    ),
    validation_manifest="same_as_train",
    expected_selected_samples=1,
    train_batch_size=1,
    validation_batch_size=1,
    num_workers=0,
    persistent_workers=False,
    shuffle_train=True,
    shuffle_validation=False,
)

dataset = dict(random_seed=0)
augmentation = dict(enabled=False)

loss = dict(
    symmetry_pose_weight=1.0,
    pose_query_classification_weight=0.0,
    pose_query_ranking=dict(weight=0.0),
    observed_region_weight=0.0,
    active_region_weight=0.0,
    region_consistency_weight=0.0,
    correspondence_weight=0.0,
    overlap_weight=0.0,
    template_visibility_weight=0.0,
    point_weight_weight=0.0,
    uncertainty_weight=0.0,
    insufficient_information_weight=0.0,
    pose_decoder_auxiliary_weight=0.0,
    cross_view_consistency_weight=0.0,
)

train = dict(
    max_optimizer_steps=1500,
    max_epochs=1500,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    amp=False,
    eval_interval_optimizer_steps=50,
    debug_visualization_interval_optimizer_steps=250,
    log_interval_optimizer_steps=10,
    early_stopping_patience_evals=0,
)

debug_visualization = dict(num_samples=1, single_fragment_layout=True)
