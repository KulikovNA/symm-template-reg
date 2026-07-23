"""Conditioned pose v2: fair exposures and geometry-driven rotation context."""

_base_ = ["../conditioned_pose/01_base_k1_pose_only.py"]

experiment = dict(name="conditioned_pose_v2_DO_NOT_RUN")

train_budget = dict(
    mode="sample_exposures",
    target_exposures_per_sample=1500,
)

model = dict(
    base_pose_source="direct_context",
    sample_context_aggregator=dict(split_rotation_translation=True),
    base_pose_head=dict(
        split_rotation_translation=True,
        rotation_uses_centroid=False,
        translation_uses_centroid=True,
        output_mode="absolute",
    ),
)

multi_view_batch = dict(
    enabled=False,
    group_by=("scene_id", "fragment_id"),
    views_per_group=4,
    require_same_fragment_mesh=True,
)

loss = dict(
    pose_query_ranking=dict(weight=0.0),
    observed_region_weight=0.0,
    active_region_weight=0.0,
    correspondence_loss=dict(enabled=False, weight=0.0),
    cross_view_world_consistency=dict(
        enabled=False,
        weight=0.05,
        rotation_weight=1.0,
        translation_weight=10.0,
        reference_mode="pairwise_medoid",
    ),
    pairwise_pose_response=dict(
        enabled=False,
        weight=1.0,
        rotation_weight=0.25,
        translation_weight=0.25,
    ),
)

train = dict(
    max_optimizer_steps=1500,
    max_epochs=15000,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    min_sample_exposures_before_early_stop=750,
    early_stopping_patience_evals=0,
)

augmentation = dict(enabled=False)

stage = dict(
    readiness_thresholds=dict(
        top1_pose_success_5deg_5mm=0.9,
        rotation_response_ratio=0.5,
        base_pose_static_fraction=0.0,
        world_axis_spread_deg=10.0,
        world_translation_range_mm=10.0,
        require_target_exposures_or_valid_early_stop=True,
    )
)
