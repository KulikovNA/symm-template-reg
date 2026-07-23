"""Stage 04: unfreeze the full K=8 model and jointly fine-tune enabled heads."""

_base_ = ["_base.py"]

experiment = dict(name="single_fragment_04_k8_joint_finetune")

loss = dict(
    symmetry_pose_weight=1.0,
    pose_query_ranking=dict(type="matched_categorical", weight=1.0),
    observed_region_weight=0.25,
    active_region_weight=0.5,
    region_consistency_weight=0.1,
)

train = dict(
    max_optimizer_steps=1500,
    optimizer=dict(type="AdamW", lr=5e-5, weight_decay=0.0),
    best_metric="eval/top1_scored_pose_cost",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/top1_pose_success_5deg_5mm",
    best_metric_tie_breaker_mode="max",
)

stage = dict(
    name="joint_finetune",
    checkpoint_filename="best_joint_top1.pth",
    requires_init_stage="regions_only",
    trainable_module_prefixes=None,
    readiness_thresholds=dict(
        metric="eval/top1_pose_success_5deg_5mm",
        mode="max",
        value=0.9,
    ),
)
