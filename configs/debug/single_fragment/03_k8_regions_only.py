"""Stage 03: train point and active-region classifiers with pose/ranking frozen."""

_base_ = ["_base.py"]

experiment = dict(name="single_fragment_03_k8_regions_only")

loss = dict(
    observed_region_weight=0.25,
    active_region_weight=0.5,
    region_consistency_weight=0.1,
)

train = dict(
    best_metric="eval/effective_group_accuracy",
    best_metric_mode="max",
    best_metric_tie_breaker="eval/active_region_exact_set_accuracy",
    best_metric_tie_breaker_mode="max",
)

stage = dict(
    name="regions_only",
    checkpoint_filename="best_regions.pth",
    requires_init_stage="ranking_only",
    trainable_module_prefixes=("symmetry_head",),
    readiness_thresholds=dict(
        metric="eval/effective_group_accuracy",
        mode="max",
        value=0.9,
    ),
)
