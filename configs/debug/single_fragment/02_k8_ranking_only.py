"""Stage 02: keep all eight coordinates fixed and train categorical scores."""

_base_ = ["_base.py"]

experiment = dict(name="single_fragment_02_k8_ranking_only")

loss = dict(
    pose_query_ranking=dict(type="matched_categorical", weight=1.0),
)

train = dict(
    best_metric="eval/top1_scored_pose_cost",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/top1_query_is_oracle",
    best_metric_tie_breaker_mode="max",
)

stage = dict(
    name="ranking_only",
    checkpoint_filename="best_top1_ranking.pth",
    requires_init_stage="pose_only",
    trainable_module_prefixes=("pose_head.logit_projection",),
    readiness_thresholds=dict(
        metric="eval/top1_query_is_oracle",
        mode="max",
        value=0.9,
    ),
)
