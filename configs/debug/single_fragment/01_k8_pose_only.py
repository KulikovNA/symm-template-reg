"""Stage 01: K=8 oracle pose generation, without ranking or regions."""

_base_ = ["_base.py"]

experiment = dict(name="single_fragment_01_k8_pose_only")

loss = dict(symmetry_pose_weight=1.0)

train = dict(
    best_metric="eval/oracle_best_pose_cost",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/oracle_topK_pose_success_5deg_5mm",
    best_metric_tie_breaker_mode="max",
)

stage = dict(
    name="pose_only",
    checkpoint_filename="best_oracle_pose.pth",
    trainable_module_prefixes=(
        "observed_encoder",
        "template_encoder",
        "geometric_embedding",
        "interaction_transformer",
        "cloud_type_embedding",
        "pose_head.layers",
        "pose_head.query_embedding",
        "pose_head.query_content",
        "pose_head.pose_projection",
    ),
    readiness_thresholds=dict(
        metric="eval/oracle_topK_pose_success_5deg_5mm",
        mode="max",
        value=0.9,
    ),
)
