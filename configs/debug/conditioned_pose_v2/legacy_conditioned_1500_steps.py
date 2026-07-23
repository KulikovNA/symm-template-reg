"""Explicit frozen-step control matching conditioned pose v1."""

_base_ = ["../conditioned_pose/01_base_k1_pose_only.py"]

experiment = dict(name="legacy_conditioned_1500_steps")
train_budget = dict(mode="optimizer_steps")
multi_view_batch = dict(enabled=False)
model = dict(base_pose_source="direct_context")
loss = dict(
    cross_view_world_consistency=dict(enabled=False),
    pairwise_pose_response=dict(enabled=False),
)
stage = dict(
    name="legacy_conditioned_fixed_1500",
    readiness_thresholds=dict(require_target_exposures_or_valid_early_stop=False),
)
