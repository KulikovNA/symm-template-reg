"""Ablation B: eight pose queries with soft-quality ranking, no region losses."""

_base_ = ["../test_overfit_faces840_gpu.py"]

experiment = dict(name="test_faces840_k8_pose_rank")
model = dict(pose_head=dict(num_queries=8))
loss = dict(
    observed_region_weight=0.0,
    active_region_weight=0.0,
    region_consistency_weight=0.0,
    auxiliary_registration_losses=False,
)
