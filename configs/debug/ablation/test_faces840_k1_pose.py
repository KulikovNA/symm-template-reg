"""Ablation A: one direct pose query, no ranking or region supervision."""

_base_ = ["../test_overfit_faces840_gpu.py"]

experiment = dict(name="test_faces840_k1_pose")
model = dict(pose_head=dict(num_queries=1))
loss = dict(
    pose_query_ranking=dict(weight=0.0),
    observed_region_weight=0.0,
    active_region_weight=0.0,
    region_consistency_weight=0.0,
    auxiliary_registration_losses=False,
)
