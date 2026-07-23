"""Ablation C: the complete K=8 V2 baseline with region supervision."""

_base_ = ["../test_overfit_faces840_gpu.py"]

experiment = dict(name="test_faces840_k8_pose_rank_regions")
model = dict(pose_head=dict(num_queries=8))
