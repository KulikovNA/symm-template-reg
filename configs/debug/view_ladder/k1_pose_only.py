"""K=1 pose-only baseline for one-frame and progressive-view diagnostics."""

_base_ = ["_base.py"]

experiment = dict(name="view_ladder_k1_pose_only")
model = dict(pose_head=dict(num_queries=1))
stage = dict(
    name="view_ladder_k1_pose_only",
    checkpoint_filename="best_k1_view_ladder_pose.pth",
)

