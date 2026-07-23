"""K=8 final-decoder-only pose baseline for view-ladder diagnostics."""

_base_ = ["_base.py"]

experiment = dict(name="view_ladder_k8_pose_only")
model = dict(pose_head=dict(num_queries=8))
stage = dict(
    name="view_ladder_k8_pose_only",
    checkpoint_filename="best_k8_view_ladder_pose.pth",
)

