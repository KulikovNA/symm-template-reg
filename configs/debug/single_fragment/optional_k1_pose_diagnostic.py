"""Optional diagnostic only: one pose query, never an initializer for K=8."""

_base_ = ["01_k8_pose_only.py"]

experiment = dict(name="single_fragment_optional_k1_pose_diagnostic")
model = dict(pose_head=dict(num_queries=1))
stage = dict(
    name="optional_k1_pose_diagnostic",
    checkpoint_filename="best_k1_pose_diagnostic.pth",
    forbids_initialization_of_num_queries=8,
)
