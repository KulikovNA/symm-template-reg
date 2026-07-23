_base_ = ["_base.py"]

experiment = dict(name="conditioned_v2_01_k1_direct_equal_exposure")
model = dict(base_pose_source="direct_context", residual_pose_head=dict(num_hypotheses=1))
stage = dict(name="conditioned_v2_k1_direct", checkpoint_filename="best_k1_direct.pth")
