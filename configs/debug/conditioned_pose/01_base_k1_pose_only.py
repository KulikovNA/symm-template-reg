_base_ = ["_base.py"]

experiment = dict(name="conditioned_01_base_k1_pose_only")
model = dict(residual_pose_head=dict(num_hypotheses=1))
stage = dict(name="conditioned_base_k1", checkpoint_filename="best_base_k1.pth")
