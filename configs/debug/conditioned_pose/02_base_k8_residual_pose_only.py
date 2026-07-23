_base_ = ["01_base_k1_pose_only.py"]

experiment = dict(name="conditioned_02_base_k8_residual_pose_only")
model = dict(residual_pose_head=dict(num_hypotheses=8))
loss = dict(
    conditioned_pose_loss=dict(
        base_pose_weight=1.0,
        best_residual_pose_weight=1.0,
        residual_regularization_weight=0.01,
    )
)
stage = dict(name="conditioned_base_k8_residual", checkpoint_filename="best_base_k8.pth")
