_base_ = ["_base.py"]

experiment = dict(name="correspondence_pose_v1_01_correspondence_only_4views")
model = dict(correspondence_only=True)
loss = dict(
    conditioned_pose_loss=dict(
        base_pose_weight=0.0,
        best_residual_pose_weight=0.0,
        residual_regularization_weight=0.0,
    ),
    correspondence_pose_loss_weight=0.0,
    direct_vs_correspondence_pose_consistency_weight=0.0,
)
stage = dict(
    name="correspondence_only_4views",
    checkpoint_filename="best_correspondence_only.pth",
)
