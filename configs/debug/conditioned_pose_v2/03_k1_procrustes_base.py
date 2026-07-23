_base_ = ["_base.py"]

experiment = dict(name="conditioned_v2_03_k1_procrustes_base")
model = dict(
    base_pose_source="weighted_procrustes",
    weighted_procrustes=dict(type="WeightedProcrustes"),
    residual_pose_head=dict(num_hypotheses=1),
)
loss = dict(
    correspondence_loss=dict(enabled=True, weight=1.0),
    correspondence_pose_loss_weight=1.0,
)
stage = dict(name="conditioned_v2_k1_procrustes", checkpoint_filename="best_k1_procrustes.pth")
