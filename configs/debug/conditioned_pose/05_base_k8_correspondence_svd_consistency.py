_base_ = ["04_base_k8_correspondence.py"]

experiment = dict(name="conditioned_05_base_k8_correspondence_svd_consistency")
model = dict(weighted_procrustes=dict(type="WeightedProcrustes"))
loss = dict(
    correspondence_pose_loss_weight=0.5,
    direct_vs_correspondence_pose_consistency_weight=0.1,
)
stage = dict(name="conditioned_k8_svd_consistency", checkpoint_filename="best_k8_svd.pth")
