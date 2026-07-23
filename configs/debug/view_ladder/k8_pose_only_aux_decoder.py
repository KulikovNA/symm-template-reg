"""K=8 ablation with the same symmetry-aware loss on intermediate decoders."""

_base_ = ["k8_pose_only.py"]

experiment = dict(name="view_ladder_k8_pose_only_aux_decoder")
loss = dict(pose_decoder_auxiliary_weight=0.5)
stage = dict(
    name="view_ladder_k8_pose_only_aux_decoder",
    checkpoint_filename="best_k8_aux_view_ladder_pose.pth",
)
