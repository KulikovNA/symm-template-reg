"""Prepared scratch ablation with full joint loss from epoch zero; do not auto-run."""

_base_ = ["views10_scratch_full.py"]
experiment = dict(name="coordinate_guided_surface_v3_views10_scratch_no_warmup")
loss = dict(joint_surface_correspondence_pose_v3=dict(warmup_epochs=0))
