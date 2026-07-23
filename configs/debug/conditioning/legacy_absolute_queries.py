"""Reproducible legacy absolute-query architecture (static-codebook baseline)."""

_base_ = ["../view_ladder/k8_pose_only.py"]

experiment = dict(name="legacy_absolute_queries")
model = dict(
    pose_head=dict(
        type="LegacyAbsolutePoseQueryHead",
        num_queries=8,
    )
)
stage = dict(
    name="legacy_absolute_queries",
    checkpoint_filename="best_legacy_absolute_queries.pth",
)
