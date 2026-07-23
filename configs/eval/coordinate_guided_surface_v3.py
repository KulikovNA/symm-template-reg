"""Explicit val/test evaluation for the production V3 model."""

_base_ = ["../train/coordinate_guided_surface_v3.py"]

runtime = "production_evaluation"
config_role = "evaluation"

data = dict(
    train=None,
    validation=dict(
        type="SplitDirectoryFragmentDataset",
        min_num_faces=840,
        min_observed_shell_points=128,
        max_observed_shell_points=4096,
        point_sampling="farthest_point_up_to_max",
        boundary_augmentation=dict(enabled=False),
    ),
    num_workers=2,
    persistent_workers=True,
)

validation = dict(max_batches=None)

