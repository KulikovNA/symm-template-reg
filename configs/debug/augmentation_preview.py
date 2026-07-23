"""Configuration values used by visualize_boundary_augmentation.py."""

runtime = "augmentation_preview"
config_role = "debug_augmentation_preview"

augmentation = dict(
    enabled=True,
    apply_probability=1.0,
    mode="mixed",
    radius_px=dict(min=1, max=2),
    max_removed_fraction=0.08,
    max_added_fraction=0.05,
    min_points_after_augmentation=128,
    max_pseudo_target_distance_m=0.002,
    max_local_depth_difference_m=0.010,
    local_depth_window_px=3,
)

