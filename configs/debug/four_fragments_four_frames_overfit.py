"""Overfit one scene: four physical fragments observed in four frames."""

_base_ = ["../train/coordinate_guided_surface_v3.py"]

runtime = "production"
config_role = "debug_four_fragments_four_frames_overfit"
initialization_mode = "scratch"

experiment = dict(
    name="coordinate_guided_surface_v3_four_fragments_four_frames_overfit"
)

_selector = dict(
    scene_ids=("scene_000000",),
    frame_ids=(2, 4, 5, 8),
    fragment_ids=(0, 1, 2, 3),
    max_samples=16,
)

data = dict(
    train_split="train",
    # Debug-only: evaluate the exact same physical observations without
    # augmentation. Production training continues to validate on split=val.
    validation_split="train",
    train=dict(
        # scene_000000/fragment_0000 has 758 faces. Lowering this threshold is
        # intentional only for reproducing the 4x4 overfit experiment.
        min_num_faces=0,
        selector=_selector,
    ),
    validation=dict(
        min_num_faces=0,
        selector=_selector,
        point_sampling="farthest_point_up_to_max",
        boundary_augmentation=dict(enabled=False),
    ),
    train_batch_size=4,
    validation_batch_size=4,
    effective_batch_size=16,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    drop_last=False,
)

# 16 observations / batch 4 / accumulation 4 = exactly one optimizer step
# per epoch. The run therefore has an unambiguous update budget.
train = dict(
    max_epochs=8000,
    max_optimizer_steps=8000,
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0,
    amp=False,
    eval_interval_optimizer_steps=100,
    latest_checkpoint_interval_optimizer_steps=100,
)

validation = dict(
    max_batches=4,
    evaluation_role="overfit_validation",
)
