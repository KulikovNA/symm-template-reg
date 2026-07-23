"""Five-step active-graph smoke; never a training result."""

_base_ = ["../train/coordinate_guided_surface_v3.py"]

runtime = "production"
config_role = "debug_smoke"
experiment = dict(name="coordinate_guided_surface_v3_smoke")

data = dict(
    train=dict(
        selector=dict(
            scene_ids=("scene_000000",),
            frame_ids=(0, 1),
            max_samples=8,
        ),
        boundary_augmentation=dict(
            enabled=True,
            apply_probability=1.0,
            mode="mixed",
        ),
    ),
    validation=dict(
        selector=dict(
            scene_ids=("scene_000000",),
            frame_ids=(0,),
            max_samples=2,
        ),
    ),
    train_batch_size=2,
    validation_batch_size=1,
    effective_batch_size=2,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

train = dict(
    max_epochs=5,
    max_optimizer_steps=5,
    gradient_accumulation_steps=1,
    eval_interval_optimizer_steps=5,
    latest_checkpoint_interval_optimizer_steps=5,
)

validation = dict(max_batches=1)

