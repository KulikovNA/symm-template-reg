"""Small selector-based overfit config; run only by an explicit user command."""

_base_ = ["../train/coordinate_guided_surface_v3.py"]

runtime = "production"
config_role = "debug_tiny_overfit"
experiment = dict(name="coordinate_guided_surface_v3_tiny_overfit")

data = dict(
    train=dict(
        selector=dict(
            scene_ids=("scene_000000",),
            frame_ids=(0, 1),
            fragment_ids=(0, 1),
            max_samples=8,
        ),
    ),
    validation=dict(
        selector=dict(
            scene_ids=("scene_000000",),
            frame_ids=(0, 1),
            fragment_ids=(0, 1),
            max_samples=8,
        ),
    ),
    train_batch_size=4,
    validation_batch_size=2,
    effective_batch_size=4,
    num_workers=0,
    persistent_workers=False,
)

train = dict(
    max_epochs=100,
    max_optimizer_steps=100,
    gradient_accumulation_steps=1,
    eval_interval_optimizer_steps=10,
    latest_checkpoint_interval_optimizer_steps=10,
)

