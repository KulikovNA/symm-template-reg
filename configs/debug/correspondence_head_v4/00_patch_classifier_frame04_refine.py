"""Optional model-only Stage A ranking refinement; start with --init-checkpoint."""

_base_ = ["00_patch_classifier_frame04.py"]

experiment = dict(name="v4_patch_classifier_frame04_refine")
stage = dict(
    name="patch_classifier_frame04_refine",
    initialization="model_only_from_stage_a4_best_via_cli_init_checkpoint",
)
train = dict(
    max_epochs=500,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="constant", warmup_epochs=0, min_lr=1e-6),
)
