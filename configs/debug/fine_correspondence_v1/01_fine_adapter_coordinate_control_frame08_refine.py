"""Prepared only; do not run before the frame-4 projection gate is analysed."""

_base_ = ["01_fine_adapter_coordinate_control_frame08.py"]
experiment = dict(name="fine_adapter_coordinate_control_frame08_refine")
stage = dict(
    name="F1_fine_adapter_coordinate_control_frame08_refine",
    initialization="model_only_via_cli_from_successful_frame08_coordinate_control",
    strict_initialization=True,
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 1e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 1e-4,
    },
)
train_budget = dict(mode="epochs", epochs=500)
train = dict(
    max_epochs=500, eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    min_sample_exposures_before_early_stop=100,
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    fine_coordinate_aux_weight=1.0,
    fine_coordinate_tail_weight=0.5,
))
