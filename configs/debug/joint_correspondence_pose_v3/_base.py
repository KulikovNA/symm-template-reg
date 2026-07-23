"""Shell-only one-view ladder for correspondence-head diagnosis."""

_base_ = ["../joint_correspondence_pose_v2/_base.py"]

experiment = dict(name="joint_correspondence_pose_v3_base_DO_NOT_RUN")

dataset = dict(
    registration_point_selection="shell_only",
    random_seed=0,
)

data = dict(
    train_batch_size=1,
    validation_batch_size=1,
    expected_selected_samples=1,
    validation_manifest="same_as_train",
    shuffle_train=True,
    shuffle_validation=False,
)

model = dict(
    correspondence_head=dict(temperature=1.0),
)

train_budget = dict(mode="epochs", epochs=1000)
train = dict(
    max_optimizer_steps=None,
    max_epochs=1000,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    eval_interval_optimizer_steps=0,
    debug_visualization_interval_optimizer_steps=0,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    evaluate_before_training=True,
    visualize_before_training=True,
)

debug_visualization = dict(
    num_samples=1,
    single_fragment_layout=True,
    joint_correspondence_pose=True,
)

target_leakage_policy = dict(
    forbid_detected=True,
    audit_path=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/"
        "joint_target_leakage_audit_20260719_152537/target_leakage_audit.json"
    ),
)

registration_contract = dict(
    point_selection="shell_only",
    gt_q_template_surface_p95_gate_mm=0.5,
    gt_reconstruction_p95_gate_mm=0.1,
)

attention_sharpness = dict(
    constant_temperature=True,
    initial_temperature=1.0,
    final_temperature=1.0,
    anneal_epochs=0,
)

stage_gate_dependencies = dict(
    require_point_contract_audit=True,
    point_contract_audit_path=(
        "/home/nikita/disser/fragment-template-registration-lab/work_dirs/"
        "correspondence_diagnostics_v3_20260719/point_contract/"
        "registration_point_contract_summary.json"
    ),
    require_parameterization_capacity=True,
    parameterization_capacity_path=None,
    parameterization_capacity_required_field="free_capacity_passed",
    require_no_correspondence_collapse=True,
    require_no_excessive_attention_diffusion=True,
    per_sample_physical_gates=True,
)
