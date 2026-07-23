_base_ = ["_base.py"]
experiment = dict(name="legacy_soft_shell_frame04")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json")
stage = dict(name="legacy_soft_frame04")
stage_gate_dependencies = dict(parameterization_capacity_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_diagnostics_v3_20260719/capacity_full_frame04/parameterization_capacity_summary.json",parameterization_capacity_required_field="soft_current_capacity_passed")
