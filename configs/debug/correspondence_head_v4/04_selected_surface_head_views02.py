"""Two-view stage. Manually set _base_ to the selected passing one-view head first."""

_base_ = ["02_surface_v2_scheduled_frame04.py"]
experiment = dict(name="v4_selected_surface_head_views02")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json", expected_selected_samples=2)
debug_visualization = dict(num_samples=2)
stage = dict(name="selected_surface_head_views02")
stage_gate_dependencies = dict(parameterization_capacity_path=["/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame04_final8/surface_parameterization_capacity_summary.json", "/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame08_final8/surface_parameterization_capacity_summary.json"])
