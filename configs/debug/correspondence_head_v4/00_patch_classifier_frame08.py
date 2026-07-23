_base_ = ["00_patch_classifier_frame04.py"]
experiment = dict(name="v4_patch_classifier_frame08")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame08_only.json")
stage = dict(name="patch_classifier_frame08")
stage_gate_dependencies = dict(parameterization_capacity_path="/home/nikita/disser/fragment-template-registration-lab/work_dirs/correspondence_head_v4_20260719/capacity_frame08_final8/surface_parameterization_capacity_summary.json")
