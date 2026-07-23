_base_ = ["01_fine_adapter_coordinate_control_frame04.py"]
experiment = dict(name="fine_adapter_coordinate_control_frame08")
data = dict(train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame08_only.json")
stage = dict(name="F1_fine_adapter_coordinate_control_frame08")

