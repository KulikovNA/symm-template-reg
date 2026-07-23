_base_ = ["_base.py"]
experiment = dict(name="single_pose_uniform_correspondence_views04")
data = dict(expected_selected_samples=4, train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views04.json")
debug_visualization = dict(num_samples=4)
stage = dict(name="uniform_joint_views04")
