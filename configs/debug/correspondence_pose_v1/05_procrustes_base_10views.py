_base_ = ["02_procrustes_base_4views.py"]

experiment = dict(name="correspondence_pose_v1_05_procrustes_base_10views")
data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/view_ladder/all_10_views.json",
    validation_manifest="same_as_train",
    expected_selected_samples=10,
)
stage = dict(name="procrustes_base_10views", checkpoint_filename="best_procrustes_base_10views.pth")
