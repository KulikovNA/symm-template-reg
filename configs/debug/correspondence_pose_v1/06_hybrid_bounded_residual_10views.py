_base_ = ["03_hybrid_bounded_residual_4views.py"]

experiment = dict(name="correspondence_pose_v1_06_hybrid_bounded_residual_10views")
data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/view_ladder/all_10_views.json",
    validation_manifest="same_as_train",
    expected_selected_samples=10,
)
stage = dict(name="hybrid_bounded_residual_10views", checkpoint_filename="best_hybrid_bounded_10views.pth")
