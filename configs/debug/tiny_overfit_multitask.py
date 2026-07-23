_base_ = ["tiny_overfit_pose_only.py"]

debug_training_on_test_split = True
results_are_not_final_evaluation = True

# Keep this explicit in every debug config until the audited value is selected.
fragment_mesh_filter = dict(enabled=False, min_num_faces=None)
data = dict(fragment_mesh_filter=fragment_mesh_filter)
dataset = dict(fragment_mesh_filter=fragment_mesh_filter)

training = dict(
    mode="multitask",
    auxiliary_registration_losses=True,
    pose_decoder_auxiliary_loss=True,
)
work_dir = "work_dirs/tiny_overfit_multitask"
