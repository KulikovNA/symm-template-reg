_base_ = ["tiny_overfit_multitask.py"]

debug_training_on_test_split = True
results_are_not_final_evaluation = True

# Keep this explicit in every debug config until the audited value is selected.
fragment_mesh_filter = dict(enabled=False, min_num_faces=None)
data = dict(fragment_mesh_filter=fragment_mesh_filter)
dataset = dict(fragment_mesh_filter=fragment_mesh_filter)

sample_manifest = "work_dirs/debug_manifests/debug_scene_split_8_1_1.json"
work_dir = "work_dirs/debug_scene_split_8_1_1"
