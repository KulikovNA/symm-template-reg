_base_ = ["../symm_template_reg_baseline.py"]

debug_training_on_test_split = True
results_are_not_final_evaluation = True

# Deliberately disabled until the user selects min_num_faces from the audit.
fragment_mesh_filter = dict(enabled=False, min_num_faces=None)

data = dict(fragment_mesh_filter=fragment_mesh_filter)
dataset = dict(
    fragment_mesh_filter=fragment_mesh_filter,
    observed_policy="farthest_point_up_to_max",
    max_observed_points=4096,
)

sample_manifest = "work_dirs/debug_manifests/tiny_overfit_16.json"

dataloader = dict(batch_size=4, num_workers=0, shuffle=True)
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-4)
training = dict(
    mode="pose_only",
    max_steps=3000,
    eval_interval=100,
    checkpoint_interval=250,
    gradient_clip_norm=1.0,
    gradient_accumulation_steps=1,
    amp=True,
    early_stopping_patience=10,
    auxiliary_registration_losses=False,
    pose_decoder_auxiliary_loss=False,
)
work_dir = "work_dirs/tiny_overfit_pose_only"
