_base_ = ["_base.py"]

experiment = dict(name="correspondence_pose_v1_00_oracle_procrustes")
oracle_procrustes = dict(
    use_gt_correspondences=True,
    uniform_weights=True,
    max_rotation_error_deg=1e-4,
    max_translation_error_mm=1e-4,
    max_orthogonality_error=1e-6,
)
stage = dict(name="oracle_procrustes_preflight", checkpoint_filename="unused_oracle.pth")
