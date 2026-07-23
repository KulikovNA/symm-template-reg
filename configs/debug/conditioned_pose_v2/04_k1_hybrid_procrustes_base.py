_base_ = ["03_k1_procrustes_base.py"]

experiment = dict(name="conditioned_v2_04_k1_hybrid_procrustes_base")
model = dict(
    base_pose_source="procrustes_plus_direct_residual",
    base_pose_head=dict(
        output_mode="bounded_correction",
        max_rotation_correction_deg=15.0,
        max_translation_correction_m=0.01,
    ),
)
stage = dict(name="conditioned_v2_k1_hybrid", checkpoint_filename="best_k1_hybrid.pth")
