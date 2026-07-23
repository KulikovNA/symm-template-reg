_base_ = ["02_base_k8_residual_pose_only.py"]

experiment = dict(name="conditioned_04_base_k8_correspondence")
loss = dict(
    correspondence_loss=dict(
        enabled=True,
        weight=1.0,
        robust_type="smooth_l1",
        use_shared_symmetry_element=True,
    )
)
stage = dict(name="conditioned_k8_correspondence", checkpoint_filename="best_k8_corr.pth")
