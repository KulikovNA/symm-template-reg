_base_ = ["02_procrustes_base_4views.py"]

experiment = dict(name="correspondence_pose_v1_04_procrustes_cross_view_ablation")
multi_view_batch = dict(enabled=True, views_per_group=4)
loss = dict(
    cross_view_world_consistency=dict(
        enabled=True,
        weight=0.05,
        rotation_weight=1.0,
        translation_weight=10.0,
        reference_mode="pairwise_medoid",
    ),
    pairwise_pose_response=dict(
        enabled=True,
        weight=0.25,
        rotation_weight=0.25,
        translation_weight=0.25,
    ),
)
stage = dict(name="procrustes_cross_view_ablation", checkpoint_filename="best_procrustes_cross_view.pth")
