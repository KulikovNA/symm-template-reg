_base_ = ["01_k1_direct_equal_exposure.py"]

experiment = dict(name="conditioned_v2_02_k1_cross_view_consistency")
multi_view_batch = dict(enabled=True, views_per_group=4)
loss = dict(cross_view_world_consistency=dict(enabled=True, weight=0.05))
stage = dict(name="conditioned_v2_k1_cross_view", checkpoint_filename="best_k1_cross_view.pth")
