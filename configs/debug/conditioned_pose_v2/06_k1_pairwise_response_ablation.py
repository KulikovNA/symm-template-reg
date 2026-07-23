_base_ = ["01_k1_direct_equal_exposure.py"]

experiment = dict(name="conditioned_v2_06_k1_pairwise_response")
multi_view_batch = dict(enabled=True, views_per_group=4)
loss = dict(pairwise_pose_response=dict(enabled=True, weight=1.0))
stage = dict(name="conditioned_v2_k1_pairwise", checkpoint_filename="best_k1_pairwise.pth")
