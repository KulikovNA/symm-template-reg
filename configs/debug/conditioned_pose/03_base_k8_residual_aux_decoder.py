_base_ = ["02_base_k8_residual_pose_only.py"]

experiment = dict(name="conditioned_03_base_k8_residual_aux_decoder")
loss = dict(pose_decoder_auxiliary_weight=0.5)
stage = dict(name="conditioned_k8_aux_decoder", checkpoint_filename="best_k8_aux.pth")
