_base_ = ["four_fragments_four_views_scratch.py"]
experiment = dict(name="audit_vectorized_loss_fp32")
loss = dict(joint_surface_correspondence_pose_v3=dict(vectorized=True))

