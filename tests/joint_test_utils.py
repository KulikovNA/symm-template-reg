from __future__ import annotations
import torch
from symm_template_reg.models.losses import JointCorrespondencePoseLoss
from symm_template_reg.models.pose.pose_representation import transform_points
from tests.test_fragment_symmetry_targets import metadata

def fixture():
    q = torch.tensor([[[.01,.00,.00],[.00,.02,.00],[-.01,.00,.01],[.00,-.02,-.01]]], dtype=torch.float64)
    pose = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor([.1,.2,.3], dtype=torch.float64)
    p = transform_points(pose, q)
    mask = torch.ones((1,4), dtype=torch.bool)
    return JointCorrespondencePoseLoss(), q, pose, p, mask, metadata(), {"type":"C","order":2}

def call(criterion, q_pred, pose_pred, q, pose, p, mask, meta, group, surface=None):
    surface = q if surface is None else surface
    surface_mask = torch.ones(surface.shape[:2], dtype=torch.bool)
    return criterion(q_pred, pose_pred, pose, p, q, mask, surface, surface_mask, [meta], [group])
