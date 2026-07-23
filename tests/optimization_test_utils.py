from __future__ import annotations

import copy

import torch

from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import CleanCoordinatePoseLossV3
from tests.test_fragment_symmetry_targets import metadata


def mixed_loss_inputs():
    torch.manual_seed(7)
    batch, points = 2, 11
    target = torch.randn(batch, points, 3) * .01
    vertices = [torch.cat((row, torch.tensor([[-.03, -.03, -.03], [.03, .03, .03]])), 0) for row in target]
    minimum = torch.stack([row.amin(0) for row in vertices])
    maximum = torch.stack([row.amax(0) for row in vertices])
    normalized = 2 * (target - minimum[:, None]) / (maximum - minimum)[:, None] - 1
    return dict(
        predicted_normalized_O=normalized,
        observed_points_C=target.clone(),
        target_points_O=target,
        valid_mask=torch.ones((batch, points), dtype=torch.bool),
        gt_pose_T_C_from_O=torch.eye(4).expand(batch, 4, 4).clone(),
        symmetry_metadata=[metadata(), metadata()],
        effective_symmetry_groups=[{"type": "C", "order": 2}, {"type": "C", "order": 4}],
        template_mesh_vertices_O=vertices,
    )


def loss_pair(requires_grad=False):
    inputs = mixed_loss_inputs()
    left_inputs = copy.copy(inputs); right_inputs = copy.copy(inputs)
    left_inputs["predicted_normalized_O"] = inputs["predicted_normalized_O"].clone().requires_grad_(requires_grad)
    right_inputs["predicted_normalized_O"] = inputs["predicted_normalized_O"].clone().requires_grad_(requires_grad)
    left = CleanCoordinatePoseLossV3(current_epoch=125, vectorized=False)(**left_inputs)
    right = CleanCoordinatePoseLossV3(current_epoch=125, vectorized=True)(**right_inputs)
    return left_inputs, right_inputs, left, right


def linear_accumulation(batch_size):
    torch.manual_seed(3)
    initial = torch.nn.Linear(4, 2)
    state = copy.deepcopy(initial.state_dict())
    x = torch.randn(16, 4); y = torch.randn(16, 2)
    model = torch.nn.Linear(4, 2); model.load_state_dict(state)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.zero_grad()
    for start in range(0, 16, batch_size):
        loss = (model(x[start:start + batch_size]) - y[start:start + batch_size]).square().mean()
        (loss * (batch_size / 16)).backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    optimizer.step()
    return gradients, [p.detach().clone() for p in model.parameters()]

