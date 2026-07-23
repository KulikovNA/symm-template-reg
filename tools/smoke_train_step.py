#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import torch

from _common import build_loss, build_model, build_real_batch, move_to_device, resolve_device
from symm_template_reg.models.losses import CorrespondenceLoss, OverlapLoss, PointConfidenceLoss
from symm_template_reg.models.losses import PoseSetLoss
from symm_template_reg.engine.trainer import compute_training_losses


def _padded_overlap(batch, output):
    labels = batch["gt"].get("overlap_labels")
    if labels is None:
        return None
    if labels.ndim == 2:
        return labels
    padded = torch.zeros_like(output.observed_valid_mask)
    start = 0
    for index, length in enumerate(output.observed_valid_mask.sum(-1).tolist()):
        padded[index, :length] = labels[start : start + length]
        start += length
    return padded


def _target_hypotheses(batch):
    gt_pose = batch["gt"]["T_C_from_O"]
    equivalents = batch["gt"].get("equivalent_T_C_from_O")
    targets = []
    for index in range(gt_pose.shape[0]):
        candidate = equivalents[index] if equivalents else None
        targets.append(candidate if isinstance(candidate, torch.Tensor) else gt_pose[index].unsqueeze(0))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one finite optimizer step on a real batch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if device is None:
        print(json.dumps({"status": "skipped", "reason": "CUDA is not available"}, indent=2))
        return 0
    torch.manual_seed(0)
    config, _, batch, lengths = build_real_batch(args.config, args.num_samples)
    batch = move_to_device(batch, device)
    model = build_model(config["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss_config = config.get("loss", {"type": "PoseSetLoss"})
    if "type" not in loss_config:
        pose_criterion = PoseSetLoss(
            translation_weight=float(loss_config.get("translation_cost_weight", 10.0)),
            rotation_weight=float(loss_config.get("rotation_cost_weight", 1.0)),
            classification_weight=float(
                loss_config.get("pose_query_classification_weight", 0.2)
            ),
            auxiliary_weight=float(
                loss_config.get("pose_decoder_auxiliary_weight", 0.0)
            ),
        )
        total, pose_losses = compute_training_losses(
            output,
            batch,
            pose_criterion,
            {
                **loss_config,
                "auxiliary_registration_losses": False,
                "pose_decoder_auxiliary_loss": False,
            },
        )
        losses = dict(pose_losses)
    else:
        pose_criterion = build_loss(loss_config)
        pose_losses = pose_criterion(
            output.pose_hypotheses,
            output.pose_logits,
            _target_hypotheses(batch),
            output.auxiliary_outputs,
        )
        losses = dict(pose_losses)
        total = None
    correspondence_target = batch["gt"].get("points_O_corresponding")
    if total is None and correspondence_target is not None:
        if hasattr(correspondence_target, "to_padded"):
            correspondence_target = correspondence_target.to_padded()["points"]
        losses["loss_correspondence"] = CorrespondenceLoss()(
            output.correspondence_points_O,
            correspondence_target,
            output.observed_valid_mask,
        )
    overlap_target = _padded_overlap(batch, output)
    if total is None and overlap_target is not None:
        losses["loss_overlap"] = OverlapLoss()(
            output.observed_overlap_logits, overlap_target, output.observed_valid_mask
        )
        confidence_logits = torch.logit(output.correspondence_confidence.clamp(1e-5, 1 - 1e-5))
        losses["loss_point_confidence"] = PointConfidenceLoss()(
            confidence_logits, overlap_target, output.observed_valid_mask
        )
    if total is None:
        losses["loss_insufficient_information"] = torch.nn.functional.binary_cross_entropy_with_logits(
            output.insufficient_information_logit, torch.zeros_like(output.insufficient_information_logit)
        )
        total = sum(value for key, value in losses.items() if key.startswith("loss_") and key not in {
            "loss_translation", "loss_rotation", "loss_pose_classification", "loss_pose_auxiliary"
        })
    if not torch.isfinite(total):
        raise RuntimeError(f"non-finite total loss: {float(total.detach())}")
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("missing or non-finite gradients")
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    print(
        json.dumps(
            {
                "status": "ok",
                "device": str(device),
                "observed_lengths": lengths,
                "total_loss": float(total.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "finite_gradients": True,
                "optimizer_step": True,
                "losses": {
                    key: (
                        float(value.detach())
                        if value.ndim == 0
                        else value.detach().cpu().tolist()
                    )
                    for key, value in losses.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
