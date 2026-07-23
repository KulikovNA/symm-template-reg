#!/usr/bin/env python3
"""Audit zero-loss, shared-symmetry, surface, uniformity and gradient contracts."""

from __future__ import annotations
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.losses import JointCorrespondencePoseLoss  # noqa: E402
from symm_template_reg.models.pose.pose_representation import transform_points  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms  # noqa: E402
from symm_template_reg.models.symmetry.groups import parse_rotation_group  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg  # noqa: E402


def _build_real_batch(config: dict, manifest_path: Path, output: Path, device: torch.device):
    dataset_cfg = copy.deepcopy(config["dataset"])
    dataset_cfg["fragment_mesh_filter"] = copy.deepcopy(config["data"]["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = copy.deepcopy(config["data"]["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = copy.deepcopy(config["data"].get("symmetry_region_activity", {}))
    dataset_cfg["fragment_mesh_cache_dir"] = str(output / "cache")
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = str(manifest["samples"][0]["sample_id"])
    index = {record.sample_id: i for i, record in enumerate(dataset.sample_records)}[wanted]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    return move_to_device(collate([dataset[index]]), device)


def _criterion(config: dict) -> JointCorrespondencePoseLoss:
    cfg = dict(config["loss"]["joint_correspondence_pose"])
    cfg.pop("enabled", None)
    return JointCorrespondencePoseLoss(**cfg)


def _call(criterion, q, pose, gt_pose, observed, target, mask, surface, surface_mask, metadata, group):
    return criterion(q, pose, gt_pose, observed, target, mask, surface, surface_mask, [metadata], [group])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    register_all_modules()
    batch = _build_real_batch(config, Path(args.manifest).expanduser().resolve(), output, device)
    criterion = _criterion(config).to(device)
    observed_dense = batch["observed"].to_padded()
    target_dense = batch["gt"]["points_O_corresponding"].to_padded()
    template_dense = batch["template"].to_padded()
    mask = observed_dense["valid_mask"]
    q_gt = target_dense["points"]
    gt_pose = batch["gt"]["T_C_from_O"]
    observed = transform_points(gt_pose, q_gt)
    metadata = batch["template_symmetry_metadata"][0]
    group_payload = batch["gt"]["effective_symmetry_group"][0]
    group = parse_rotation_group(group_payload)
    transforms = symmetry_transforms(group, metadata.axis.direction, metadata.axis.origin, dtype=q_gt.dtype, device=device)
    symmetry = transforms[1]
    q_equiv = transform_points(torch.linalg.inv(symmetry), q_gt[0]).unsqueeze(0)
    pose_equiv = (gt_pose[0] @ symmetry).unsqueeze(0)
    observed_equiv = transform_points(pose_equiv, q_equiv)
    audit_surface = torch.cat((q_gt, q_equiv), dim=1)
    audit_surface_mask = torch.cat((mask, mask), dim=1)
    perfect = _call(criterion, q_gt, gt_pose, gt_pose, observed, q_gt, mask, audit_surface, audit_surface_mask, metadata, group_payload)
    equivalent = _call(criterion, q_equiv, pose_equiv, gt_pose, observed_equiv, q_gt, mask, audit_surface, audit_surface_mask, metadata, group_payload)
    wrong_q = q_gt + torch.tensor([0.010, -0.007, 0.005], device=device)
    wrong_pose = gt_pose.clone(); wrong_pose[:, :3, 3] += 0.01
    wrong = _call(criterion, wrong_q, wrong_pose, gt_pose, observed, q_gt, mask, q_gt, mask, metadata, group_payload)
    subset_q = q_gt.clone(); valid_ids = torch.nonzero(mask[0]).flatten(); subset_q[0, valid_ids[::2]] += 0.01
    subset = _call(criterion, subset_q, gt_pose, gt_pose, observed, q_gt, mask, q_gt, mask, metadata, group_payload)
    off = _call(criterion, wrong_q, gt_pose, gt_pose, observed, q_gt, mask, q_gt, mask, metadata, group_payload)

    model = build_model(config["model"]).to(device).train()
    prediction = model(batch)
    losses = _call(
        criterion, prediction.correspondence_points_O, prediction.correspondence_pose,
        gt_pose, observed_dense["points"], q_gt, prediction.observed_valid_mask,
        template_dense["points"], template_dense["valid_mask"], metadata, group_payload,
    )
    pose_objective = losses["weighted_loss_rotation"] + losses["weighted_loss_translation"]
    model.zero_grad(set_to_none=True)
    pose_objective.backward()
    module_gradients = {}
    for name in ("correspondence_head", "interaction_transformer", "observed_encoder", "template_encoder"):
        module = getattr(model, name)
        module_gradients[name] = sum(float(p.grad.detach().norm()) for p in module.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    weights = prediction.correspondence_confidence[0, prediction.observed_valid_mask[0]]
    count = len(weights)
    checks = {
        "perfect_all_losses_near_zero": float(perfect["loss_total"]) < 1e-5,
        "c2_equivalent_all_losses_near_zero": float(equivalent["loss_total"]) < 1e-5,
        "wrong_correspondence_and_pose_positive": float(wrong["loss_total"]) > 1.0,
        "wrong_subset_not_hidden": float(subset["loss_correspondence_normalized"]) > 0.1,
        "off_template_surface_positive": float(off["loss_template_surface_normalized"]) > 0.1,
        "pose_weight_nonzero": criterion.weights["rotation"] > 0 and criterion.weights["translation"] > 0,
        "alignment_weight_nonzero": criterion.weights["alignment"] > 0,
        "pose_gradient_reaches_all_required_modules": all(value > 0 and math.isfinite(value) for value in module_gradients.values()),
        "confidence_head_absent": model.point_weight_head is None,
        "uniform_weights_all_points": count > 0 and torch.allclose(weights, torch.full_like(weights, 1.0 / count), atol=1e-7),
        "one_shared_symmetry_element": int(equivalent["selected_shared_symmetry_element"][0]) == 1,
    }
    report = {
        "audit_passed": all(checks.values()), "checks": checks,
        "perfect_total": float(perfect["loss_total"]), "c2_equivalent_total": float(equivalent["loss_total"]),
        "wrong_total": float(wrong["loss_total"]), "wrong_subset_correspondence": float(subset["loss_correspondence_normalized"]),
        "off_template_surface": float(off["loss_template_surface_normalized"]),
        "selected_shared_symmetry_element": int(equivalent["selected_shared_symmetry_element"][0]),
        "module_pose_gradient_norms": module_gradients,
        "weighting_mode": prediction.weighting_mode, "valid_point_count": count,
        "effective_correspondence_count": float(1.0 / weights.square().sum()),
    }
    (output / "joint_loss_contract_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "joint_loss_contract_audit.md").write_text("# Joint loss contract audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **report}, indent=2))
    return 0 if report["audit_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
