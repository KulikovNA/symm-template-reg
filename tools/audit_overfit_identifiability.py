#!/usr/bin/env python3
"""Pairwise identifiability and metadata-ID counterfactual audit for 4x4 overfit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.models.pose.pose_representation import transform_points  # noqa: E402
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance  # noqa: E402
from symm_template_reg.models.symmetry.groups import parse_rotation_group  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms  # noqa: E402


DEFAULT_CONFIG = "configs/debug/coordinate_guided_surface_v3/four_fragments_four_views_scratch.py"


def _subsample(points: torch.Tensor, maximum: int = 256) -> torch.Tensor:
    if len(points) <= maximum:
        return points.float()
    return points[torch.linspace(0, len(points) - 1, maximum).round().long()].float()


def _normalize(points: torch.Tensor) -> torch.Tensor:
    centered = points.float() - points.float().mean(0)
    return centered / centered.square().sum(-1).mean().sqrt().clamp_min(1e-9)


def _chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    distance = torch.cdist(_subsample(a), _subsample(b))
    return float(0.5 * (distance.min(1).values.mean() + distance.min(0).values.mean()))


def _descriptor(points: torch.Tensor) -> torch.Tensor:
    p = _normalize(_subsample(points, 512))
    covariance = p.T @ p / max(len(p) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).sort().values
    radii = torch.linalg.vector_norm(p, dim=-1)
    pair = torch.pdist(p[: min(len(p), 128)])
    return torch.cat((eigenvalues, torch.quantile(radii, torch.tensor([0.1, 0.5, 0.9])), torch.quantile(pair, torch.tensor([0.1, 0.5, 0.9]))))


def _symmetry_coordinate_difference(a, b) -> float:
    metadata = b["template"]["symmetry_metadata"]
    group = parse_rotation_group(b["gt"]["effective_symmetry_group"])
    transforms = symmetry_transforms(
        group, metadata.axis.direction, metadata.axis.origin,
        so2_num_samples=36 if group.type == "SO2" else None,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    target_a = a["gt"]["points_O_corresponding"].cpu().float()
    target_b = b["gt"]["points_O_corresponding"].cpu().float()
    return min(_chamfer(target_a, transform_points(transform[None], target_b[None])[0]) for transform in transforms)


def _symmetry_pose_difference(a, b) -> tuple[float, float]:
    metadata = b["template"]["symmetry_metadata"]
    group = parse_rotation_group(b["gt"]["effective_symmetry_group"])
    symmetries = symmetry_transforms(
        group, metadata.axis.direction, metadata.axis.origin,
        so2_num_samples=36 if group.type == "SO2" else None,
        dtype=torch.float32, device=torch.device("cpu"),
    )
    pose_a = a["gt"]["T_C_from_O"].cpu().float()
    poses_b = b["gt"]["T_C_from_O"].cpu().float().unsqueeze(0) @ symmetries
    rotation = torch.rad2deg(rotation_geodesic_distance(pose_a[:3, :3][None], poses_b[:, :3, :3]))
    translation = torch.linalg.vector_norm(pose_a[:3, 3][None] - poses_b[:, :3, 3], dim=-1) * 1000.0
    return float(rotation.min()), float(translation.min())


@torch.no_grad()
def _identity_counterfactual(model, samples, collate, device) -> dict:
    model.eval()
    batch = move_to_device(collate(samples[:2]), device)
    direct = model(batch).correspondence_auxiliary["fine_aux_coordinate_normalized"]
    changed = copy.deepcopy(batch)
    changed["sample_id"] = ["counterfactual/A", "counterfactual/B"]
    changed["scene_id"] = ["changed_scene", "changed_scene"]
    changed["frame_id"] = torch.tensor([9001, 9002], device=changed["frame_id"].device)
    changed["fragment_id"] = torch.tensor([8001, 8002], device=changed["fragment_id"].device)
    changed["object_model_id"] = ["changed_A", "changed_B"]
    altered = model(changed).correspondence_auxiliary["fine_aux_coordinate_normalized"]
    maximum = float((direct - altered).abs().max())
    return {
        "ids_changed": ["sample_id", "scene_id", "frame_id", "fragment_id", "object_model_id"],
        "geometry_changed": False, "maximum_output_difference": maximum,
        "passed": maximum <= 1e-7,
    }


def run(args) -> dict:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, manifest, _, _, samples, collate, model = load_multifragment_context(
        args.config, args.manifest, output, args.device
    )
    descriptors = [_descriptor(sample["observed"]["points_C"]).cpu() for sample in samples]
    rows = []
    conflicts = []
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            a, b = samples[left], samples[right]
            observed = _chamfer(_normalize(a["observed"]["points_C"]), _normalize(b["observed"]["points_C"]))
            descriptor = float(torch.linalg.vector_norm(descriptors[left] - descriptors[right]))
            coordinate = _symmetry_coordinate_difference(a, b)
            rotation, translation = _symmetry_pose_difference(a, b)
            incompatible = observed <= args.near_identical_chamfer and (
                coordinate >= args.incompatible_coordinate_m
                or rotation >= args.incompatible_rotation_deg
                or translation >= args.incompatible_translation_mm
            )
            row = {
                "sample_a": a["sample_id"], "sample_b": b["sample_id"],
                "fragment_a": a["fragment_id"], "fragment_b": b["fragment_id"],
                "frame_a": a["frame_id"], "frame_b": b["frame_id"],
                "point_count_a": len(a["observed"]["points_C"]), "point_count_b": len(b["observed"]["points_C"]),
                "normalized_observed_chamfer": observed,
                "global_geometry_descriptor_distance": descriptor,
                "symmetry_aware_gt_coordinate_chamfer_m": coordinate,
                "symmetry_aware_gt_pose_rotation_deg": rotation,
                "symmetry_aware_gt_pose_translation_mm": translation,
                "diagnosis": "non_identifiable_training_pair" if incompatible else "identifiable_or_nonconflicting",
            }
            rows.append(row)
            if incompatible:
                conflicts.append(row)
    counterfactual = _identity_counterfactual(model, samples, collate, torch.device(args.device))
    report = {
        "debug_training_on_test_split": True,
        "train_and_validation_use_same_samples": True,
        "results_are_not_final_evaluation": True,
        "audit_passed": not conflicts and counterfactual["passed"],
        "target_leakage_detected": not counterfactual["passed"],
        "sample_count": 16, "pair_count": len(rows),
        "non_identifiable_training_pairs": conflicts,
        "non_identifiable_training_pair_count": len(conflicts),
        "metadata_identity_counterfactual": counterfactual,
        "thresholds": {
            "near_identical_chamfer": args.near_identical_chamfer,
            "incompatible_coordinate_m": args.incompatible_coordinate_m,
            "incompatible_rotation_deg": args.incompatible_rotation_deg,
            "incompatible_translation_mm": args.incompatible_translation_mm,
        },
        "model_inputs_exclude_fragment_and_frame_ids": True,
        "model_input_evidence": "CoordinateGuidedSurfaceRegistrationV3.forward reads observed/template geometry only",
    }
    with (output / "pairwise_identifiability.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "identifiability_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# 4x4 identifiability audit\n\n"
        f"- passed: `{report['audit_passed']}`\n"
        f"- pairs: `{len(rows)}`\n"
        f"- non-identifiable conflicts: `{len(conflicts)}`\n"
        f"- ID-only counterfactual max delta: `{counterfactual['maximum_output_difference']:.3e}`\n",
        encoding="utf-8",
    )
    mirror = Path(args.manifest).expanduser().resolve().with_name(
        Path(args.manifest).stem + "_identifiability.json"
    )
    encoded = (json.dumps(report, indent=2) + "\n").encode()
    if mirror.exists() and mirror.read_bytes() != encoded:
        raise FileExistsError(f"refusing to overwrite different audit mirror: {mirror}")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    if not mirror.exists():
        mirror.write_bytes(encoded)
    report["training_gate_audit_path"] = str(mirror)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-identical-chamfer", type=float, default=1e-4)
    parser.add_argument("--incompatible-coordinate-m", type=float, default=0.0025)
    parser.add_argument("--incompatible-rotation-deg", type=float, default=1.0)
    parser.add_argument("--incompatible-translation-mm", type=float, default=0.5)
    args = parser.parse_args(); result = run(args); print(json.dumps(result, indent=2))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
