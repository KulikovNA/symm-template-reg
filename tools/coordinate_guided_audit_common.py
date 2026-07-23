"""Shared read-only loading and metrics for coordinate-guided audits."""

from __future__ import annotations

import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_local_triangle_target_contract import (  # noqa: E402
    _padded_target, _sample, shared_symmetry_target,
)
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.models import build_model  # noqa: E402
from symm_template_reg.models.pose.pose_representation import (  # noqa: E402
    invert_transform,
    transform_points,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms  # noqa: E402
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)


def quantile_metrics_mm(distance_m: torch.Tensor, prefix: str) -> dict[str, float]:
    values = distance_m.detach().float()
    return {
        f"{prefix}_rmse_mm": float(values.square().mean().sqrt() * 1000.0),
        f"{prefix}_p50_mm": float(torch.quantile(values, .50) * 1000.0),
        f"{prefix}_p95_mm": float(torch.quantile(values, .95) * 1000.0),
        f"{prefix}_max_mm": float(values.max() * 1000.0),
    }


def _selected_element(run: Path) -> int:
    with (run / "best_evaluation" / "per_sample_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        return int(float(next(csv.DictReader(stream))["selected_shared_symmetry_element"]))


def _selected_elements(run: Path) -> dict[str, int]:
    path = run / "best_evaluation" / "per_sample_metrics.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            str(row["sample_id"]): int(float(row["selected_shared_symmetry_element"]))
            for row in csv.DictReader(stream)
            if row.get("selected_shared_symmetry_element") not in {None, ""}
        }


@torch.no_grad()
def load_coordinate_audit_contexts(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> list[dict]:
    """Load one checkpoint once and return active-path contexts for all samples."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run = checkpoint.parents[1]
    output = Path(output_dir).expanduser().resolve()
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    dataset_cfg = deepcopy(config["dataset"])
    data_cfg = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data_cfg["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data_cfg.get("symmetry_region_activity", {})
    )
    dataset_cfg["fragment_mesh_cache_dir"] = str(
        output / "cache" / "fragment_mesh_metadata"
    )
    dataset = build_from_cfg(dataset_cfg, DATASETS)
    manifest, _ = load_and_validate_manifest(
        str(Path(manifest_path).expanduser().resolve()), config, dataset
    )
    indices = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    model = build_model(config["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    head = model.correspondence_head
    if hasattr(head, "teacher_forcing_probability"):
        head.teacher_forcing_probability = 0.0
        head.teacher_forcing_during_evaluation = False
    known_elements = _selected_elements(run)
    contexts: list[dict] = []
    for manifest_sample in manifest["samples"]:
        sample_id = str(manifest_sample["sample_id"])
        sample = dataset[indices[sample_id]]
        batch = move_to_device(collate([sample]), device)
        prediction = model(batch)
        auxiliary = prediction.correspondence_auxiliary
        if auxiliary is None or "fine_aux_coordinate_normalized" not in auxiliary:
            raise ValueError("checkpoint does not expose fine auxiliary coordinates")
        mask = prediction.observed_valid_mask[0]
        vertices = batch["template_mesh_vertices_O"][0].to(device)
        faces = batch["template_mesh_faces"][0].to(device=device, dtype=torch.long)
        extent = (vertices.amax(0) - vertices.amin(0)).clamp_min(1e-8)
        q_aux = (
            0.5 * (auxiliary["fine_aux_coordinate_normalized"][0] + 1.0) * extent
            + vertices.amin(0)
        )
        raw_target = _padded_target(batch)[0]
        metadata = batch["template_symmetry_metadata"][0]
        group = parse_rotation_group(batch["gt"]["effective_symmetry_group"][0])
        symmetries = symmetry_transforms(
            group,
            metadata.axis.direction,
            metadata.axis.origin,
            so2_num_samples=36 if group.type == "SO2" else None,
            dtype=q_aux.dtype,
            device=device,
        )
        targets = transform_points(invert_transform(symmetries), raw_target[None])
        selected = known_elements.get(sample_id)
        if selected is None:
            mean_errors = torch.linalg.vector_norm(
                q_aux[mask][None] - targets[:, mask], dim=-1
            ).mean(-1)
            selected = int(mean_errors.argmin())
        target = targets[selected]
        equivalent_pose = batch["gt"]["T_C_from_O"][0] @ symmetries[selected]
        contexts.append(
            {
                "checkpoint": checkpoint,
                "run": run,
                "config": config,
                "sample": sample,
                "manifest_sample": manifest_sample,
                "manifest": manifest,
                "batch": batch,
                "model": model,
                "checkpoint_payload": payload,
                "prediction": prediction,
                "predicted_aux": auxiliary,
                "mask": mask,
                "vertices": vertices,
                "faces": faces,
                "q_aux": q_aux,
                "target": target,
                "observed": batch["observed"].to_padded()["points"][0],
                "selected_symmetry_element": selected,
                "equivalent_pose": equivalent_pose,
                "metadata": metadata,
                "effective_group": batch["gt"]["effective_symmetry_group"][0],
                "T_W_from_C": batch["gt"]["T_W_from_C"][0],
            }
        )
    return contexts


@torch.no_grad()
def load_f1_audit_context(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
) -> dict:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    run = checkpoint.parents[1]
    output = Path(output_dir).expanduser().resolve()
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    sample, manifest = _sample(
        config, str(Path(manifest_path).expanduser().resolve()),
        output / "cache" / "fragment_mesh_metadata",
    )
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), device)
    model = build_model(config["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    head = model.correspondence_head
    head.teacher_forcing_probability = 1.0
    head.teacher_forcing_during_evaluation = True
    teacher_prediction = model(batch)
    teacher_aux = teacher_prediction.correspondence_auxiliary
    if teacher_aux is None or "fine_aux_coordinate_normalized" not in teacher_aux:
        raise ValueError("checkpoint does not expose F1 auxiliary coordinates")
    mask = teacher_prediction.observed_valid_mask[0]
    vertices = batch["template_mesh_vertices_O"][0].to(device)
    faces = batch["template_mesh_faces"][0].to(device=device, dtype=torch.long)
    bbox_min, bbox_max = vertices.amin(0), vertices.amax(0)
    extent = (bbox_max - bbox_min).clamp_min(1e-8)
    q_normalized = teacher_aux["fine_aux_coordinate_normalized"][0]
    q_aux = .5 * (q_normalized + 1.0) * extent + bbox_min
    selected = _selected_element(run)
    raw_target = _padded_target(batch)[0]
    target = shared_symmetry_target(
        raw_target, batch["template_symmetry_metadata"][0],
        batch["gt"]["effective_symmetry_group"][0], selected,
    )
    group = parse_rotation_group(batch["gt"]["effective_symmetry_group"][0])
    metadata = batch["template_symmetry_metadata"][0]
    symmetries = symmetry_transforms(
        group, metadata.axis.direction, metadata.axis.origin,
        so2_num_samples=36 if group.type == "SO2" else None,
        dtype=q_aux.dtype, device=device,
    )
    equivalent_pose = batch["gt"]["T_C_from_O"][0] @ symmetries[selected]
    observed = batch["observed"].to_padded()["points"][0]
    head.teacher_forcing_probability = 0.0
    head.teacher_forcing_during_evaluation = False
    predicted_candidate_prediction = model(batch)
    predicted_aux = predicted_candidate_prediction.correspondence_auxiliary
    return {
        "checkpoint": checkpoint, "run": run, "config": config,
        "sample": sample, "manifest": manifest, "batch": batch,
        "model": model, "checkpoint_payload": payload, "mask": mask,
        "vertices": vertices, "faces": faces, "q_aux": q_aux,
        "target": target, "observed": observed,
        "selected_symmetry_element": selected,
        "equivalent_pose": equivalent_pose, "metadata": metadata,
        "teacher_prediction": teacher_prediction, "teacher_aux": teacher_aux,
        "predicted_prediction": predicted_candidate_prediction,
        "predicted_aux": predicted_aux,
    }


def pose_and_alignment_metrics(
    q: torch.Tensor,
    observed: torch.Tensor,
    valid_mask: torch.Tensor,
    equivalent_pose: torch.Tensor,
    metadata,
    procrustes,
    *,
    prefix: str,
) -> tuple[dict, dict]:
    weights = valid_mask.to(q.dtype)
    solution = procrustes.solve(
        q.unsqueeze(0).float(), observed.unsqueeze(0).float(),
        weights.unsqueeze(0).float(), valid_mask.unsqueeze(0),
    )
    pose = solution["transform"][0].to(q)
    relative_rotation = pose[:3, :3].T @ equivalent_pose[:3, :3]
    trace = relative_rotation.trace()
    rotation = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1)))
    translation_delta = pose[:3, 3] - equivalent_pose[:3, 3]
    axis_O = torch.as_tensor(metadata.axis.direction, dtype=q.dtype, device=q.device)
    axis_O = axis_O / torch.linalg.vector_norm(axis_O).clamp_min(1e-12)
    axis_C = equivalent_pose[:3, :3] @ axis_O
    signed_along = torch.dot(translation_delta, axis_C)
    perpendicular = torch.linalg.vector_norm(translation_delta - signed_along * axis_C)
    reconstructed = transform_points(pose.unsqueeze(0), q.unsqueeze(0))[0]
    alignment = torch.linalg.vector_norm(reconstructed - observed, dim=-1)[valid_mask]
    metrics = {
        f"{prefix}_rotation_error_deg": float(rotation),
        f"{prefix}_translation_total_mm": float(torch.linalg.vector_norm(translation_delta) * 1000),
        f"{prefix}_translation_along_axis_mm": float(signed_along.abs() * 1000),
        f"{prefix}_translation_perpendicular_axis_mm": float(perpendicular * 1000),
        f"{prefix}_alignment_rmse_mm": float(alignment.square().mean().sqrt() * 1000),
        f"{prefix}_alignment_p95_mm": float(torch.quantile(alignment.float(), .95) * 1000),
        f"{prefix}_correspondence_rank": int(solution["rank"][0]),
        f"{prefix}_procrustes_rank_valid": bool(solution["rank_valid"][0]),
        f"{prefix}_procrustes_determinant": float(solution["determinant"][0]),
        f"{prefix}_procrustes_orthogonality": float(solution["orthogonality_error"][0]),
        f"{prefix}_covariance_eigenvalues": torch.linalg.eigvalsh(
            torch.cov(q[valid_mask].float().T)
        ).clamp_min(0).tolist(),
    }
    return metrics, {"pose": pose, "alignment_distance_m": alignment}


__all__ = [
    "load_coordinate_audit_contexts", "load_f1_audit_context",
    "pose_and_alignment_metrics", "quantile_metrics_mm",
]
