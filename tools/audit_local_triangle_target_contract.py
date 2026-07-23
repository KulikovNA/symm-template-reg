#!/usr/bin/env python3
"""Audit local triangle indices, candidates and shared-symmetry target consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.manifest import load_and_validate_manifest  # noqa: E402
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.triangle_targets import (  # noqa: E402
    triangle_target_sets,
)
from symm_template_reg.models.pose.pose_representation import (  # noqa: E402
    invert_transform,
    transform_points,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import (  # noqa: E402
    symmetry_transforms,
)
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    DATASETS,
    build_from_cfg,
)


def _sample(config: dict[str, Any], manifest_path: str, cache: Path):
    cfg = deepcopy(config["dataset"])
    data = config["data"]
    cfg["fragment_mesh_filter"] = deepcopy(data["fragment_mesh_filter"])
    cfg["observed_filter"] = deepcopy(data["observed_filter"])
    cfg["symmetry_region_activity"] = deepcopy(data.get("symmetry_region_activity", {}))
    cfg["fragment_mesh_cache_dir"] = str(cache)
    dataset = build_from_cfg(cfg, DATASETS)
    manifest, _ = load_and_validate_manifest(manifest_path, config, dataset)
    indices = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    sample_id = str(manifest["samples"][0]["sample_id"])
    return dataset[indices[sample_id]], manifest


def _padded_target(batch: dict[str, Any]) -> torch.Tensor:
    target = batch["gt"]["points_O_corresponding"]
    return target.to_padded()["points"] if hasattr(target, "to_padded") else target


def _selected_element(run: Path) -> int:
    path = run / "best_evaluation" / "per_sample_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    return int(float(row["selected_shared_symmetry_element"]))


def shared_symmetry_target(
    raw_target: torch.Tensor,
    metadata: Any,
    group_value: Any,
    selected_element: int,
) -> torch.Tensor:
    group = parse_rotation_group(group_value)
    symmetries = symmetry_transforms(
        group,
        metadata.axis.direction,
        metadata.axis.origin,
        so2_num_samples=36 if group.type == "SO2" else None,
        dtype=raw_target.dtype,
        device=raw_target.device,
    )
    if not 0 <= selected_element < len(symmetries):
        raise ValueError(f"selected symmetry element {selected_element} is out of range")
    return transform_points(
        invert_transform(symmetries[selected_element : selected_element + 1]),
        raw_target.unsqueeze(0),
    )[0]


@torch.no_grad()
def load_contract_context(
    run: Path,
    checkpoint: Path,
    manifest_path: str | None,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    manifest_path = manifest_path or str(config["data"]["train_manifest"])
    sample, manifest = _sample(
        config, manifest_path, output / "cache" / "fragment_mesh_metadata"
    )
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    batch = move_to_device(collate([sample]), device)
    model = build_model(config["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    head = model.correspondence_head
    head.teacher_forcing_probability = 1.0
    model.train()
    torch.manual_seed(0)
    train_prediction = model(batch)
    torch.manual_seed(0)
    repeated_train_prediction = model(batch)
    model.eval()
    evaluation_prediction = model(batch)
    mask = train_prediction.observed_valid_mask[0]
    raw_target = _padded_target(batch)[0]
    selected = _selected_element(run)
    target = shared_symmetry_target(
        raw_target,
        batch["template_symmetry_metadata"][0],
        batch["gt"]["effective_symmetry_group"][0],
        selected,
    )
    return {
        "config": config,
        "sample": sample,
        "manifest": manifest,
        "batch": batch,
        "model": model,
        "checkpoint_payload": payload,
        "mask": mask,
        "raw_target": raw_target,
        "target": target,
        "selected_symmetry_element": selected,
        "train_prediction": train_prediction,
        "repeated_train_prediction": repeated_train_prediction,
        "evaluation_prediction": evaluation_prediction,
    }


def _duplicates(row: torch.Tensor) -> int:
    valid = row[row >= 0]
    return int(valid.numel() - torch.unique(valid).numel())


def audit_contract(context: dict[str, Any], output: Path, run: Path, checkpoint: Path):
    mask = context["mask"]
    batch = context["batch"]
    target = context["target"][mask]
    train_aux = context["train_prediction"].correspondence_auxiliary
    repeated_aux = context["repeated_train_prediction"].correspondence_auxiliary
    eval_aux = context["evaluation_prediction"].correspondence_auxiliary
    assert train_aux is not None and repeated_aux is not None and eval_aux is not None
    vertices = batch["template_mesh_vertices_O"][0]
    faces = batch["template_mesh_faces"][0].long()
    tolerance = float(
        context["config"]["loss"]["joint_surface_correspondence_pose_v3"].get(
            "triangle_target_tolerance_m", 0.00015
        )
    )
    target_sets = triangle_target_sets(
        target, vertices, faces, tolerance_m=tolerance, point_chunk_size=256
    )
    gt_triangle = target_sets["face_ids"]
    candidates = train_aux["candidate_triangle_ids"][0, mask]
    candidate_mask = train_aux.get("candidate_triangle_mask")
    candidate_mask = (
        candidates.ge(0) if candidate_mask is None else candidate_mask[0, mask]
    )
    exact_occurrences = candidates.eq(gt_triangle[:, None]) & candidate_mask
    centroids = vertices[faces].mean(1)
    fallback_distance = torch.linalg.vector_norm(
        centroids[candidates.clamp_min(0)] - target[:, None], dim=-1
    ).masked_fill(~candidate_mask, float("inf"))
    local_target = torch.where(
        exact_occurrences.any(-1),
        exact_occurrences.to(torch.int64).argmax(-1),
        fallback_distance.argmin(-1),
    )
    point_rows = torch.arange(len(target), device=target.device)
    target_candidate = candidates[point_rows, local_target]
    valid_alternative = target_sets["valid_triangle_mask"].gather(
        -1, target_candidate.clamp_min(0)[:, None]
    ).squeeze(-1) & target_candidate.ge(0)
    mismatch = target_candidate.ne(gt_triangle) & ~valid_alternative
    duplicate_counts = torch.tensor(
        [_duplicates(row) for row in candidates], device=target.device
    )
    train_repeat_same = train_aux["candidate_triangle_ids"][0, mask].eq(
        repeated_aux["candidate_triangle_ids"][0, mask]
    ).all(-1)
    train_eval_same = train_aux["candidate_triangle_ids"][0, mask].eq(
        eval_aux["candidate_triangle_ids"][0, mask]
    ).all(-1)
    head_symmetry = train_aux.get("teacher_forcing_selected_symmetry_element")
    head_selected = -1 if head_symmetry is None else int(head_symmetry[0])
    head_gt = train_aux.get("teacher_forcing_gt_triangle_ids")
    head_gt = torch.full_like(gt_triangle, -1) if head_gt is None else head_gt[0, mask]
    teacher_exact_matches_loss = head_gt.eq(gt_triangle)
    local_loss = float(
        context["config"]["loss"]["joint_surface_correspondence_pose_v3"].get(
            "triangle_target_tolerance_m", tolerance
        )
    )
    del local_loss
    rows: list[dict[str, Any]] = []
    for index in range(len(target)):
        valid_ids = candidates[index, candidate_mask[index]].detach().cpu().tolist()
        rows.append(
            {
                "point_index": index,
                "selected_shared_symmetry_element": context["selected_symmetry_element"],
                "teacher_forcing_symmetry_element": head_selected,
                "q_gt_s_x_m": float(target[index, 0]),
                "q_gt_s_y_m": float(target[index, 1]),
                "q_gt_s_z_m": float(target[index, 2]),
                "gt_triangle_global_id": int(gt_triangle[index]),
                "candidate_global_ids": " ".join(map(str, valid_ids)),
                "candidate_count": len(valid_ids),
                "duplicate_candidate_count": int(duplicate_counts[index]),
                "local_target_index": int(local_target[index]),
                "candidate_id_at_local_target_index": int(target_candidate[index]),
                "target_index_matches_exact_or_valid_alternative": bool(
                    ~mismatch[index]
                ),
                "distance_q_gt_to_exact_triangle_mm": float(
                    target_sets["distances"][index] * 1000.0
                ),
                "gt_triangle_occurrence_count": int(exact_occurrences[index].sum()),
                "candidate_order_deterministic": bool(train_repeat_same[index]),
                "train_evaluation_candidates_identical": bool(train_eval_same[index]),
                "teacher_forcing_exact_triangle_matches_loss_target": bool(
                    teacher_exact_matches_loss[index]
                ),
            }
        )
    mismatch_count = int(mismatch.sum())
    symmetry_consistent = (
        head_selected == context["selected_symmetry_element"]
        and bool(teacher_exact_matches_loss.all())
    )
    summary = {
        "audit_passed": (
            mismatch_count == 0
            and symmetry_consistent
            and bool(train_repeat_same.all())
            and bool(train_eval_same.all())
        ),
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": context["checkpoint_payload"].get("epoch"),
        "sample_id": context["sample"].get("sample_id"),
        "observed_point_count": int(mask.sum()),
        "selected_shared_symmetry_element": context["selected_symmetry_element"],
        "teacher_forcing_symmetry_element": head_selected,
        "shared_symmetry_target_consistent": symmetry_consistent,
        "teacher_forcing_exact_triangle_match_fraction": float(
            teacher_exact_matches_loss.float().mean()
        ),
        "target_index_mismatch_count": mismatch_count,
        "target_index_mismatch_fraction": mismatch_count / max(len(target), 1),
        "points_with_duplicate_candidates": int(duplicate_counts.gt(0).sum()),
        "mean_duplicate_candidate_count": float(duplicate_counts.float().mean()),
        "max_duplicate_candidate_count": int(duplicate_counts.max()),
        "candidate_order_deterministic_fraction": float(train_repeat_same.float().mean()),
        "train_evaluation_candidates_identical_fraction": float(
            train_eval_same.float().mean()
        ),
        "training_blocked": mismatch_count > 0 or not symmetry_consistent,
        "diagnosis": (
            "teacher_forcing_and_loss_use_different_shared_symmetry_elements"
            if not symmetry_consistent
            else "local_candidate_target_index_mismatch"
            if mismatch_count
            else "contract_verified"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "local_triangle_target_contract_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "local_triangle_target_contract_per_point.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "local_triangle_target_contract_report.md").write_text(
        "# Local triangle target contract\n\n"
        f"- diagnosis: `{summary['diagnosis']}`\n"
        f"- loss/shared S: `{summary['selected_shared_symmetry_element']}`\n"
        f"- teacher-forcing S: `{summary['teacher_forcing_symmetry_element']}`\n"
        f"- target-index mismatches: `{summary['target_index_mismatch_count']}`\n"
        f"- duplicate candidates (mean): `{summary['mean_duplicate_candidate_count']:.4f}`\n"
        f"- train/eval identical fraction: "
        f"`{summary['train_evaluation_candidates_identical_fraction']:.8f}`\n\n"
        "Training is blocked whenever a target-index mismatch or shared-S mismatch is present.\n",
        encoding="utf-8",
    )
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    run = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    register_all_modules()
    context = load_contract_context(
        run, checkpoint, args.manifest, output, torch.device(args.device)
    )
    summary, _ = audit_contract(context, output, run, checkpoint)
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0 if summary["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
