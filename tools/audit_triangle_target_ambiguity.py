#!/usr/bin/env python3
"""Measure tolerance-aware triangle target ambiguity for a local Stage B run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_local_triangle_target_contract import load_contract_context  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.geometry.triangle_targets import (  # noqa: E402
    local_valid_triangle_mask,
    triangle_target_sets,
)
from symm_template_reg.models.geometry.patch_targets import (  # noqa: E402
    multi_positive_softmax_loss,
)


@torch.no_grad()
def audit_ambiguity(context, output: Path, run: Path, checkpoint: Path, tolerance: float):
    mask = context["mask"]
    target = context["target"][mask]
    batch = context["batch"]
    prediction = context["train_prediction"]
    auxiliary = prediction.correspondence_auxiliary
    assert auxiliary is not None
    vertices = batch["template_mesh_vertices_O"][0]
    faces = batch["template_mesh_faces"][0].long()
    targets = triangle_target_sets(
        target,
        vertices,
        faces,
        tolerance_m=tolerance,
        point_chunk_size=256,
    )
    valid_global = targets["valid_triangle_mask"]
    valid_count = valid_global.sum(-1)
    candidates = auxiliary["candidate_triangle_ids"][0, mask]
    candidate_mask = auxiliary.get("candidate_triangle_mask")
    candidate_mask = candidates.ge(0) if candidate_mask is None else candidate_mask[0, mask]
    valid_local = local_valid_triangle_mask(candidates, valid_global) & candidate_mask
    logits = auxiliary["fine_local_logits"][0, mask]
    selected_local = logits.argmax(-1)
    row = torch.arange(len(logits), device=logits.device)
    selected_global = candidates[row, selected_local]
    single_hit = selected_global.eq(targets["face_ids"])
    valid_top1 = valid_local.gather(-1, selected_local[:, None]).squeeze(-1)
    top4 = logits.topk(min(4, logits.shape[-1]), dim=-1).indices
    valid_top4 = valid_local.gather(-1, top4).any(-1)
    has_valid = valid_local.any(-1)
    set_loss_rows = torch.full(
        (len(logits),), float("inf"), dtype=logits.dtype, device=logits.device
    )
    if bool(has_valid.any()):
        set_loss_rows[has_valid] = multi_positive_softmax_loss(
            logits[has_valid], valid_local[has_valid], reduction="none"
        )
    candidate_count = candidate_mask.sum(-1)
    random_ce = candidate_count.float().log()
    finite_rows = torch.isfinite(set_loss_rows)
    trained_loss = float(set_loss_rows[finite_rows].mean()) if bool(finite_rows.any()) else math.inf
    random_loss = float(random_ce.mean())
    rows = []
    for index in range(len(target)):
        valid_ids = torch.nonzero(valid_global[index], as_tuple=False).flatten().tolist()
        rows.append(
            {
                "point_index": index,
                "exact_nearest_triangle": int(targets["face_ids"][index]),
                "valid_triangle_ids": " ".join(map(str, valid_ids)),
                "valid_triangle_count": len(valid_ids),
                "adjacent_valid_triangle_count": int(
                    targets["adjacent_valid_mask"][index].sum()
                ),
                "best_triangle_distance_mm": float(targets["distances"][index] * 1000.0),
                "predicted_triangle": int(selected_global[index]),
                "single_owner_top1_correct": bool(single_hit[index]),
                "valid_set_top1_correct": bool(valid_top1[index]),
                "valid_set_top4_hit": bool(valid_top4[index]),
                "valid_triangle_in_candidates": bool(has_valid[index]),
                "candidate_count": int(candidate_count[index]),
                "set_valued_triangle_ce": float(set_loss_rows[index]),
                "uniform_random_ce": float(random_ce[index]),
            }
        )
    summary = {
        "audit_passed": True,
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "sample_id": context["sample"].get("sample_id"),
        "selected_shared_symmetry_element": context["selected_symmetry_element"],
        "triangle_target_tolerance_m": tolerance,
        "tolerance_justification": (
            "0.15 mm equals the upper scale of the existing GT-to-template projection audit"
        ),
        "observed_point_count": len(target),
        "single_owner_triangle_top1": float(single_hit.float().mean()),
        "valid_triangle_set_top1": float(valid_top1.float().mean()),
        "valid_triangle_set_top4": float(valid_top4.float().mean()),
        "mean_valid_triangle_count": float(valid_count.float().mean()),
        "max_valid_triangle_count": int(valid_count.max()),
        "fraction_with_multiple_valid_triangles": float(valid_count.gt(1).float().mean()),
        "valid_triangle_candidate_recall": float(has_valid.float().mean()),
        "trained_set_valued_triangle_ce": trained_loss,
        "random_cross_entropy": random_loss,
        "random_cross_entropy_ln32": math.log(32.0),
        "warnings": (
            ["local_triangle_classifier_worse_than_uniform"]
            if trained_loss > random_loss
            else []
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "triangle_target_ambiguity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "triangle_target_ambiguity_per_point.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "triangle_target_ambiguity_report.md").write_text(
        "# Triangle target ambiguity\n\n"
        f"- tolerance: `{tolerance * 1000.0:.3f} mm`\n"
        f"- single-owner top-1: `{summary['single_owner_triangle_top1']:.8f}`\n"
        f"- valid-set top-1: `{summary['valid_triangle_set_top1']:.8f}`\n"
        f"- valid-set top-4: `{summary['valid_triangle_set_top4']:.8f}`\n"
        f"- mean valid triangle count: `{summary['mean_valid_triangle_count']:.4f}`\n"
        f"- trained CE / random CE: `{trained_loss:.6f} / {random_loss:.6f}`\n"
        f"- warnings: `{summary['warnings']}`\n",
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
    parser.add_argument("--triangle-target-tolerance-m", type=float, default=0.00015)
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
    summary, _ = audit_ambiguity(
        context, output, run, checkpoint, args.triangle_target_tolerance_m
    )
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
