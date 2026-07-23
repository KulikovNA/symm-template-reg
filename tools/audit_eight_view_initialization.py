#!/usr/bin/env python3
"""Evaluate the four-view best checkpoint on eight shell-only views, without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from coordinate_guided_audit_common import load_coordinate_audit_contexts  # noqa: E402
from symm_template_reg.evaluation.active_coordinate import (  # noqa: E402
    active_row,
    active_world_metrics,
    active_world_pairwise_matrices,
    evaluate_active_sample,
    strict_and_practical_stage_gates,
)
from symm_template_reg.models import register_all_modules  # noqa: E402


EXPECTED_FRAMES = (4, 5, 2, 8, 0, 1, 6, 9)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(checkpoint, manifest, output: Path, device) -> dict:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    checkpoint_before, manifest_before = _sha(checkpoint_path), _sha(manifest_path)
    contexts = load_coordinate_audit_contexts(
        checkpoint_path, manifest_path, output, torch.device(device)
    )
    frames = tuple(int(context["sample"]["frame_id"]) for context in contexts)
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"eight-view frame order mismatch: {frames}")
    rows = []
    for context in contexts:
        result = evaluate_active_sample(
            q_aux_O=context["q_aux"], valid_mask=context["mask"],
            target_O=context["target"], observed_C=context["observed"],
            vertices_O=context["vertices"], faces=context["faces"],
            equivalent_pose=context["equivalent_pose"],
            procrustes=context["model"].weighted_procrustes,
            candidate_k=16, projection_chunk_size=256,
        )
        row = active_row(
            result, sample_id=context["sample"]["sample_id"],
            frame_id=int(context["sample"]["frame_id"]),
            T_W_from_C=context["T_W_from_C"], target_leakage_detected=False,
        )
        row["selected_shared_symmetry_element"] = int(
            context["selected_symmetry_element"]
        )
        rows.append(row)
    gates = strict_and_practical_stage_gates(rows, EXPECTED_FRAMES)
    world = active_world_metrics(
        rows, contexts[0]["metadata"], contexts[0]["effective_group"]
    )
    pairwise = active_world_pairwise_matrices(
        rows, contexts[0]["metadata"], contexts[0]["effective_group"]
    )
    scalar_fields = [
        key for key, value in rows[0].items()
        if isinstance(value, (str, bool, int, float))
    ]
    with (output / "eight_view_initialization_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_fields} for row in rows)
    summary = {
        "audit_completed": True,
        "training_performed": False,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "frames": list(frames),
        "sample_count": len(rows),
        "strict_and_practical_gates_at_initialization": gates,
        "active_world_metrics": world,
        "world_pairwise_matrices": pairwise,
        "checkpoint_unchanged": checkpoint_before == _sha(checkpoint_path),
        "manifest_unchanged": manifest_before == _sha(manifest_path),
        "per_sample": rows,
    }
    (output / "eight_view_initialization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Eight-view initialization audit", "", "No training was performed.", "",
        "| frame | shell | q RMSE | q p50 | q p95 | q max | exact p95 | K16 p95 | align | rot | trans | rank | recall | fallback |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['frame_id']} | {row['num_shell_points']} | "
            f"{row['aux_coordinate_rmse_mm']:.6f} | {row['aux_coordinate_p50_mm']:.6f} | "
            f"{row['aux_coordinate_p95_mm']:.6f} | {row['aux_coordinate_max_mm']:.6f} | "
            f"{row['exact_global_projected_correspondence_p95_mm']:.6f} | "
            f"{row['k16_projected_correspondence_p95_mm']:.6f} | "
            f"{row['exact_global_projection_alignment_p95_mm']:.6f} | "
            f"{row['exact_global_projection_rotation_error_deg']:.6f} | "
            f"{row['exact_global_projection_translation_error_mm']:.6f} | "
            f"{row['exact_global_projection_rank']} | "
            f"{row['k16_exact_global_triangle_recall']:.6f} | "
            f"{row['k16_fallback_fraction']:.6f} |"
        )
    (output / "eight_view_initialization_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    result = run_audit(args.checkpoint, args.manifest, output, args.device)
    print(json.dumps({"output_dir": str(output), **result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
