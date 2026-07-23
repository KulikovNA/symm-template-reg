#!/usr/bin/env python3
"""Audit a two-view checkpoint on the prepared four-view set without training."""

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
    evaluate_active_sample,
    four_view_stage_gate,
)
from symm_template_reg.models import register_all_modules  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(checkpoint, manifest, output, device) -> dict:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    checkpoint_before = _sha(checkpoint_path)
    manifest_before = _sha(manifest_path)
    contexts = load_coordinate_audit_contexts(
        checkpoint_path, manifest_path, output, torch.device(device)
    )
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
        active = active_row(
            result, sample_id=context["sample"]["sample_id"],
            frame_id=int(context["sample"]["frame_id"]),
            T_W_from_C=context["T_W_from_C"], target_leakage_detected=False,
        )
        rows.append(
            {
                "sample_id": active["sample_id"],
                "frame_id": active["frame_id"],
                "selected_shared_symmetry_element": context["selected_symmetry_element"],
                "raw_q_aux_rmse_mm": active["aux_coordinate_rmse_mm"],
                "raw_q_aux_p95_mm": active["aux_coordinate_p95_mm"],
                "exact_global_correspondence_p95_mm": active["exact_global_projected_correspondence_p95_mm"],
                "k16_correspondence_p95_mm": active["k16_projected_correspondence_p95_mm"],
                "exact_global_alignment_p95_mm": active["exact_global_projection_alignment_p95_mm"],
                "k16_alignment_p95_mm": active["k16_projection_alignment_p95_mm"],
                "exact_global_rotation_error_deg": active["exact_global_projection_rotation_error_deg"],
                "k16_rotation_error_deg": active["k16_projection_rotation_error_deg"],
                "exact_global_translation_error_mm": active["exact_global_projection_translation_error_mm"],
                "k16_translation_error_mm": active["k16_projection_translation_error_mm"],
                "exact_global_rank": active["exact_global_projection_rank"],
                "k16_rank": active["k16_projection_rank"],
                "surface_membership_p95_mm": active["exact_global_surface_membership_p95_mm"],
                "k16_exact_global_triangle_recall": active["k16_exact_global_triangle_recall"],
                "k16_fallback_fraction": active["k16_fallback_fraction"],
                "active_nonfinite_detected": active["active_nonfinite_detected"],
                "target_leakage_detected": False,
                "exact_global_T_W_from_O": active["exact_global_T_W_from_O"],
                "k16_T_W_from_O": active["k16_T_W_from_O"],
                # Gate helper consumes the canonical active row names.
                **{
                    key: active[key]
                    for key in (
                        "exact_global_projected_correspondence_p95_mm",
                        "exact_global_projection_alignment_p95_mm",
                        "exact_global_projection_rotation_error_deg",
                        "exact_global_projection_translation_error_mm",
                        "exact_global_projection_rank",
                        "exact_global_surface_membership_p95_mm",
                        "exact_global_sample_gate_passed",
                    )
                },
            }
        )
    gate = four_view_stage_gate(rows)
    world = active_world_metrics(
        rows, contexts[0]["metadata"], contexts[0]["effective_group"]
    )
    scalar_fields = [
        key for key, value in rows[0].items() if isinstance(value, (str, bool, int, float))
    ]
    with (output / "four_view_initialization_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in scalar_fields} for row in rows])
    summary = {
        "audit_completed": True,
        "training_performed": False,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "frames": [int(row["frame_id"]) for row in rows],
        "sample_count": len(rows),
        "stage_gate_at_initialization": gate,
        "active_world_metrics": world,
        "checkpoint_unchanged": checkpoint_before == _sha(checkpoint_path),
        "manifest_unchanged": manifest_before == _sha(manifest_path),
        "per_sample": [
            {key: value for key, value in row.items() if key not in {
                "exact_global_T_W_from_O", "k16_T_W_from_O"
            }}
            for row in rows
        ],
    }
    (output / "four_view_initialization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Four-view initialization audit",
        "",
        "No training was performed.",
        "",
        "| frame | q_aux RMSE | q_aux p95 | exact p95 | K16 p95 | alignment p95 | rotation | translation | rank | recall | fallback |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['frame_id']} | {row['raw_q_aux_rmse_mm']:.6f} | "
            f"{row['raw_q_aux_p95_mm']:.6f} | {row['exact_global_correspondence_p95_mm']:.6f} | "
            f"{row['k16_correspondence_p95_mm']:.6f} | {row['exact_global_alignment_p95_mm']:.6f} | "
            f"{row['exact_global_rotation_error_deg']:.6f} | {row['exact_global_translation_error_mm']:.6f} | "
            f"{row['exact_global_rank']} | {row['k16_exact_global_triangle_recall']:.6f} | "
            f"{row['k16_fallback_fraction']:.6f} |"
        )
    (output / "four_view_initialization_report.md").write_text(
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
