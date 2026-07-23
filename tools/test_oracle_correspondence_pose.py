#!/usr/bin/env python3
"""Validate GT-correspondence Weighted Procrustes on real manifest samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402
from symm_template_reg.models.pose.metrics import rotation_error_deg, translation_error  # noqa: E402
from symm_template_reg.models.pose.pose_representation import transform_points  # noqa: E402
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import (  # noqa: E402
    equivalent_gt_pose_set,
    symmetry_transforms,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group  # noqa: E402
from symm_template_reg.registry import DATASETS, build_from_cfg  # noqa: E402


def _solve(solver, source, target, weights, mask):
    return solver.solve(
        source[None], target[None], weights[None], mask[None],
        fail_on_degenerate=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--config",
        default="configs/debug/conditioned_pose_v2/01_k1_direct_equal_exposure.py",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    config["dataset"]["fragment_mesh_cache_dir"] = str(output / "cache")
    for key in ("fragment_mesh_filter", "observed_filter", "symmetry_region_activity"):
        if key in config["data"]:
            config["dataset"][key] = config["data"][key]
    register_all_modules()
    dataset = build_from_cfg(config["dataset"], DATASETS)
    manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
    by_id = {record.sample_id: index for index, record in enumerate(dataset.sample_records)}
    solver = WeightedProcrustes(
        minimum_effective_points=3.0, rank_tolerance=1e-10, fail_on_degenerate=True
    ).to(device)
    rows = []
    generator = torch.Generator(device=device).manual_seed(0)
    for item in manifest["samples"]:
        sample = dataset[by_id[item["sample_id"]]]
        source_gt = sample["gt"]["points_O_corresponding"].to(device=device, dtype=torch.float64)
        observed = sample["observed"]["points_C"].to(device=device, dtype=torch.float64)
        gt_pose = sample["gt"]["T_C_from_O"].to(device=device, dtype=torch.float64)
        metadata = sample["template"]["symmetry_metadata"]
        group = parse_rotation_group(sample["gt"]["effective_symmetry_group"])
        pose_set = equivalent_gt_pose_set(
            gt_pose, metadata, effective_group=group, so2_num_samples=36
        )
        symmetry_T = symmetry_transforms(
            group, metadata.axis.direction, metadata.axis.origin,
            so2_num_samples=36 if group.type == "SO2" else None,
            dtype=torch.float64, device=device,
        )
        for symmetry_index, (S, expected_pose) in enumerate(zip(symmetry_T, pose_set.poses)):
            canonical = transform_points(torch.linalg.inv(S), source_gt)
            mask = torch.ones(len(canonical), dtype=torch.bool, device=device)
            weights = torch.ones(len(canonical), dtype=torch.float64, device=device)
            result = _solve(solver, canonical, observed, weights, mask)
            predicted = result["transform"][0]
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "frame_id": int(item["frame_id"]),
                    "symmetry_index": symmetry_index,
                    "num_points": len(canonical),
                    "rotation_error_deg": float(rotation_error_deg(predicted, expected_pose)),
                    "translation_error_mm": float(translation_error(predicted, expected_pose) * 1000.0),
                    "determinant": float(result["determinant"][0]),
                    "orthogonality_error": float(result["orthogonality_error"][0]),
                    "source_rank": int(result["source_rank"][0]),
                    "target_rank": int(result["target_rank"][0]),
                    "variant": "all_points",
                }
            )
        random_mask = torch.rand(len(source_gt), generator=generator, device=device) > 0.3
        if int(random_mask.sum()) < 3:
            random_mask[:3] = True
        weighted = torch.rand(len(source_gt), generator=generator, dtype=torch.float64, device=device) + 0.05
        for name, mask, weights in (
            ("random_valid_mask", random_mask, torch.ones_like(weighted)),
            ("weighted_subset", random_mask, weighted),
        ):
            result = _solve(solver, source_gt, observed, weights, mask)
            predicted = result["transform"][0]
            rows.append(
                {
                    "sample_id": item["sample_id"], "frame_id": int(item["frame_id"]),
                    "symmetry_index": 0, "num_points": int(mask.sum()),
                    "rotation_error_deg": float(rotation_error_deg(predicted, gt_pose)),
                    "translation_error_mm": float(translation_error(predicted, gt_pose) * 1000.0),
                    "determinant": float(result["determinant"][0]),
                    "orthogonality_error": float(result["orthogonality_error"][0]),
                    "source_rank": int(result["source_rank"][0]),
                    "target_rank": int(result["target_rank"][0]), "variant": name,
                }
            )
    # Reflection correction: solver must still return a proper rotation.
    source = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        dtype=torch.float64, device=device,
    )
    reflected = source.clone(); reflected[:, 0] *= -1
    reflection = _solve(
        solver, source, reflected, torch.ones(4, dtype=torch.float64, device=device),
        torch.ones(4, dtype=torch.bool, device=device),
    )
    reflection_corrected = abs(float(reflection["determinant"][0]) - 1.0) < 1e-8
    degenerate_failure = None
    line = torch.stack((torch.linspace(0, 1, 8, device=device), torch.zeros(8, device=device), torch.zeros(8, device=device)), dim=-1).double()
    try:
        _solve(
            solver, line, line,
            torch.ones(8, dtype=torch.float64, device=device),
            torch.ones(8, dtype=torch.bool, device=device),
        )
    except ValueError as exc:
        degenerate_failure = str(exc)
    criterion = all(
        row["rotation_error_deg"] < 1e-4
        and row["translation_error_mm"] < 1e-4
        and abs(row["determinant"] - 1.0) < 1e-6
        and row["orthogonality_error"] < 1e-6
        for row in rows
    ) and reflection_corrected and degenerate_failure is not None
    summary = {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "device": str(device),
        "criterion_passed": criterion,
        "max_rotation_error_deg": max(row["rotation_error_deg"] for row in rows),
        "max_translation_error_mm": max(row["translation_error_mm"] for row in rows),
        "max_orthogonality_error": max(row["orthogonality_error"] for row in rows),
        "minimum_determinant": min(row["determinant"] for row in rows),
        "maximum_determinant": max(row["determinant"] for row in rows),
        "reflection_corrected": reflection_corrected,
        "degenerate_subset_failed_informatively": degenerate_failure is not None,
        "degenerate_failure": degenerate_failure,
        "rows": rows,
    }
    (output / "oracle_procrustes_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "oracle_procrustes_report.md").write_text(
        "# Oracle Weighted Procrustes\n\n"
        f"- passed: `{criterion}`\n"
        f"- max rotation error: `{summary['max_rotation_error_deg']:.9g}` deg\n"
        f"- max translation error: `{summary['max_translation_error_mm']:.9g}` mm\n"
        f"- reflection corrected: `{reflection_corrected}`\n"
        f"- degenerate subset rejected: `{degenerate_failure is not None}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **{k: v for k, v in summary.items() if k != "rows"}}, indent=2))
    return 0 if criterion else 2


if __name__ == "__main__":
    raise SystemExit(main())
