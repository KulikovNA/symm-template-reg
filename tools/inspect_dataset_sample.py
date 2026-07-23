#!/usr/bin/env python3
"""Print a compact, numeric inspection of one loader sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset


def _tensor_summary(value: torch.Tensor | None) -> Any:
    if value is None:
        return None
    summary: dict[str, Any] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if value.numel() and value.dtype != torch.bool:
        summary["min"] = float(value.min())
        summary["max"] = float(value.max())
    if value.dtype == torch.bool:
        summary["true"] = int(value.sum())
    return summary


def inspect_sample(sample: dict[str, Any]) -> dict[str, Any]:
    points_O = sample["gt"]["points_O_corresponding"]
    transform = sample["gt"]["T_C_from_O"]
    transform_error = None
    if points_O is not None:
        transformed = points_O @ transform[:3, :3].T + transform[:3, 3]
        transform_error = float(
            (transformed - sample["observed"]["points_C"]).abs().max()
        )
    return {
        "sample_id": sample["sample_id"],
        "scene_id": sample["scene_id"],
        "frame_id": sample["frame_id"],
        "fragment_id": sample["fragment_id"],
        "object_model_id": sample["object_model_id"],
        "observed": {
            key: _tensor_summary(value)
            for key, value in sample["observed"].items()
        },
        "template": {
            key: _tensor_summary(sample["template"].get(key))
            for key in ("points_O", "normals_O", "faces", "fine_points_O", "coarse_points_O")
        },
        "gt": {
            "T_C_from_O": transform.tolist(),
            "points_O_corresponding": _tensor_summary(points_O),
            "overlap_labels": _tensor_summary(sample["gt"]["overlap_labels"]),
            "active_symmetry_regions": _tensor_summary(
                sample["gt"]["active_symmetry_regions"]
            ),
            "effective_symmetry_group": sample["gt"]["effective_symmetry_group"],
            "equivalent_T_C_from_O": _tensor_summary(
                sample["gt"]["equivalent_T_C_from_O"]
            ),
            "point_transform_max_abs_error_m": transform_error,
        },
        "meta": sample["meta"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--scene-id")
    parser.add_argument("--frame-id", type=int)
    parser.add_argument("--fragment-id", type=int)
    parser.add_argument("--observed-policy", default="precomputed_dataset_points")
    parser.add_argument("--max-observed-points", type=int, default=4096)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FragmentTemplateRegistrationDataset(
        args.dataset_root,
        observed_policy=args.observed_policy,
        min_observed_points=0,
        max_observed_points=args.max_observed_points,
    )
    index = args.index
    if args.scene_id is not None or args.frame_id is not None or args.fragment_id is not None:
        matches = [
            i
            for i, record in enumerate(dataset.sample_records)
            if (args.scene_id is None or record.scene_id == args.scene_id)
            and (args.frame_id is None or record.frame_id == args.frame_id)
            and (args.fragment_id is None or record.fragment_id == args.fragment_id)
        ]
        if not matches:
            raise ValueError(
                "no dataset sample matches the requested scene/frame/fragment selector"
            )
        index = matches[0]
    result = inspect_sample(dataset[index])
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
