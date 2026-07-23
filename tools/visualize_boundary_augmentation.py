#!/usr/bin/env python3
"""Сохранить mask/PLY preview train-only boundary augmentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets import SplitDirectoryFragmentDataset  # noqa: E402
from symm_template_reg.datasets.boundary_augmentation import (  # noqa: E402
    binary_dilation,
)
from symm_template_reg.visualization.ply import write_colored_ply  # noqa: E402


COLORS = {
    "retained": np.asarray([40, 200, 80], dtype=np.uint8),
    "removed": np.asarray([230, 50, 50], dtype=np.uint8),
    "fracture": np.asarray([220, 40, 220], dtype=np.uint8),
    "depth_ring": np.asarray([255, 145, 30], dtype=np.uint8),
    "rejected": np.asarray([135, 135, 135], dtype=np.uint8),
}


def _mask_png(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def _colored_points(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    available = [(points, color) for points, color in parts if len(points)]
    if not available:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    return (
        np.concatenate([points for points, _ in available], axis=0),
        np.concatenate(
            [np.broadcast_to(color, (len(points), 3)) for points, color in available],
            axis=0,
        ).copy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-id", required=True, type=int)
    parser.add_argument("--fragment-id", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("none", "erode", "dilate", "mixed"), default="mixed"
    )
    parser.add_argument("--radius-min", type=int, default=1)
    parser.add_argument("--radius-max", type=int, default=2)
    parser.add_argument("--max-removed-fraction", type=float, default=0.08)
    parser.add_argument("--max-added-fraction", type=float, default=0.05)
    parser.add_argument("--max-pseudo-target-distance-m", type=float, default=0.002)
    parser.add_argument(
        "--candidate-source",
        choices=("both", "fracture", "depth-ring"),
        default="both",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.split != "train" and args.mode != "none":
        raise ValueError("boundary augmentation preview is disabled for val/test")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    augmentation = {
        "enabled": args.mode != "none",
        "apply_probability": 1.0,
        "mode": args.mode,
        "radius_px": {"min": args.radius_min, "max": args.radius_max},
        "max_removed_fraction": args.max_removed_fraction,
        "max_added_fraction": args.max_added_fraction,
        "max_pseudo_target_distance_m": args.max_pseudo_target_distance_m,
        "include_fracture_candidates": args.candidate_source in {"both", "fracture"},
        "include_depth_ring_candidates": args.candidate_source in {
            "both",
            "depth-ring",
        },
    }
    dataset = SplitDirectoryFragmentDataset(
        args.dataset_root,
        split=args.split,
        selector={
            "scene_ids": [args.scene_id],
            "frame_ids": [args.frame_id],
            "fragment_ids": [args.fragment_id],
            "max_samples": 1,
        },
        random_seed=args.seed,
        boundary_augmentation=augmentation,
        index_cache_dir=output / "dataset_index",
    )
    preview = dataset.augmentation_preview(0)
    debug = preview["debug"]
    if debug is None:
        sample = preview["sample"]
        points = sample["observed"]["points_C"].cpu().numpy()
        debug = {
            "before_mask": np.zeros((1, 1), dtype=bool),
            "after_mask": np.zeros((1, 1), dtype=bool),
            "boundary": np.zeros((1, 1), dtype=bool),
            "retained_points_C": points,
            "removed_points_C": np.empty((0, 3), dtype=np.float32),
            "added_points_C": np.empty((0, 3), dtype=np.float32),
            "added_sources": [],
            "rejected_points_C": np.empty((0, 3), dtype=np.float32),
        }
    _mask_png(output / "before_mask.png", debug["before_mask"])
    _mask_png(output / "after_mask.png", debug["after_mask"])
    rings = np.zeros((*debug["before_mask"].shape, 3), dtype=np.uint8)
    rings[debug["before_mask"]] = COLORS["retained"]
    rings[debug["boundary"]] = np.asarray([255, 220, 40], dtype=np.uint8)
    outer = binary_dilation(debug["before_mask"], args.radius_max) & ~debug["before_mask"]
    rings[outer] = np.asarray([50, 130, 255], dtype=np.uint8)
    Image.fromarray(rings, mode="RGB").save(output / "boundary_rings.png")

    before_points = np.concatenate(
        (debug["retained_points_C"], debug["removed_points_C"]), axis=0
    )
    write_colored_ply(
        output / "before_cloud.ply",
        before_points,
        np.broadcast_to(COLORS["retained"], (len(before_points), 3)),
    )
    added_sources = debug["added_sources"]
    added_points = debug["added_points_C"]
    fracture = added_points[
        np.asarray([source == "fracture" for source in added_sources], dtype=bool)
    ]
    depth_ring = added_points[
        np.asarray([source == "depth_ring" for source in added_sources], dtype=bool)
    ]
    after_points, after_colors = _colored_points(
        [
            (debug["retained_points_C"], COLORS["retained"]),
            (fracture, COLORS["fracture"]),
            (depth_ring, COLORS["depth_ring"]),
        ]
    )
    write_colored_ply(output / "after_cloud.ply", after_points, after_colors)
    overlay_points, overlay_colors = _colored_points(
        [
            (debug["retained_points_C"], COLORS["retained"]),
            (debug["removed_points_C"], COLORS["removed"]),
            (fracture, COLORS["fracture"]),
            (depth_ring, COLORS["depth_ring"]),
            (debug["rejected_points_C"], COLORS["rejected"]),
        ]
    )
    write_colored_ply(
        output / "augmentation_overlay.ply", overlay_points, overlay_colors
    )
    summary = {
        "sample_id": preview["sample"]["sample_id"],
        "split": args.split,
        "scene_id": args.scene_id,
        "frame_id": args.frame_id,
        "fragment_id": args.fragment_id,
        "augmentation": preview["metadata"],
        "colors": {key: value.tolist() for key, value in COLORS.items()},
        "gt_pose_is_model_input": False,
    }
    (output / "augmentation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
