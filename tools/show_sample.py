#!/usr/bin/env python3
"""Export one dataset sample as dependency-free colored ASCII PLY artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets import FragmentTemplateRegistrationDataset


def _write_points(path: Path, points: torch.Tensor, colors: np.ndarray) -> None:
    values = points.detach().cpu().numpy()
    colors = np.broadcast_to(np.asarray(colors, dtype=np.uint8), (len(values), 3))
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(values)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("end_header\n")
        for point, color in zip(values, colors):
            stream.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("work_dirs/show_sample"))
    parser.add_argument("--max-observed-points", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = FragmentTemplateRegistrationDataset(
        args.dataset_root,
        observed_policy="precomputed_dataset_points",
        min_observed_points=0,
        max_observed_points=args.max_observed_points,
    )
    sample = dataset[args.index]
    output = args.out_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated_names = (
        "observed_C.ply",
        "template_transformed_C.ply",
        "observed_template_overlay_C.ply",
        "corresponding_transformed_C.ply",
        "sample.json",
    )
    for name in generated_names:
        stale = output / name
        if stale.exists():
            stale.unlink()
    observed = sample["observed"]["points_C"]
    corresponding = sample["gt"]["points_O_corresponding"]
    transform = sample["gt"]["T_C_from_O"]
    template = sample["template"]["fine_points_O"]
    template_C = template @ transform[:3, :3].T + transform[:3, 3]
    observed_colors = np.broadcast_to(
        np.array([40, 220, 80], dtype=np.uint8), (len(observed), 3)
    ).copy()
    surface_labels = sample["observed"].get("surface_labels")
    if surface_labels is not None:
        labels = surface_labels.detach().cpu().numpy()
        observed_colors[labels == 1] = [255, 145, 20]
        observed_colors[labels == 255] = [150, 150, 150]
    _write_points(output / "observed_C.ply", observed, observed_colors)
    _write_points(output / "template_transformed_C.ply", template_C, np.array([230, 50, 210]))
    _write_points(
        output / "observed_template_overlay_C.ply",
        torch.cat((observed, template_C), dim=0),
        np.concatenate(
            (
                observed_colors,
                np.broadcast_to(np.array([230, 50, 210], dtype=np.uint8), (len(template_C), 3)),
            ),
            axis=0,
        ),
    )
    if corresponding is not None:
        corresponding_C = corresponding @ transform[:3, :3].T + transform[:3, 3]
        _write_points(
            output / "corresponding_transformed_C.ply",
            corresponding_C,
            np.array([50, 150, 255]),
        )
    summary = {
        "sample_id": sample["sample_id"],
        "observed_points": len(observed),
        "template_points": len(template),
        "coord_unit": sample["meta"]["coord_unit"],
        "files": sorted(path.name for path in output.glob("*.ply")),
    }
    (output / "sample.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"out_dir": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
