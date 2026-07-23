#!/usr/bin/env python3
"""Раскрасить регионы template и построить gallery допустимых симметрий."""

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

from symm_template_reg.datasets.template_repository import load_ply  # noqa: E402
from symm_template_reg.models.symmetry.groups import group_to_dict  # noqa: E402
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms  # noqa: E402
from symm_template_reg.models.symmetry.metadata import load_symmetry_metadata  # noqa: E402
from symm_template_reg.models.symmetry.region_assignment import (  # noqa: E402
    active_symmetry_regions,
    effective_group_from_regions,
    validate_region_partition,
)
from symm_template_reg.visualization.ply import write_colored_ply  # noqa: E402


PALETTE = np.asarray(
    [[0, 210, 210], [230, 50, 210], [255, 150, 20], [110, 210, 70],
     [160, 90, 250], [250, 210, 40], [60, 145, 250], [240, 80, 80]],
    dtype=np.uint8,
)


def _face_expanded(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    expanded = vertices[faces].reshape(-1, 3)
    expanded_faces = np.arange(len(expanded), dtype=np.int64).reshape(-1, 3)
    expanded_colors = np.repeat(colors, 3, axis=0)
    return expanded, expanded_faces, expanded_colors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--canonical-point", type=float, nargs=3, action="append", default=[]
    )
    parser.add_argument("--so2-samples", type=int, default=12)
    parser.add_argument("--gallery-spacing", type=float, default=1.5)
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    mesh = load_ply(args.template)
    vertices = np.asarray(mesh["points"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int64)
    metadata = load_symmetry_metadata(args.sidecar)
    partition = validate_region_partition(
        torch.from_numpy(vertices), torch.from_numpy(faces), metadata
    )
    face_regions = partition.face_region_indices.cpu().numpy()
    face_colors = PALETTE[face_regions % len(PALETTE)]
    expanded, expanded_faces, expanded_colors = _face_expanded(
        vertices, faces, face_colors
    )
    write_colored_ply(
        output / "template_symmetry_regions.ply",
        expanded,
        expanded_colors,
        faces=expanded_faces,
        comments=["faces are split at region boundaries"],
    )
    points = torch.as_tensor(
        args.canonical_point if args.canonical_point else vertices,
        dtype=torch.float32,
    )
    active = active_symmetry_regions(points, metadata)
    effective = effective_group_from_regions(metadata, active)
    transforms = symmetry_transforms(
        effective,
        metadata.axis.direction,
        metadata.axis.origin,
        so2_num_samples=(
            args.so2_samples if effective.type == "SO2" else None
        ),
    ).cpu().numpy()
    span = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    gallery_vertices, gallery_faces, gallery_colors = [], [], []
    offset = 0
    for index, transform in enumerate(transforms):
        transformed = (
            vertices @ transform[:3, :3].T + transform[:3, 3]
        )
        transformed[:, 0] += index * span * float(args.gallery_spacing)
        gallery_vertices.append(transformed)
        gallery_faces.append(faces + offset)
        gallery_colors.append(np.full((len(vertices), 3), 185, dtype=np.uint8))
        offset += len(vertices)
    write_colored_ply(
        output / "symmetry_hypotheses_gallery.ply",
        np.concatenate(gallery_vertices),
        np.concatenate(gallery_colors),
        faces=np.concatenate(gallery_faces),
    )
    legend = {
        "template": str(Path(args.template).expanduser().resolve()),
        "sidecar": str(Path(args.sidecar).expanduser().resolve()),
        "regions": [
            {
                "region_id": region.region_id,
                "color_rgb": PALETTE[index % len(PALETTE)].tolist(),
                "rotation_group": group_to_dict(region.rotation_group),
            }
            for index, region in enumerate(metadata.regions)
        ],
        "canonical_points": points.tolist(),
        "active_regions": {
            region.region_id: bool(active[index])
            for index, region in enumerate(metadata.regions)
        },
        "effective_symmetry_group": group_to_dict(effective),
        "gallery_hypotheses": len(transforms),
    }
    (output / "legend.json").write_text(
        json.dumps(legend, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(legend, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
