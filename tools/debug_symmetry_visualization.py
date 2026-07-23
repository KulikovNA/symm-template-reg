#!/usr/bin/env python3
"""Audit annotated fragment symmetry targets and export per-hypothesis PLY galleries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.visualization.symmetry_debug import (
    DEFAULT_DEBUG_OUTPUT_ROOT,
    run_annotated_fragment_symmetry_debug,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--object-model-id", required=True)
    parser.add_argument("--template-mesh", default="auto", help="auto or explicit PLY path")
    parser.add_argument(
        "--symmetry-sidecar", default="auto", help="auto or explicit JSON path"
    )
    parser.add_argument("--mode", choices=("template", "fragments", "all"), default="all")
    parser.add_argument(
        "--scene-ids",
        nargs="+",
        default=["scene_000000", "scene_000001", "scene_000002"],
    )
    parser.add_argument("--fragment-ids", nargs="+", type=int)
    parser.add_argument("--max-fragments-per-scene", type=int)
    parser.add_argument(
        "--all-annotated-fragments",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--so2-visualization-samples", type=int, default=12)
    parser.add_argument("--gallery-columns", type=int, default=4)
    parser.add_argument("--gallery-spacing-scale", type=float, default=2.5)
    parser.add_argument(
        "--template-projection-distance-m",
        type=float,
        default=5e-4,
        help="maximum shell-to-template distance used to color template faces",
    )
    parser.add_argument(
        "--template-boundary-resolution-m",
        type=float,
        default=1e-4,
        help="target edge length near the boundary, capped by boundary max depth",
    )
    parser.add_argument("--template-boundary-max-depth", type=int, default=2)
    parser.add_argument("--min-region-surface-area-m2", type=float, default=0.0)
    parser.add_argument("--min-region-surface-area-fraction", type=float, default=0.01)
    parser.add_argument("--area-sample-count", type=int, default=2048)
    parser.add_argument("--min-area-sample-count", type=int, default=16)
    parser.add_argument("--output-root", default=str(DEFAULT_DEBUG_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run_annotated_fragment_symmetry_debug(
        dataset_root=args.dataset_root,
        object_model_id=args.object_model_id,
        scene_ids=args.scene_ids,
        fragment_ids=args.fragment_ids,
        max_fragments_per_scene=args.max_fragments_per_scene,
        all_annotated_fragments=args.all_annotated_fragments,
        template_mesh=args.template_mesh,
        symmetry_sidecar=args.symmetry_sidecar,
        mode=args.mode,
        so2_visualization_samples=args.so2_visualization_samples,
        gallery_columns=args.gallery_columns,
        gallery_spacing_scale=args.gallery_spacing_scale,
        template_projection_distance_m=args.template_projection_distance_m,
        template_boundary_resolution_m=args.template_boundary_resolution_m,
        template_boundary_max_depth=args.template_boundary_max_depth,
        min_surface_area_m2=args.min_region_surface_area_m2,
        min_surface_area_fraction=args.min_region_surface_area_fraction,
        area_sample_count=args.area_sample_count,
        min_area_sample_count=args.min_area_sample_count,
        output_root=args.output_root,
    )
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
