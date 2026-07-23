#!/usr/bin/env python3
"""Audit one row per physical fragment mesh without selecting a threshold."""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.datasets.fragment_mesh_filter import (  # noqa: E402
    REQUIRED_TEST_SPLIT_FLAGS,
    scan_fragment_mesh_metadata,
)


PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _observation_counts(dataset_root: Path) -> dict[tuple[str, int], list[int]]:
    result: dict[tuple[str, int], list[int]] = {}
    for scene_dir in sorted(dataset_root.glob("scene_*")):
        gt_path = scene_dir / "gt_annotations.json"
        if not gt_path.is_file():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        scene_id = str(gt.get("scene_id", scene_dir.name))
        for frame in gt.get("frames", []):
            frame_id = int(frame["frame_id"])
            npz_path = scene_dir / str(
                frame.get("visible_points", f"visible_points/frame_{frame_id:06d}.npz")
            )
            with np.load(npz_path, allow_pickle=False) as arrays:
                fragment_ids, counts = np.unique(
                    arrays["fragment_id"], return_counts=True
                )
            for fragment_id, count in zip(fragment_ids.tolist(), counts.tolist()):
                result.setdefault((scene_id, int(fragment_id)), []).append(int(count))
    return result


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    result = {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }
    for percentile in PERCENTILES:
        result[f"p{percentile:02d}"] = float(np.percentile(array, percentile))
    return result


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_histogram_png(path: Path, values: list[float], color: tuple[int, int, int]) -> None:
    width, height = 800, 500
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    counts, _ = np.histogram(np.asarray(values, dtype=np.float64), bins=min(32, max(4, len(values))))
    left, right, top, bottom = 55, width - 20, 20, height - 45
    image[bottom : bottom + 2, left:right] = 0
    image[top:bottom, left : left + 2] = 0
    maximum = max(int(counts.max()), 1)
    span = right - left - 4
    for index, count in enumerate(counts):
        x0 = left + 3 + round(index * span / len(counts))
        x1 = left + 3 + round((index + 1) * span / len(counts)) - 1
        bar_height = round((bottom - top - 5) * int(count) / maximum)
        image[bottom - bar_height : bottom, x0:max(x0 + 1, x1)] = color
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def run_audit(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    candidate_face_thresholds: list[int] | None = None,
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata, cache_report = scan_fragment_mesh_metadata(
        root,
        filter_config={
            "enabled": False,
            "cache_metadata": True,
            "missing_mesh_policy": "error",
            "manifest_mismatch_policy": "error",
        },
        cache_dir=REPO_ROOT / "work_dirs" / "cache",
    )
    observations = _observation_counts(root)
    rows: list[dict[str, Any]] = []
    for key, item in sorted(metadata.items()):
        counts = observations.get(key, [])
        rows.append(
            {
                "scene_id": item.scene_id,
                "fragment_id": item.fragment_id,
                "fragment_key": item.fragment_key,
                "mesh_path": str(item.mesh_path),
                "mesh_sha256": item.sha256,
                "num_vertices": item.num_vertices,
                "num_faces": item.num_faces,
                "polygon_size_distribution": json.dumps(
                    item.polygon_size_distribution, sort_keys=True
                ),
                "surface_area_m2": item.surface_area_m2,
                "bbox_min": json.dumps(item.bbox_min),
                "bbox_max": json.dumps(item.bbox_max),
                "bbox_diagonal_m": item.bbox_diagonal_m,
                "observations_count": len(counts),
                "min_observed_points_across_views": min(counts) if counts else 0,
                "median_observed_points_across_views": float(np.median(counts)) if counts else 0,
                "max_observed_points_across_views": max(counts) if counts else 0,
            }
        )
    if not rows:
        raise ValueError(f"no physical fragment meshes found below {root}")
    fieldnames = list(rows[0])
    with (output / "fragment_mesh_sizes.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output / "fragment_mesh_sizes.json",
        {**REQUIRED_TEST_SPLIT_FLAGS, "dataset_root": str(root), "fragments": rows},
    )

    face_values = [int(row["num_faces"]) for row in rows]
    area_values = [float(row["surface_area_m2"]) for row in rows]
    diagonal_values = [float(row["bbox_diagonal_m"]) for row in rows]
    face_stats = _statistics([float(value) for value in face_values])
    automatic = {
        int(round(face_stats["p05"])),
        int(round(face_stats["p10"])),
        int(round(face_stats["p25"])),
    }
    automatic.update(int(value) for value in (candidate_face_thresholds or []))
    candidate_rows = []
    for threshold in sorted(value for value in automatic if value >= 0):
        accepted_keys = {
            (str(row["scene_id"]), int(row["fragment_id"]))
            for row in rows
            if int(row["num_faces"]) >= threshold
        }
        accepted_observations = sum(
            len(counts) for key, counts in observations.items() if key in accepted_keys
        )
        total_observations = sum(len(counts) for counts in observations.values())
        candidate_rows.append(
            {
                "candidate_min_faces": threshold,
                "accepted_physical_fragments": len(accepted_keys),
                "rejected_physical_fragments": len(rows) - len(accepted_keys),
                "accepted_observations": accepted_observations,
                "rejected_observations": total_observations - accepted_observations,
            }
        )
    summary = {
        **REQUIRED_TEST_SPLIT_FLAGS,
        "dataset_root": str(root),
        "physical_fragments": len(rows),
        "frame_observations": sum(len(counts) for counts in observations.values()),
        "statistics": {
            "num_faces": face_stats,
            "surface_area_m2": _statistics(area_values),
            "bbox_diagonal_m": _statistics(diagonal_values),
        },
        "candidate_thresholds": candidate_rows,
        "selected_min_num_faces": None,
        "threshold_selection_note": "Threshold selection is a user/modeling decision.",
        "cache": cache_report,
    }
    _write_json(output / "fragment_size_summary.json", summary)
    _write_histogram_png(
        output / "fragment_face_count_histogram.png", face_values, (50, 110, 220)
    )
    _write_histogram_png(
        output / "fragment_surface_area_histogram.png", area_values, (40, 170, 100)
    )
    lines = [
        "# Physical fragment size audit",
        "",
        f"Dataset: `{root}`",
        "",
        "- `debug_training_on_test_split = true`",
        "- `results_are_not_final_evaluation = true`",
        f"- Physical fragments: {len(rows)}",
        f"- Frame observations: {summary['frame_observations']}",
        "",
        "## Distributions",
        "",
        "| metric | min | p05 | p10 | p25 | p50 | p75 | p90 | p95 | max | mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summary["statistics"].items():
        lines.append(
            f"| {name} | {stats['min']:.9g} | {stats['p05']:.9g} | {stats['p10']:.9g} | "
            f"{stats['p25']:.9g} | {stats['p50']:.9g} | {stats['p75']:.9g} | "
            f"{stats['p90']:.9g} | {stats['p95']:.9g} | {stats['max']:.9g} | {stats['mean']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Candidate face thresholds",
            "",
            "| min faces | accepted fragments | rejected fragments | accepted observations | rejected observations |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate in candidate_rows:
        lines.append(
            "| {candidate_min_faces} | {accepted_physical_fragments} | "
            "{rejected_physical_fragments} | {accepted_observations} | "
            "{rejected_observations} |".format(**candidate)
        )
    lines.extend(
        [
            "",
            "**Threshold selection is a user/modeling decision.**",
            "",
            "No threshold was selected by this audit.",
            "",
        ]
    )
    (output / "fragment_size_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-face-thresholds", nargs="*", type=int)
    args = parser.parse_args()
    summary = run_audit(
        args.dataset_root,
        args.output_dir,
        candidate_face_thresholds=args.candidate_face_thresholds,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
