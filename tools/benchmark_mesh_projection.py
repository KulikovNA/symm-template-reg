#!/usr/bin/env python3
"""Benchmark deterministic exact global point-to-triangle projection."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]; TOOLS = Path(__file__).resolve().parent
for value in (ROOT, TOOLS):
    if str(value) not in sys.path: sys.path.insert(0, str(value))
from coordinate_guided_audit_common import load_f1_audit_context  # noqa: E402
from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh  # noqa: E402
from symm_template_reg.models import register_all_modules  # noqa: E402


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _measure(points, vertices, faces, chunk, device, reference):
    points = points.to(device); vertices = vertices.to(device); faces = faces.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _sync(device); started = time.perf_counter()
    result = closest_points_on_triangle_mesh(
        points, vertices, faces, point_chunk_size=chunk
    )
    _sync(device); wall = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    difference = torch.linalg.vector_norm(
        result["points"].detach().cpu() - reference, dim=-1
    )
    return {
        "device": str(device), "chunk_size": int(chunk),
        "point_count": len(points), "triangle_count": len(faces),
        "triangles_tested": len(points) * len(faces),
        "wall_time_sec": wall, "runtime_ms": wall * 1000,
        "points_per_second": len(points) / max(wall, 1e-12),
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else 0.0
        ),
        "cpu_peak_rss_delta_mb": max(0.0, (rss_after - rss_before) / 1024),
        "max_difference_vs_unchunked_reference_mm": float(difference.max() * 1000),
        "mean_difference_vs_unchunked_reference_mm": float(difference.mean() * 1000),
    }


def run(args):
    output = Path(args.output_dir).expanduser().resolve(); rows = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available(): devices.append(torch.device("cuda"))
    elif args.device == "cuda": raise RuntimeError("CUDA requested but unavailable")
    for manifest_index, manifest in enumerate(args.manifest):
        context = load_f1_audit_context(
            args.checkpoint, manifest, output / f"context_{manifest_index}",
            torch.device(args.device),
        )
        points = context["q_aux"][context["mask"]].detach().cpu()
        vertices = context["vertices"].detach().cpu(); faces = context["faces"].detach().cpu()
        # Literal unchunked correctness reference: one chunk containing all points.
        reference_started = time.perf_counter()
        reference = closest_points_on_triangle_mesh(
            points, vertices, faces, point_chunk_size=len(points)
        )["points"].cpu()
        reference_time = time.perf_counter() - reference_started
        frame = int(context["sample"].get("frame_id", manifest_index))
        for device in devices:
            for chunk in args.chunk_sizes:
                row = _measure(points, vertices, faces, chunk, device, reference)
                row.update({
                    "frame_id": frame, "sample_id": context["sample"].get("sample_id"),
                    "manifest": str(Path(manifest).resolve()),
                    "unchunked_cpu_reference_time_sec": reference_time,
                    "projection_mode": "exact_global",
                })
                rows.append(row)
    with (output / "projection_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    best = {}
    for row in rows:
        key = f"frame{row['frame_id']:02d}_{row['device']}"
        if key not in best or row["wall_time_sec"] < best[key]["wall_time_sec"]: best[key] = row
    summary = {
        "benchmark_completed": True, "projection_mode": "exact_global",
        "performance_threshold_invented": False,
        "chunk_sizes": args.chunk_sizes, "best_measured_tradeoffs": best,
        "rows": rows,
    }
    (output / "projection_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# Exact global projection benchmark", "", "No hard runtime threshold was applied.", "", "| frame | device | chunk | time ms | points/s | peak MB | max diff mm |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        peak = row["gpu_peak_memory_mb"] if row["device"] == "cuda" else row["cpu_peak_rss_delta_mb"]
        report.append(f"| {row['frame_id']} | {row['device']} | {row['chunk_size']} | {row['runtime_ms']:.3f} | {row['points_per_second']:.1f} | {peak:.1f} | {row['max_difference_vs_unchunked_reference_mm']:.9f} |")
    (output / "projection_benchmark_report.md").write_text("\n".join(report) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[64,128,256,512,1024])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir).expanduser().resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); register_all_modules()
    result = run(args); print(json.dumps({"output_dir": str(output), **result}, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
