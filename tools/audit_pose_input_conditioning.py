#!/usr/bin/env python3
"""Counterfactual audit of observed/template/centroid influence on pose outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.engine.checkpoint import load_checkpoint  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_dataset  # noqa: E402
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.evaluation.context_conditioning import (  # noqa: E402
    input_permutation_equivariance_error,
)
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.pose.metrics import rotation_error_deg  # noqa: E402
from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg  # noqa: E402


def _checkpoint(run: Path) -> Path:
    summary = json.loads((run / "stage_summary.json").read_text(encoding="utf-8"))
    return Path(summary["best_checkpoint"]).expanduser().resolve()


def _points(payload: dict[str, Any], template: bool = False) -> tuple[str, torch.Tensor]:
    names = ("fine_points_O", "points_O") if template else ("points_C", "points")
    for name in names:
        if isinstance(payload.get(name), torch.Tensor):
            return name, payload[name]
    raise KeyError("could not locate point tensor")


def _reorder_aligned(payload: dict[str, Any], order: torch.Tensor) -> None:
    length = len(order)
    for key, value in list(payload.items()):
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and len(value) == length:
            payload[key] = value[order]


def _intervene(sample: dict[str, Any], kind: str, replacement: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(sample)
    if kind == "observed_permuted":
        result["observed"] = deepcopy(replacement["observed"])
        return result
    if kind == "observed_zeroed":
        key, points = _points(result["observed"])
        result["observed"][key] = torch.zeros_like(points)
        return result
    if kind == "observed_tokens_shuffled":
        _, points = _points(result["observed"])
        order = torch.arange(len(points) - 1, -1, -1, device=points.device)
        _reorder_aligned(result["observed"], order)
        return result
    if kind == "template_zeroed":
        payload = result["template"]
        for name in ("fine_points_O", "coarse_points_O", "points_O"):
            if isinstance(payload.get(name), torch.Tensor):
                payload[name] = torch.zeros_like(payload[name])
        return result
    if kind == "centroid_scale_only":
        key, points = _points(result["observed"])
        centroid = points.mean(dim=0)
        scale = torch.linalg.vector_norm(points - centroid, dim=-1).max()
        synthetic = centroid.expand_as(points).clone()
        if len(points) >= 2:
            synthetic[0, 0] += scale
            synthetic[1, 0] -= scale
        result["observed"][key] = synthetic
        return result
    if kind != "original":
        raise ValueError(kind)
    return result


@torch.no_grad()
def _predict(model: torch.nn.Module, collate: Any, sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    prediction = model(move_to_device(collate([sample]), device))
    primary = (
        prediction.base_pose
        if prediction.base_pose is not None
        else prediction.pose_hypotheses[:, 0]
    )
    return {
        "primary": primary.cpu(),
        "queries": prediction.pose_hypotheses.cpu(),
        "context": (
            prediction.context_diagnostics["sample_context"].cpu()
            if prediction.context_diagnostics is not None
            else torch.empty((1, 0))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[4, 8, 6, 5])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    register_all_modules()
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    config["data"]["train_manifest"] = str(manifest_path)
    config["data"]["validation_manifest"] = "same_as_train"
    config["dataset"]["fragment_mesh_cache_dir"] = str(output / "cache")
    dataset = _build_dataset(config)
    manifest, _ = load_faces840_manifest(manifest_path, config, dataset)
    wanted = set(args.frames)
    selected_ids = {
        str(item["sample_id"]) for item in manifest["samples"] if int(item["frame_id"]) in wanted
    }
    indices = [
        index
        for index, record in enumerate(dataset.sample_records)
        if record.sample_id in selected_ids
    ]
    samples = sorted(
        [dataset[index] for index in indices], key=lambda item: int(item["frame_id"])
    )
    if len(samples) < 2:
        raise ValueError("conditioning audit requires at least two selected views")
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    model = build_model(config["model"]).to(device).eval()
    load_checkpoint(_checkpoint(run), model=model, map_location=device, strict=True)
    originals = [_predict(model, collate, sample, device) for sample in samples]
    interventions = (
        "original",
        "observed_permuted",
        "observed_zeroed",
        "observed_tokens_shuffled",
        "template_zeroed",
        "centroid_scale_only",
    )
    rows = []
    permuted_primary = []
    permutation = [(index + 1) % len(samples) for index in range(len(samples))]
    for index, sample in enumerate(samples):
        original = originals[index]
        for kind in interventions:
            replacement_index = permutation[index] if kind == "observed_permuted" else index
            changed_sample = _intervene(sample, kind, samples[replacement_index])
            changed = (
                original if kind == "original" else _predict(model, collate, changed_sample, device)
            )
            if kind == "observed_permuted":
                permuted_primary.append(changed["primary"][0])
            primary_rotation = float(
                rotation_error_deg(changed["primary"], original["primary"]).mean()
            )
            primary_translation = float(
                torch.linalg.vector_norm(
                    changed["primary"][..., :3, 3]
                    - original["primary"][..., :3, 3],
                    dim=-1,
                ).mean()
                * 1000.0
            )
            query_rotation = float(
                rotation_error_deg(changed["queries"], original["queries"]).mean()
            )
            query_translation = float(
                torch.linalg.vector_norm(
                    changed["queries"][..., :3, 3]
                    - original["queries"][..., :3, 3],
                    dim=-1,
                ).mean()
                * 1000.0
            )
            context_distance = (
                float(torch.linalg.vector_norm(changed["context"] - original["context"]))
                if changed["context"].numel()
                else 0.0
            )
            rows.append(
                {
                    "frame_id": int(sample["frame_id"]),
                    "intervention": kind,
                    "replacement_frame_id": int(samples[replacement_index]["frame_id"]),
                    "primary_rotation_change_deg": primary_rotation,
                    "primary_translation_change_mm": primary_translation,
                    "mean_query_rotation_change_deg": query_rotation,
                    "mean_query_translation_change_mm": query_translation,
                    "sample_context_change_l2": context_distance,
                }
            )
    with (output / "conditioning_interventions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_kind = {
        kind: [row for row in rows if row["intervention"] == kind]
        for kind in interventions
    }
    permutation_metrics = input_permutation_equivariance_error(
        torch.stack([value["primary"][0] for value in originals]),
        torch.stack(permuted_primary),
        permutation,
    )
    mean = lambda kind, key: sum(float(row[key]) for row in by_kind[kind]) / len(by_kind[kind])
    permuted_query_rotation = mean("observed_permuted", "mean_query_rotation_change_deg")
    centroid_rotation = mean("centroid_scale_only", "primary_rotation_change_deg")
    if permuted_query_rotation < 0.01:
        diagnosis = "static_query_codebook"
    elif centroid_rotation < 0.1:
        diagnosis = "centroid_only_shortcut"
    else:
        diagnosis = "pose_responds_to_local_and_template_geometry"
    summary = {
        "run": str(run),
        "manifest": str(manifest_path),
        "frames": [int(sample["frame_id"]) for sample in samples],
        "diagnosis": diagnosis,
        "contributions": {
            "local_observed_geometry_rotation_change_deg": mean(
                "observed_zeroed", "primary_rotation_change_deg"
            ),
            "template_geometry_rotation_change_deg": mean(
                "template_zeroed", "primary_rotation_change_deg"
            ),
            "centroid_scale_only_rotation_change_deg": centroid_rotation,
            "mode_embedding_query_response_deg": permuted_query_rotation,
            "point_order_sensitivity_deg": mean(
                "observed_tokens_shuffled", "primary_rotation_change_deg"
            ),
        },
        **permutation_metrics,
    }
    (output / "conditioning_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Pose input-conditioning audit",
        "",
        f"- diagnosis: `{diagnosis}`",
        f"- views: `{summary['frames']}`",
        f"- permuted query rotation response: `{permuted_query_rotation:.8g} deg`",
        f"- permutation equivariance error: `{summary['input_permutation_equivariance_error']:.8g}`",
        "",
        "See `conditioning_interventions.csv` for every frame/intervention.",
    ]
    (output / "conditioning_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
