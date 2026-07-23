"""Validation/test evaluation for the production coordinate-guided model."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from symm_template_reg.evaluation.active_coordinate import (
    active_row,
    evaluate_active_sample,
)

from .runtime import move_to_device


SAMPLE_METRIC_FIELDS = (
    "aux_coordinate_p95_mm",
    "exact_global_projected_correspondence_p95_mm",
    "exact_global_projection_alignment_p95_mm",
    "exact_global_projection_rotation_error_deg",
    "exact_global_projection_translation_error_mm",
    "physical_score",
    "k16_exact_global_triangle_recall",
    "k16_fallback_fraction",
)


def _physical_score(row: Mapping[str, Any]) -> float:
    return (
        float(row["exact_global_projected_correspondence_p95_mm"]) / 2.5
        + float(row["exact_global_projection_alignment_p95_mm"]) / 2.5
        + float(row["exact_global_projection_rotation_error_deg"]) / 1.0
        + float(row["exact_global_projection_translation_error_mm"]) / 0.5
    )


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _statistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": math.nan,
            "median": math.nan,
            "p90": math.nan,
            "max": math.nan,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": _quantile(values, 0.90),
        "max": float(np.max(array)),
    }


def _pose_success(row: Mapping[str, Any]) -> bool:
    return (
        float(row["exact_global_projection_rotation_error_deg"]) <= 1.0
        and float(row["exact_global_projection_translation_error_mm"]) <= 0.5
    )


def _practical_surface_success(row: Mapping[str, Any]) -> bool:
    return (
        float(row["exact_global_projected_correspondence_p95_mm"]) <= 2.5
        and float(row["exact_global_projection_alignment_p95_mm"]) <= 2.5
    )


def _source_dataset_split(row: Mapping[str, Any]) -> str:
    # ``split`` is accepted only for reports produced by older callers.
    return str(row.get("source_dataset_split", row.get("split", "unknown")))


def _evaluation_role(row: Mapping[str, Any]) -> str:
    return str(row.get("evaluation_role", "validation"))


def _flat_group_metrics(
    *,
    group_id: str,
    values: list[dict[str, Any]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    output = {
        "group_id": group_id,
        **dict(identity),
        "num_observations": len(values),
        "num_frames": len(
            {(str(row["scene_id"]), int(row["frame_id"])) for row in values}
        ),
        "num_fragments": len(
            {(str(row["scene_id"]), int(row["fragment_id"])) for row in values}
        ),
    }
    for metric in SAMPLE_METRIC_FIELDS:
        statistics = _statistics([float(row[metric]) for row in values])
        for statistic, value in statistics.items():
            output[f"{statistic}_{metric}"] = value
    pose_count = sum(_pose_success(row) for row in values)
    surface_count = sum(_practical_surface_success(row) for row in values)
    joint_count = sum(
        _pose_success(row) and _practical_surface_success(row) for row in values
    )
    output.update(
        {
            "pose_success_count": pose_count,
            "pose_success_rate": pose_count / max(len(values), 1),
            "practical_surface_success_count": surface_count,
            "practical_surface_success_rate": surface_count
            / max(len(values), 1),
            "joint_success_count": joint_count,
            "joint_success_rate": joint_count / max(len(values), 1),
        }
    )
    return output


def _fragment_frame_matrix(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    frame_ids = sorted({int(row["frame_id"]) for row in rows})
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            _source_dataset_split(row),
            _evaluation_role(row),
            str(row["scene_id"]),
            int(row["fragment_id"]),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (source_split, role, scene_id, fragment_id), values in sorted(
        grouped.items()
    ):
        matrix_row: dict[str, Any] = {
            "source_dataset_split": source_split,
            "evaluation_role": role,
            "scene_id": scene_id,
            "fragment_id": fragment_id,
            "num_observations": len(values),
        }
        by_frame: dict[int, list[float]] = defaultdict(list)
        for row in values:
            by_frame[int(row["frame_id"])].append(float(row["physical_score"]))
        for frame_id in frame_ids:
            scores = by_frame.get(frame_id, [])
            matrix_row[f"frame_{frame_id:06d}_physical_score"] = (
                float(np.mean(scores)) if scores else None
            )
        output.append(matrix_row)
    return [f"frame_{value:06d}_physical_score" for value in frame_ids], output


def aggregate_production_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_fragment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        source_split = _source_dataset_split(row)
        role = _evaluation_role(row)
        physical_key = (
            f"{source_split}/{row['scene_id']}/"
            f"fragment_{int(row['fragment_id']):04d}"
        )
        by_fragment[physical_key].append(row)
        frame_key = (
            f"{source_split}/{row['scene_id']}/frame_{int(row['frame_id']):06d}"
        )
        by_frame[frame_key].append(row)
        by_scene[f"{source_split}/{row['scene_id']}"].append(row)

    def aggregate(
        groups: Mapping[str, list[dict[str, Any]]],
        group_type: str,
    ) -> list[dict[str, Any]]:
        output = []
        for key, values in sorted(groups.items()):
            first = values[0]
            identity: dict[str, Any] = {
                "source_dataset_split": _source_dataset_split(first),
                "evaluation_role": _evaluation_role(first),
                "scene_id": str(first["scene_id"]),
            }
            if group_type == "frame":
                identity["frame_id"] = int(first["frame_id"])
            elif group_type == "fragment":
                identity["fragment_id"] = int(first["fragment_id"])
            output.append(
                _flat_group_metrics(
                    group_id=key,
                    values=values,
                    identity=identity,
                )
            )
        return output

    fragment_rows = aggregate(by_fragment, "fragment")
    frame_rows = aggregate(by_frame, "frame")
    scene_rows = aggregate(by_scene, "scene")
    fragment_scores = [row["mean_physical_score"] for row in fragment_rows]
    metric_statistics = {
        metric: _statistics([float(row[metric]) for row in rows])
        for metric in SAMPLE_METRIC_FIELDS
    }
    pose_success_count = sum(_pose_success(row) for row in rows)
    surface_success_count = sum(_practical_surface_success(row) for row in rows)
    joint_success_count = sum(
        _pose_success(row) and _practical_surface_success(row) for row in rows
    )
    frame_ids = sorted({int(row["frame_id"]) for row in rows})
    fragment_ids = sorted({int(row["fragment_id"]) for row in rows})
    source_splits = sorted({_source_dataset_split(row) for row in rows})
    evaluation_roles = sorted({_evaluation_role(row) for row in rows})
    _, matrix_rows = _fragment_frame_matrix(rows)
    worst_samples = [
        {
            "sample_id": str(row["sample_id"]),
            "scene_id": str(row["scene_id"]),
            "frame_id": int(row["frame_id"]),
            "fragment_id": int(row["fragment_id"]),
            "physical_score": float(row["physical_score"]),
        }
        for row in sorted(
            rows, key=lambda value: float(value["physical_score"]), reverse=True
        )[:3]
    ]
    return {
        "num_samples": len(rows),
        "num_frames": len(
            {(str(row["scene_id"]), int(row["frame_id"])) for row in rows}
        ),
        "num_physical_fragments": len(fragment_rows),
        "num_scenes": len(scene_rows),
        "frame_ids": frame_ids,
        "fragment_ids": fragment_ids,
        "source_dataset_split": (
            source_splits[0] if len(source_splits) == 1 else source_splits
        ),
        "evaluation_role": (
            evaluation_roles[0]
            if len(evaluation_roles) == 1
            else evaluation_roles
        ),
        "sample_metric_statistics": metric_statistics,
        "success_counts": {
            "num_samples": len(rows),
            "pose_success_count": pose_success_count,
            "practical_surface_success_count": surface_success_count,
            "joint_success_count": joint_success_count,
        },
        "validation/p90_physical_score": _quantile(fragment_scores, 0.90),
        "pose_success_rate": pose_success_count / len(rows) if rows else math.nan,
        "practical_surface_success_rate": (
            surface_success_count / len(rows) if rows else math.nan
        ),
        "joint_success_rate": joint_success_count / len(rows) if rows else math.nan,
        "median_physical_score": (
            float(np.median(fragment_scores)) if fragment_scores else math.nan
        ),
        "max_per_sample_score": (
            max(float(row["physical_score"]) for row in rows)
            if rows
            else math.nan
        ),
        "worst_sample": worst_samples[0]["sample_id"] if worst_samples else None,
        "worst_samples": worst_samples,
        "per_frame": frame_rows,
        "per_physical_fragment": fragment_rows,
        "per_scene": scene_rows,
        "fragment_frame_matrix": matrix_rows,
    }


@torch.no_grad()
def evaluate_production(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    *,
    source_dataset_split: str | None = None,
    evaluation_role: str | None = None,
    split: str | None = None,
    max_batches: int | None = None,
    candidate_k: int = 16,
    projection_chunk_size: int = 64,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if source_dataset_split is None:
        source_dataset_split = split
    elif split is not None and split != source_dataset_split:
        raise ValueError("split and source_dataset_split disagree")
    if not source_dataset_split:
        raise ValueError("source_dataset_split is required")
    if evaluation_role is None:
        evaluation_role = (
            "test_evaluation"
            if source_dataset_split == "test"
            else "validation"
        )
    if not evaluation_role:
        raise ValueError("evaluation_role must be non-empty")
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        moved = move_to_device(batch, device)
        prediction = model(moved)
        observed = moved["observed"]["points_C"]
        valid = prediction.observed_valid_mask
        target = moved["gt"]["points_O_corresponding"]
        world = moved["gt"].get("T_W_from_C")
        equivalent = moved["gt"].get("equivalent_T_C_from_O")
        for index, sample_id in enumerate(moved["sample_id"]):
            equivalent_pose = moved["gt"]["T_C_from_O"][index]
            candidate = equivalent[index] if isinstance(equivalent, list) else None
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 3:
                equivalent_pose = candidate[0].to(device)
            result = evaluate_active_sample(
                q_aux_O=prediction.correspondence_points_O[index],
                valid_mask=valid[index],
                target_O=target[index],
                observed_C=observed[index],
                vertices_O=moved["template_mesh_vertices_O"][index],
                faces=moved["template_mesh_faces"][index],
                equivalent_pose=equivalent_pose,
                procrustes=model.weighted_procrustes,
                candidate_k=candidate_k,
                projection_chunk_size=projection_chunk_size,
            )
            T_W_from_C = (
                world[index]
                if isinstance(world, torch.Tensor)
                else torch.eye(4, dtype=observed.dtype, device=device)
            )
            row = active_row(
                result,
                sample_id=str(sample_id),
                frame_id=int(moved["frame_id"][index]),
                T_W_from_C=T_W_from_C,
            )
            row.update(
                {
                    "source_dataset_split": source_dataset_split,
                    "evaluation_role": evaluation_role,
                    "scene_id": str(moved["scene_id"][index]),
                    "fragment_id": int(moved["fragment_id"][index]),
                }
            )
            row["physical_score"] = _physical_score(row)
            rows.append(row)
    summary = aggregate_production_metrics(rows)
    summary.update(
        {
            "source_dataset_split": source_dataset_split,
            "evaluation_role": evaluation_role,
            "runtime_seconds": time.perf_counter() - started,
            "max_batches": max_batches,
            "test_results_must_not_be_used_for_model_selection": (
                source_dataset_split == "test"
                or evaluation_role == "test_evaluation"
            ),
        }
    )
    if was_training:
        model.train()
    return summary, rows


def write_evaluation_report(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if rows:
        scalar_fields = [
            key
            for key, value in rows[0].items()
            if isinstance(value, (str, int, float, bool)) or value is None
        ]
        with (output / "per_sample_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_fields)
            writer.writeheader()
            writer.writerows(
                [{key: row.get(key) for key in scalar_fields} for row in rows]
            )
    for name in ("per_frame", "per_physical_fragment", "per_scene"):
        values = list(summary.get(name, []))
        if values:
            with (output / f"{name}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    matrix = list(summary.get("fragment_frame_matrix", []))
    if matrix:
        fixed = [
            "source_dataset_split",
            "evaluation_role",
            "scene_id",
            "fragment_id",
            "num_observations",
        ]
        frame_fields = sorted(
            {
                key
                for row in matrix
                for key in row
                if key.startswith("frame_") and key.endswith("_physical_score")
            }
        )
        with (output / "fragment_frame_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=[*fixed, *frame_fields])
            writer.writeheader()
            writer.writerows(matrix)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validation_history_record(
    summary: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    statistics = dict(summary.get("sample_metric_statistics", {}))
    success = dict(summary.get("success_counts", {}))
    record: dict[str, Any] = {
        "record_type": "validation",
        **{
            key: int(training_state.get(key, 0))
            for key in (
                "epoch",
                "batch_in_epoch",
                "batch_step",
                "optimizer_step",
                "samples_seen",
            )
        },
        "source_dataset_split": summary.get("source_dataset_split"),
        "evaluation_role": summary.get("evaluation_role"),
        "num_samples": int(summary.get("num_samples", 0)),
        "num_frames": int(summary.get("num_frames", 0)),
        "num_physical_fragments": int(
            summary.get("num_physical_fragments", 0)
        ),
        "num_scenes": int(summary.get("num_scenes", 0)),
        "frame_ids": list(summary.get("frame_ids", [])),
        "fragment_ids": list(summary.get("fragment_ids", [])),
        "validation/p90_physical_score": float(
            summary.get("validation/p90_physical_score", math.nan)
        ),
        "max_per_sample_score": float(
            summary.get("max_per_sample_score", math.nan)
        ),
        "pose_success_rate": float(summary.get("pose_success_rate", math.nan)),
        "practical_surface_success_rate": float(
            summary.get("practical_surface_success_rate", math.nan)
        ),
        "joint_success_rate": float(
            summary.get("joint_success_rate", math.nan)
        ),
        "success_counts": success,
        "sample_metric_statistics": statistics,
        "worst_samples": list(summary.get("worst_samples", [])),
    }
    return record


def _flat_validation_history_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        key: record.get(key)
        for key in (
            "record_type",
            "epoch",
            "batch_in_epoch",
            "batch_step",
            "optimizer_step",
            "samples_seen",
            "source_dataset_split",
            "evaluation_role",
            "num_samples",
            "num_frames",
            "num_physical_fragments",
            "num_scenes",
            "validation/p90_physical_score",
            "max_per_sample_score",
            "pose_success_rate",
            "practical_surface_success_rate",
            "joint_success_rate",
        )
    }
    output["frame_ids"] = json.dumps(record.get("frame_ids", []))
    output["fragment_ids"] = json.dumps(record.get("fragment_ids", []))
    output["worst_sample_ids"] = json.dumps(
        [
            value.get("sample_id")
            for value in record.get("worst_samples", [])
        ]
    )
    for name, values in record.get("sample_metric_statistics", {}).items():
        for statistic in ("mean", "median", "p90", "max"):
            output[f"{statistic}_{name}"] = values.get(statistic)
    for name, value in record.get("success_counts", {}).items():
        output[name] = value
    return output


def write_validation_tracking(
    run_dir: str | Path,
    summary: Mapping[str, Any],
    training_state: Mapping[str, Any],
    *,
    combined_history_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    record = validation_history_record(summary, training_state)
    _atomic_json(
        root / "latest_validation_metrics.json",
        {
            "training_state": {
                key: int(training_state.get(key, 0))
                for key in (
                    "epoch",
                    "batch_in_epoch",
                    "batch_step",
                    "optimizer_step",
                    "samples_seen",
                )
            },
            **dict(summary),
        },
    )
    jsonl_path = root / "validation_history.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    if combined_history_path is not None:
        with Path(combined_history_path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    flat = _flat_validation_history_record(record)
    csv_path = root / "validation_history.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)
    return record


def _compact_ids(values: list[Any], limit: int = 24) -> str:
    if len(values) <= limit:
        return "[" + ", ".join(str(value) for value in values) + "]"
    head = ", ".join(str(value) for value in values[:limit])
    return f"[{head}, ... +{len(values) - limit}]"


def _format_group_table(
    title: str,
    rows: list[Mapping[str, Any]],
    *,
    id_field: str,
    max_rows: int,
) -> list[str]:
    lines = [
        title,
        "  scene         id      n       mean        p90        max   pose%  surf%",
    ]
    ordered = sorted(
        rows,
        key=lambda row: float(row["max_physical_score"]),
        reverse=True,
    )
    for row in ordered[:max_rows]:
        lines.append(
            "  "
            f"{str(row['scene_id']):<13.13} "
            f"{int(row[id_field]):>6d} "
            f"{int(row['num_observations']):>6d} "
            f"{float(row['mean_physical_score']):>10.3f} "
            f"{float(row['p90_physical_score']):>10.3f} "
            f"{float(row['max_physical_score']):>10.3f} "
            f"{100.0 * float(row['pose_success_rate']):>6.1f} "
            f"{100.0 * float(row['practical_surface_success_rate']):>6.1f}"
        )
    if len(ordered) > max_rows:
        lines.append(
            f"  ... {len(ordered) - max_rows} rows omitted; see the full CSV"
        )
    return lines


def format_validation_report(
    summary: Mapping[str, Any],
    training_state: Mapping[str, Any],
    *,
    max_group_rows: int = 12,
) -> str:
    statistics = summary["sample_metric_statistics"]
    success = summary["success_counts"]
    lines = [
        (
            f"[VALIDATION step {int(training_state.get('optimizer_step', 0)):06d}] "
            f"role={summary['evaluation_role']} "
            f"source={summary['source_dataset_split']} "
            f"samples={summary['num_samples']} frames={summary['num_frames']} "
            f"fragments={summary['num_physical_fragments']} "
            f"scenes={summary['num_scenes']}"
        ),
        f"  frame_ids={_compact_ids(list(summary['frame_ids']))}",
        f"  fragment_ids={_compact_ids(list(summary['fragment_ids']))}",
        "  metric                                      mean     median        p90        max",
    ]
    labels = (
        ("aux_coordinate_p95_mm", "aux coordinate p95 (mm)"),
        (
            "exact_global_projected_correspondence_p95_mm",
            "exact-global correspondence p95 (mm)",
        ),
        (
            "exact_global_projection_alignment_p95_mm",
            "alignment p95 (mm)",
        ),
        (
            "exact_global_projection_rotation_error_deg",
            "rotation error (deg)",
        ),
        (
            "exact_global_projection_translation_error_mm",
            "translation error (mm)",
        ),
        ("physical_score", "physical score"),
        ("k16_exact_global_triangle_recall", "K16 recall"),
        ("k16_fallback_fraction", "K16 fallback"),
    )
    for key, label in labels:
        values = statistics[key]
        lines.append(
            f"  {label:<42} "
            f"{float(values['mean']):>9.3f} "
            f"{float(values['median']):>10.3f} "
            f"{float(values['p90']):>10.3f} "
            f"{float(values['max']):>10.3f}"
        )
    lines.extend(
        [
            (
                "  success: "
                f"pose={success['pose_success_count']}/{success['num_samples']} "
                f"({100.0 * float(summary['pose_success_rate']):.1f}%) "
                "surface="
                f"{success['practical_surface_success_count']}/"
                f"{success['num_samples']} "
                f"({100.0 * float(summary['practical_surface_success_rate']):.1f}%) "
                f"joint={success['joint_success_count']}/{success['num_samples']} "
                f"({100.0 * float(summary['joint_success_rate']):.1f}%)"
            ),
            (
                "  selection: "
                "p90_per_fragment="
                f"{float(summary['validation/p90_physical_score']):.3f} "
                f"max_per_sample={float(summary['max_per_sample_score']):.3f}"
            ),
            "  worst samples:",
        ]
    )
    for index, row in enumerate(summary["worst_samples"], start=1):
        lines.append(
            f"    {index}. score={float(row['physical_score']):.3f} "
            f"frame={int(row['frame_id'])} fragment={int(row['fragment_id'])} "
            f"{row['sample_id']}"
        )
    lines.extend(
        _format_group_table(
            "  BY FRAME (worst first)",
            list(summary["per_frame"]),
            id_field="frame_id",
            max_rows=max_group_rows,
        )
    )
    lines.extend(
        _format_group_table(
            "  BY FRAGMENT (worst first)",
            list(summary["per_physical_fragment"]),
            id_field="fragment_id",
            max_rows=max_group_rows,
        )
    )
    return "\n".join(lines)


__all__ = [
    "SAMPLE_METRIC_FIELDS",
    "aggregate_production_metrics",
    "evaluate_production",
    "format_validation_report",
    "validation_history_record",
    "write_validation_tracking",
    "write_evaluation_report",
]
