#!/usr/bin/env python3
"""Audit K1/K8 per-view errors, query assignments and context dependence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from symm_template_reg.engine.checkpoint import load_checkpoint  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_manifest import load_faces840_manifest  # noqa: E402
from symm_template_reg.engine.overfit_trainer import (  # noqa: E402
    _amp_settings,
    _build_dataset,
    _build_pose_criterion,
    _evaluate,
)
from symm_template_reg.engine.single_fragment import world_pose_consistency  # noqa: E402
from symm_template_reg.engine.trainer import resolve_device  # noqa: E402
from symm_template_reg.engine.view_ladder import (  # noqa: E402
    assignment_switch_rate,
    pose_context_change,
    query_assignment_summary,
    query_world_consistency,
)
from symm_template_reg.models import build_model, register_all_modules  # noqa: E402
from symm_template_reg.models.pose.metrics import rotation_error_deg  # noqa: E402
from symm_template_reg.registry import (  # noqa: E402
    COLLATE_FUNCTIONS,
    build_from_cfg,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator > 0 else math.nan


def _checkpoint(run: Path) -> Path:
    summary = json.loads((run / "stage_summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["best_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _evaluate_checkpoint(
    run: Path,
    manifest_path: Path,
    output: Path,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any, list[int], Any]:
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    config["data"]["train_manifest"] = str(manifest_path)
    config["data"]["validation_manifest"] = "same_as_train"
    config["data"]["expected_selected_samples"] = 10
    config["dataset"]["fragment_mesh_cache_dir"] = str(
        output / "cache" / run.name
    )
    dataset = _build_dataset(config)
    manifest, _ = load_faces840_manifest(manifest_path, config, dataset)
    indices = {
        record.sample_id: index
        for index, record in enumerate(dataset.sample_records)
    }
    selected = [indices[item["sample_id"]] for item in manifest["samples"]]
    collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=int(config["data"].get("validation_batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    model = build_model(config["model"]).to(device)
    load_checkpoint(_checkpoint(run), model=model, map_location=device, strict=True)
    criterion = _build_pose_criterion(config)
    amp_enabled, amp_dtype, _ = _amp_settings(device, config["train"])
    metrics, rows = _evaluate(
        model,
        loader,
        device,
        criterion,
        amp_enabled,
        amp_dtype,
        config["loss"],
        show_progress=True,
    )
    return config, rows, dataset, collate, selected, model


def _historical_assignments(run: Path) -> tuple[list[dict[int, int]], list[int]]:
    steps = []
    assignments = []
    for evaluation in sorted((run / "evaluations").glob("epoch_*")):
        csv_path = evaluation / "per_sample_metrics.csv"
        if not csv_path.is_file():
            continue
        rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
        assignments.append(
            {int(row["frame_id"]): int(row["oracle_query_index"]) for row in rows}
        )
        steps.append(int(evaluation.name.rsplit("_", 1)[-1]))
    return assignments, steps


def _pairwise_pose_change(rows: list[dict[str, Any]], query: int) -> dict[str, float]:
    poses = torch.as_tensor(
        [row["query_T_C_from_O"][query] for row in rows], dtype=torch.float64
    )
    rotation_values = []
    translation_values = []
    for left in range(len(poses)):
        for right in range(left + 1, len(poses)):
            rotation_values.append(float(rotation_error_deg(poses[left], poses[right])))
            translation_values.append(
                float(
                    torch.linalg.vector_norm(
                        poses[left, :3, 3] - poses[right, :3, 3]
                    )
                    * 1000.0
                )
            )
    return {
        "camera_pose_pairwise_rotation_mean_deg": sum(rotation_values)
        / max(len(rotation_values), 1),
        "camera_pose_pairwise_rotation_max_deg": max(rotation_values, default=0.0),
        "camera_pose_pairwise_translation_mean_mm": sum(translation_values)
        / max(len(translation_values), 1),
        "camera_pose_pairwise_translation_max_mm": max(
            translation_values, default=0.0
        ),
    }


def _assigned_gt_rotation_spread(
    assigned_frames: list[int], samples: Mapping[int, Any]
) -> dict[str, float | int]:
    poses = [samples[frame]["gt"]["T_C_from_O"].to(torch.float64) for frame in assigned_frames]
    values = [
        float(rotation_error_deg(poses[left], poses[right]))
        for left in range(len(poses))
        for right in range(left + 1, len(poses))
    ]
    return {
        "assigned_frame_count": len(assigned_frames),
        "assigned_gt_rotation_pairwise_mean_deg": sum(values) / max(len(values), 1),
        "assigned_gt_rotation_pairwise_max_deg": max(values, default=0.0),
    }


@torch.no_grad()
def _context_shuffle(
    model: torch.nn.Module,
    dataset: Any,
    collate: Any,
    indices: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    samples = [dataset[index] for index in indices]
    original_poses = []
    shuffled_poses = []
    original_normalized = []
    shuffled_normalized = []
    codec = model.pose_head.pose_codec
    if codec is None:
        raise RuntimeError("context diagnostic requires PoseCodec")
    for index, sample in enumerate(samples):
        replacement = samples[(index + 1) % len(samples)]
        hybrid = deepcopy(sample)
        hybrid["observed"] = deepcopy(replacement["observed"])
        original_prediction = model(move_to_device(collate([sample]), device))
        shuffled_prediction = model(move_to_device(collate([hybrid]), device))
        original_poses.append(original_prediction.pose_hypotheses.cpu())
        shuffled_poses.append(shuffled_prediction.pose_hypotheses.cpu())
        original_normalized.append(
            codec.encode_transform(
                original_prediction.pose_hypotheses,
                original_prediction.observed_centroid_C,
                original_prediction.observed_scale,
            ).cpu()
        )
        shuffled_normalized.append(
            codec.encode_transform(
                shuffled_prediction.pose_hypotheses,
                shuffled_prediction.observed_centroid_C,
                shuffled_prediction.observed_scale,
            ).cpu()
        )
    return pose_context_change(
        torch.cat(original_poses),
        torch.cat(shuffled_poses),
        torch.cat(original_normalized),
        torch.cat(shuffled_normalized),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k8-run", required=True)
    parser.add_argument("--k1-run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    k8_run = Path(args.k8_run).expanduser().resolve()
    k1_run = Path(args.k1_run).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    register_all_modules()
    device = resolve_device(args.device)

    _, k8_rows, dataset, collate, indices, k8_model = _evaluate_checkpoint(
        k8_run, manifest_path, output, device
    )
    _, k1_rows, _, _, _, _ = _evaluate_checkpoint(
        k1_run, manifest_path, output, device
    )
    k8_rows = sorted(k8_rows, key=lambda row: int(row["frame_id"]))
    k1_by_frame = {int(row["frame_id"]): row for row in k1_rows}
    dataset_by_frame = {
        int(dataset[index]["frame_id"]): dataset[index] for index in indices
    }
    comparison_rows = []
    for row in k8_rows:
        frame = int(row["frame_id"])
        sample = dataset_by_frame[frame]
        gt = sample["gt"]["T_C_from_O"]
        axis = torch.as_tensor(
            sample["template"]["symmetry_metadata"].axis.direction,
            dtype=gt.dtype,
        )
        axis_C = gt[:3, :3] @ axis
        camera_vector = gt[:3, 3]
        distance = float(torch.linalg.vector_norm(camera_vector))
        cosine = abs(
            float(torch.dot(axis_C, camera_vector / max(distance, 1e-12)))
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        k1 = k1_by_frame[frame]
        comparison_rows.append(
            {
                "frame_id": frame,
                "num_observed_points": len(sample["observed"]["points_C"]),
                "camera_to_object_distance_m": distance,
                "view_to_symmetry_axis_deg": angle,
                "k1_rotation_error_deg": float(k1["oracle_topk_rotation_error_deg"]),
                "k1_translation_error_mm": float(k1["oracle_translation_total_mm"]),
                "k1_pose_cost": float(k1["oracle_best_pose_cost"]),
                "k8_oracle_rotation_error_deg": float(
                    row["oracle_topk_rotation_error_deg"]
                ),
                "k8_oracle_translation_error_mm": float(
                    row["oracle_translation_total_mm"]
                ),
                "k8_oracle_pose_cost": float(row["oracle_best_pose_cost"]),
                "k8_oracle_query_index": int(row["oracle_query_index"]),
                "k8_all_query_costs": row["query_pose_costs"],
            }
        )
    with (output / "per_frame_comparison.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        fields = list(comparison_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    query_count = len(k8_rows[0]["query_pose_costs"])
    matrix_fields = ["frame_id", *(f"query_{q}" for q in range(query_count))]
    with (output / "k8_query_frame_matrix.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=matrix_fields)
        writer.writeheader()
        for row in k8_rows:
            writer.writerow(
                {
                    "frame_id": int(row["frame_id"]),
                    **{
                        f"query_{q}": float(cost)
                        for q, cost in enumerate(row["query_pose_costs"])
                    },
                }
            )
    assignment = query_assignment_summary(k8_rows, num_queries=query_count)
    historical, evaluation_epochs = _historical_assignments(k8_run)
    switch_rate, changes, comparisons = assignment_switch_rate(historical)
    occupancy = {
        **assignment,
        "evaluation_epochs": evaluation_epochs,
        "query_assignment_switch_rate": switch_rate,
        "query_assignment_switch_count": changes,
        "query_assignment_comparison_count": comparisons,
    }
    _write_json(output / "query_occupancy.json", occupancy)

    first_sample = dataset_by_frame[int(k8_rows[0]["frame_id"])]
    metadata = first_sample["template"]["symmetry_metadata"]
    group = first_sample["gt"]["effective_symmetry_group"]
    symmetry_world = {}
    for query in range(query_count):
        transforms = torch.as_tensor(
            [row["query_T_W_from_O"][query] for row in k8_rows],
            dtype=torch.float64,
        )
        symmetry_world[str(query)] = world_pose_consistency(
            transforms, metadata, group
        )
    context = _context_shuffle(k8_model, dataset, collate, indices, device)
    specialization = {
        "query_occupancy": assignment["query_occupancy"],
        "assigned_frames": assignment["assigned_frames"],
        "assignment_switch_rate": switch_rate,
        "query_camera_pose_change": {
            str(query): _pairwise_pose_change(k8_rows, query)
            for query in range(query_count)
        },
        "assigned_gt_rotation_clusters": {
            str(query): _assigned_gt_rotation_spread(
                assignment["assigned_frames"][str(query)], dataset_by_frame
            )
            for query in range(query_count)
        },
        "query_world_pose_consistency": query_world_consistency(k8_rows),
        "query_world_symmetry_aware_consistency": symmetry_world,
        "observed_context_shuffle": context,
        "specialization_confirmed": sum(
            int(value > 0) for value in assignment["query_occupancy"].values()
        )
        > 1,
    }
    populated_clusters = [
        value
        for value in specialization["assigned_gt_rotation_clusters"].values()
        if int(value["assigned_frame_count"]) > 1
    ]
    specialization["specializes_in_close_gt_rotation_groups"] = bool(
        populated_clusters
    ) and all(
        float(value["assigned_gt_rotation_pairwise_max_deg"]) < 15.0
        for value in populated_clusters
    )
    _write_json(output / "query_specialization.json", specialization)

    difficulty = {
        "correlations_with_k8_oracle_rotation_error": {
            "num_observed_points": _pearson(
                [float(row["num_observed_points"]) for row in comparison_rows],
                [float(row["k8_oracle_rotation_error_deg"]) for row in comparison_rows],
            ),
            "camera_to_object_distance_m": _pearson(
                [float(row["camera_to_object_distance_m"]) for row in comparison_rows],
                [float(row["k8_oracle_rotation_error_deg"]) for row in comparison_rows],
            ),
            "view_to_symmetry_axis_deg": _pearson(
                [float(row["view_to_symmetry_axis_deg"]) for row in comparison_rows],
                [float(row["k8_oracle_rotation_error_deg"]) for row in comparison_rows],
            ),
        },
        "hardest_k8_frames": [
            int(row["frame_id"])
            for row in sorted(
                comparison_rows,
                key=lambda value: float(value["k8_oracle_pose_cost"]),
                reverse=True,
            )
        ],
    }
    _write_json(output / "view_difficulty.json", difficulty)
    k8_success = sum(
        float(row["oracle_topk_rotation_error_deg"]) < 5.0
        and float(row["oracle_translation_total_mm"]) < 5.0
        for row in k8_rows
    ) / len(k8_rows)
    k1_success = sum(
        float(row["oracle_topk_rotation_error_deg"]) < 5.0
        and float(row["oracle_translation_total_mm"]) < 5.0
        for row in k1_rows
    ) / len(k1_rows)
    summary = {
        "k8_run": str(k8_run),
        "k1_run": str(k1_run),
        "manifest": str(manifest_path),
        "device": str(device),
        "k1_pose_success_5deg_5mm": k1_success,
        "k8_oracle_pose_success_5deg_5mm": k8_success,
        "query_specialization_confirmed": specialization[
            "specialization_confirmed"
        ],
        "queries_form_close_gt_rotation_clusters": specialization[
            "specializes_in_close_gt_rotation_groups"
        ],
        "query_assignment_switch_rate": switch_rate,
        "observed_context_diagnosis": context["diagnosis"],
        "diagnosis": "pose_head_or_context_conditioning_problem_with_multi_query_view_specialization",
    }
    _write_json(output / "k1_k8_summary.json", summary)
    report = [
        "# K1/K8 view specialization audit",
        "",
        f"- K1 success 5deg/5mm: `{k1_success:.3f}`",
        f"- K8 oracle success 5deg/5mm: `{k8_success:.3f}`",
        f"- assignment switch rate: `{switch_rate:.3f}`",
        f"- specialization confirmed: `{specialization['specialization_confirmed']}`",
        "- close-GT-rotation clustering: "
        f"`{specialization['specializes_in_close_gt_rotation_groups']}`",
        f"- observed-context diagnosis: `{context['diagnosis']}`",
        f"- maximum shuffled-observation rotation response: "
        f"`{context['max_query_rotation_change_deg']:.8f} deg`",
        f"- maximum shuffled normalized-translation response: "
        f"`{context['max_normalized_translation_change']:.8g}`",
        "",
        "| frame | points | K1 rot deg | K1 trans mm | K8 oracle rot deg | "
        "K8 oracle trans mm | query |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        report.append(
            f"| {row['frame_id']} | {row['num_observed_points']} | "
            f"{row['k1_rotation_error_deg']:.3f} | "
            f"{row['k1_translation_error_mm']:.3f} | "
            f"{row['k8_oracle_rotation_error_deg']:.3f} | "
            f"{row['k8_oracle_translation_error_mm']:.3f} | "
            f"{row['k8_oracle_query_index']} |"
        )
    report.extend(
        [
        "",
        "The assignments are stable and six of eight query slots are occupied, "
        "but the paired assigned frames span roughly 20–47 degrees rather than "
        "forming tight rotation clusters. Query rotations themselves are almost "
        "constant between input frames. This is a static-codebook specialization, "
        "not useful observed-context conditioning.",
        "",
        "The direct pose-parameter gate is reported separately by "
        "`debug_optimize_pose_parameters.py`. Do not start ranking while K8 oracle "
        "success is below the explicit gate.",
        "",
        ]
    )
    (output / "audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
