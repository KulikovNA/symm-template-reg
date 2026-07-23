"""Physical metrics for the coordinate-guided active registration path.

The helpers in this module intentionally know nothing about legacy pose
queries, patch/triangle classifiers, regions, ranking, or learned confidence.
They are shared by training evaluation and read-only checkpoint audits so the
stage gate cannot silently switch to an inactive pose source.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from symm_template_reg.engine.single_fragment import world_pose_consistency
from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import (
    AuxGuidedTriangleCandidateBuilder,
)
from symm_template_reg.models.heads.coordinate_guided_surface_projection import (
    CoordinateGuidedSurfaceProjectionHead,
)
from symm_template_reg.models.pose.pose_representation import transform_points
from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance


ACTIVE_EXACT_PREFIX = "eval/active/exact_global"
ACTIVE_K16_PREFIX = "eval/active/k16"


def _distance_summary_mm(values_m: Tensor, prefix: str) -> dict[str, float]:
    values = values_m.detach().float()
    return {
        f"{prefix}_rmse_mm": float(values.square().mean().sqrt() * 1000.0),
        f"{prefix}_p50_mm": float(torch.quantile(values, 0.50) * 1000.0),
        f"{prefix}_p95_mm": float(torch.quantile(values, 0.95) * 1000.0),
        f"{prefix}_max_mm": float(values.max() * 1000.0),
    }


def active_values_are_finite(values: Mapping[str, Any]) -> bool:
    """Return whether every numeric active-path value is finite."""

    for value in values.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


STRICT_THRESHOLDS = {
    "projected_correspondence_p95_mm": 1.0,
    "alignment_p95_mm": 1.0,
    "rotation_error_deg": 1.0,
    "translation_error_mm": 1.0,
}

PRACTICAL_THRESHOLDS = {
    "projected_correspondence_p95_mm": 1.5,
    "alignment_p95_mm": 1.5,
    "rotation_error_deg": 0.25,
    "translation_error_mm": 0.10,
}

TEN_VIEW_STRICT_THRESHOLDS = {
    "projected_correspondence_p95_mm": 1.0,
    "alignment_p95_mm": 1.0,
    "rotation_error_deg": 0.25,
    "translation_error_mm": 0.10,
}

TEN_VIEW_PRACTICAL_THRESHOLDS = {
    "projected_correspondence_p95_mm": 2.0,
    "alignment_p95_mm": 2.0,
    "rotation_error_deg": 0.50,
    "translation_error_mm": 0.50,
}


def _active_sample_gate(
    row: Mapping[str, Any], thresholds: Mapping[str, float], gate_name: str
) -> dict[str, Any]:

    checks = {
        "projected_correspondence_p95_mm": float(
            row["exact_global_projected_correspondence_p95_mm"]
        )
        <= float(thresholds["projected_correspondence_p95_mm"]) + 1e-6,
        "alignment_p95_mm": float(row["exact_global_projection_alignment_p95_mm"])
        <= float(thresholds["alignment_p95_mm"]) + 1e-6,
        "rotation_error_deg": float(row["exact_global_projection_rotation_error_deg"])
        <= float(thresholds["rotation_error_deg"]) + 1e-6,
        "translation_error_mm": float(
            row["exact_global_projection_translation_error_mm"]
        )
        <= float(thresholds["translation_error_mm"]) + 1e-6,
        "rank_three": int(row["exact_global_projection_rank"]) == 3,
        "surface_membership_p95_mm": float(
            row["exact_global_surface_membership_p95_mm"]
        )
        <= 0.1 + 1e-6,
        "k16_exact_global_recall": float(row["k16_exact_global_triangle_recall"])
        >= 0.995 - 1e-6,
        "k16_zero_fallback": float(row["k16_fallback_fraction"]) <= 1e-6,
        "target_leakage": not bool(row.get("target_leakage_detected", False)),
        "active_path_nonfinite": not bool(row.get("active_nonfinite_detected", False)),
    }
    return {
        "sample_id": row.get("sample_id"),
        "frame_id": (
            None if row.get("frame_id") is None else int(row["frame_id"])
        ),
        "gate_name": gate_name,
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
    }


def active_sample_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible strict submillimetre gate for one frame."""

    return _active_sample_gate(row, STRICT_THRESHOLDS, "strict_submillimetre_gate")


def practical_sample_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    return _active_sample_gate(row, PRACTICAL_THRESHOLDS, "practical_pose_first_gate")


def strict_surface_sample_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _active_sample_gate(row, TEN_VIEW_STRICT_THRESHOLDS, "strict_surface_gate")
    checks = {name: value for name, value in result["checks"].items() if name != "surface_membership_p95_mm"}
    return {**result, "checks": checks, "passed": all(checks.values()), "failures": [name for name, value in checks.items() if not value]}


def practical_surface_sample_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = _active_sample_gate(row, TEN_VIEW_PRACTICAL_THRESHOLDS, "practical_surface_gate")
    checks = {name: value for name, value in result["checks"].items() if name != "surface_membership_p95_mm"}
    return {**result, "checks": checks, "passed": all(checks.values()), "failures": [name for name, value in checks.items() if not value]}


def pose_placement_sample_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    full = _active_sample_gate(row, TEN_VIEW_PRACTICAL_THRESHOLDS, "pose_placement_gate")
    retained = {
        name: passed for name, passed in full["checks"].items()
        if name not in {"projected_correspondence_p95_mm", "alignment_p95_mm"}
    }
    return {
        **full,
        "checks": retained,
        "passed": all(retained.values()),
        "failures": [name for name, passed in retained.items() if not passed],
    }


def _stage_gate(
    rows: Sequence[Mapping[str, Any]], expected_frames: Sequence[int],
    sample_gate: Any, gate_name: str,
) -> dict[str, Any]:
    per_sample = [sample_gate(row) for row in rows]
    frames = [int(item["frame_id"]) for item in per_sample]
    expected = list(map(int, expected_frames))
    frame_contract_passed = len(frames) == len(expected) and set(frames) == set(expected)
    passed = frame_contract_passed and all(item["passed"] for item in per_sample)
    return {
        "stage_passed": passed,
        "next_stage_allowed": passed,
        "gate_name": gate_name,
        "active_path_only": True,
        "expected_frames": expected,
        "observed_frames": frames,
        "frame_contract_passed": frame_contract_passed,
        "per_sample": per_sample,
        "failures": [] if passed else [
            *("frame_contract" for _ in [0] if not frame_contract_passed),
            *(f"frame_{item['frame_id']}" for item in per_sample if not item["passed"]),
        ],
    }


def four_view_stage_gate(rows: Sequence[Mapping[str, Any]], expected_frames=(4, 5, 2, 8)) -> dict[str, Any]:
    return _stage_gate(
        rows, expected_frames, active_sample_gate, "strict_submillimetre_gate"
    )


def strict_and_practical_stage_gates(
    rows: Sequence[Mapping[str, Any]],
    expected_frames=(4, 5, 2, 8, 0, 1, 6, 9),
) -> dict[str, Any]:
    strict = _stage_gate(
        rows, expected_frames, active_sample_gate, "strict_submillimetre_gate"
    )
    practical = _stage_gate(
        rows, expected_frames, practical_sample_gate, "practical_pose_first_gate"
    )
    return {
        "strict_submillimetre_gate": strict,
        "practical_pose_first_gate": practical,
        "stage_passed": practical["stage_passed"],
        "next_stage_allowed": practical["stage_passed"],
        "transition_policy": "practical_pose_first_gate",
        "strict_is_diagnostic": True,
    }


def ten_view_stage_gates(
    rows: Sequence[Mapping[str, Any]], expected_frames=tuple(range(10)),
) -> dict[str, Any]:
    strict = _stage_gate(rows, expected_frames, strict_surface_sample_gate, "strict_surface_gate")
    practical = _stage_gate(rows, expected_frames, practical_surface_sample_gate, "practical_surface_gate")
    pose = _stage_gate(rows, expected_frames, pose_placement_sample_gate, "pose_placement_gate")
    return {
        "strict_surface_gate": strict,
        "practical_surface_gate": practical,
        "pose_placement_gate": pose,
        "stage_passed": practical["stage_passed"],
        "next_stage_allowed": practical["stage_passed"],
        "transition_policy": "practical_surface_gate",
        "strict_failure_is_never_rewritten": True,
    }


def worst_sample_projection_score(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("at least one active sample is required")
    return max(
        float(row["exact_global_projected_correspondence_p95_mm"])
        + float(row["exact_global_projection_alignment_p95_mm"])
        + float(row["exact_global_projection_rotation_error_deg"])
        + float(row["exact_global_projection_translation_error_mm"])
        for row in rows
    )


def practical_sample_score(row: Mapping[str, Any]) -> float:
    return (
        float(row["exact_global_projected_correspondence_p95_mm"]) / 1.5
        + float(row["exact_global_projection_alignment_p95_mm"]) / 1.5
        + float(row["exact_global_projection_rotation_error_deg"]) / 0.25
        + float(row["exact_global_projection_translation_error_mm"]) / 0.10
    )


def worst_sample_practical_score(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("at least one active sample is required")
    return max(practical_sample_score(row) for row in rows)


def ten_view_sample_score(row: Mapping[str, Any]) -> float:
    return (
        float(row["exact_global_projected_correspondence_p95_mm"]) / 2.0
        + float(row["exact_global_projection_alignment_p95_mm"]) / 2.0
        + float(row["exact_global_projection_rotation_error_deg"]) / 0.25
        + float(row["exact_global_projection_translation_error_mm"]) / 0.10
    )


def worst_ten_view_sample_score(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise ValueError("at least one active sample is required")
    return max(ten_view_sample_score(row) for row in rows)


def _pose_metrics(
    projected_O: Tensor,
    observed_C: Tensor,
    equivalent_pose: Tensor,
    procrustes: Any,
) -> tuple[dict[str, float | int], Tensor]:
    valid = torch.ones((1, len(projected_O)), dtype=torch.bool, device=projected_O.device)
    solution = procrustes.solve(
        projected_O[None].float(),
        observed_C[None].float(),
        projected_O.new_ones((1, len(projected_O))).float(),
        valid,
    )
    pose = solution["transform"][0].to(projected_O)
    reconstructed = transform_points(pose[None], projected_O[None])[0]
    alignment = torch.linalg.vector_norm(reconstructed - observed_C, dim=-1)
    rotation = torch.rad2deg(
        rotation_geodesic_distance(
            pose[:3, :3][None], equivalent_pose[:3, :3][None]
        )
    )[0]
    translation = torch.linalg.vector_norm(
        pose[:3, 3] - equivalent_pose[:3, 3]
    ) * 1000.0
    return {
        **_distance_summary_mm(alignment, "projection_alignment"),
        "projection_rotation_error_deg": float(rotation),
        "projection_translation_error_mm": float(translation),
        "projection_rank": int(solution["rank"][0]),
        "projection_rank_valid": bool(solution["rank_valid"][0]),
    }, pose


@torch.no_grad()
def evaluate_active_sample(
    *,
    q_aux_O: Tensor,
    valid_mask: Tensor,
    target_O: Tensor,
    observed_C: Tensor,
    vertices_O: Tensor,
    faces: Tensor,
    equivalent_pose: Tensor,
    procrustes: Any,
    candidate_k: int = 16,
    projection_chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate exact-global and q_aux-guided K16 projected poses."""

    mask = valid_mask.bool()
    q_valid = q_aux_O[mask]
    target_valid = target_O[mask]
    observed_valid = observed_C[mask]
    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    exact_started = time.perf_counter()
    exact = closest_points_on_triangle_mesh(
        q_valid, vertices_O, faces.long(), point_chunk_size=projection_chunk_size
    )
    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    exact_runtime_ms = (time.perf_counter() - exact_started) * 1000.0
    exact_points = exact["points"]
    exact_ids = exact["face_ids"].long()

    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    k16_started = time.perf_counter()
    builder = AuxGuidedTriangleCandidateBuilder(
        mode="aux_guided_global_topk",
        candidate_k=int(candidate_k),
        projection_chunk_size=int(projection_chunk_size),
    ).to(q_aux_O.device)
    built = builder(q_aux_O[None], [vertices_O], [faces], mask[None])
    ids = built["candidate_triangle_ids"]
    candidate_mask = built["candidate_triangle_mask"]
    projected = CoordinateGuidedSurfaceProjectionHead().to(q_aux_O.device)(
        q_aux_O[None], ids, [vertices_O], [faces], mask[None], candidate_mask
    )
    k16_points = projected["surface_correspondence_points_O"][0, mask]
    k16_ids = projected["selected_triangle_ids"][0, mask]
    if q_aux_O.device.type == "cuda":
        torch.cuda.synchronize(q_aux_O.device)
    k16_runtime_ms = (time.perf_counter() - k16_started) * 1000.0
    shortlist = ids[0, mask]
    shortlist_mask = candidate_mask[0, mask]
    recall = ((shortlist == exact_ids[:, None]) & shortlist_mask).any(-1).float().mean()
    fallback_fraction = float((~shortlist_mask.any(-1)).float().mean())

    modes: dict[str, Any] = {}
    poses: dict[str, Tensor] = {}
    for name, points, selected_ids in (
        ("exact_global", exact_points, exact_ids),
        ("k16", k16_points, k16_ids),
    ):
        pose_values, pose = _pose_metrics(
            points, observed_valid, equivalent_pose, procrustes
        )
        triangles = vertices_O[faces.long()[selected_ids]]
        # Projection points are analytically on their selected triangles; the
        # residual below is retained as a numerical contract check.
        from symm_template_reg.models.geometry.triangle_targets import (
            closest_barycentric_on_triangles,
        )

        membership = closest_barycentric_on_triangles(points, triangles)["distances"]
        values = {
            **_distance_summary_mm(
                torch.linalg.vector_norm(points - target_valid, dim=-1),
                "projected_correspondence",
            ),
            **pose_values,
            "surface_membership_p95_mm": float(
                torch.quantile(membership.float(), 0.95) * 1000.0
            ),
            "runtime_ms": (
                exact_runtime_ms if name == "exact_global" else k16_runtime_ms
            ),
        }
        values["nonfinite_detected"] = not active_values_are_finite(values)
        modes[name] = values
        poses[name] = pose
    modes["k16"].update(
        exact_global_triangle_recall=float(recall),
        fallback_fraction=fallback_fraction,
        candidate_count_min=int(shortlist_mask.sum(-1).min()),
        candidate_count_max=int(shortlist_mask.sum(-1).max()),
    )
    return {
        "num_shell_points": int(mask.sum()),
        "raw_q_aux": _distance_summary_mm(
            torch.linalg.vector_norm(q_valid - target_valid, dim=-1), "aux_coordinate"
        ),
        "exact_global": modes["exact_global"],
        "k16": modes["k16"],
        "T_C_from_O": poses,
        "projected_points_O": {
            "exact_global": exact_points,
            "k16": k16_points,
        },
        "selected_triangle_ids": {
            "exact_global": exact_ids,
            "k16": k16_ids,
        },
    }


def active_row(
    result: Mapping[str, Any], *, sample_id: str, frame_id: int,
    T_W_from_C: Tensor, target_leakage_detected: bool = False,
) -> dict[str, Any]:
    exact = result["exact_global"]
    k16 = result["k16"]
    exact_pose = result["T_C_from_O"]["exact_global"]
    k16_pose = result["T_C_from_O"]["k16"]
    values = {
        "sample_id": str(sample_id),
        "frame_id": int(frame_id),
        "num_shell_points": int(result.get("num_shell_points", 0)),
        **result["raw_q_aux"],
        **{f"exact_global_{key}": value for key, value in exact.items()},
        **{f"k16_{key}": value for key, value in k16.items()},
        "k16_exact_global_triangle_recall": k16["exact_global_triangle_recall"],
        "k16_fallback_fraction": k16["fallback_fraction"],
        "exact_global_T_C_from_O": exact_pose.detach().cpu().tolist(),
        "k16_T_C_from_O": k16_pose.detach().cpu().tolist(),
        "exact_global_T_W_from_O": (T_W_from_C @ exact_pose).detach().cpu().tolist(),
        "k16_T_W_from_O": (T_W_from_C @ k16_pose).detach().cpu().tolist(),
        "target_leakage_detected": bool(target_leakage_detected),
    }
    values["exact_global_projection_score"] = (
        float(values["exact_global_projected_correspondence_p95_mm"])
        + float(values["exact_global_projection_alignment_p95_mm"])
        + float(values["exact_global_projection_rotation_error_deg"])
        + float(values["exact_global_projection_translation_error_mm"])
    )
    values["exact_global_practical_score"] = practical_sample_score(values)
    values["active_nonfinite_detected"] = not active_values_are_finite(
        {
            key: value
            for key, value in values.items()
            if isinstance(value, (bool, int, float)) and key != "target_leakage_detected"
        }
    )
    values["exact_global_sample_gate_passed"] = active_sample_gate(values)["passed"]
    values["strict_submillimetre_sample_gate_passed"] = active_sample_gate(values)["passed"]
    values["practical_pose_first_sample_gate_passed"] = practical_sample_gate(values)["passed"]
    return values


def active_world_metrics(
    rows: Sequence[Mapping[str, Any]], symmetry_metadata: Any, effective_group: Any
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for mode in ("exact_global", "k16"):
        transforms = torch.as_tensor(
            [row[f"{mode}_T_W_from_O"] for row in rows], dtype=torch.float64
        )
        for name, value in world_pose_consistency(
            transforms, symmetry_metadata, effective_group
        ).items():
            metrics[f"{mode}_{name}"] = value
    return metrics


def active_world_pairwise_matrices(
    rows: Sequence[Mapping[str, Any]], symmetry_metadata: Any,
    effective_group: Any, mode: str = "exact_global",
) -> dict[str, list[list[float]]]:
    poses = torch.as_tensor(
        [row[f"{mode}_T_W_from_O"] for row in rows], dtype=torch.float64
    )
    translations = torch.cdist(poses[:, :3, 3], poses[:, :3, 3]) * 1000.0
    axis = torch.as_tensor(symmetry_metadata.axis.direction, dtype=poses.dtype)
    axes = torch.nn.functional.normalize(poses[:, :3, :3] @ axis, dim=-1)
    axis_pairwise = torch.rad2deg(
        torch.acos(torch.abs(axes @ axes.T).clamp(-1.0, 1.0))
    )
    rotation_rows = []
    for index in range(len(poses)):
        rotation_rows.append(
            symmetry_aware_pose_errors(
                poses, poses[index], symmetry_metadata,
                effective_group=effective_group,
            )["rotation_deg"]
        )
    rotations = torch.stack(rotation_rows)
    return {
        "world_translation_pairwise_mm": translations.tolist(),
        "world_axis_pairwise_deg": axis_pairwise.tolist(),
        "world_rotation_pairwise_deg": rotations.tolist(),
    }


def diagnose_four_view_failure(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify a failed fine-only run without authorizing another stage."""

    gates = {int(row["frame_id"]): active_sample_gate(row) for row in rows}
    exact_pass = {
        frame: all(
            passed
            for name, passed in gate["checks"].items()
            if name not in {"k16_exact_global_recall", "k16_zero_fallback"}
        )
        for frame, gate in gates.items()
    }
    k16_pass = {
        frame: gate["checks"]["k16_exact_global_recall"]
        and gate["checks"]["k16_zero_fallback"]
        for frame, gate in gates.items()
    }
    diagnosis = None
    if all(exact_pass.get(frame, False) for frame in (2, 5)) and any(
        not exact_pass.get(frame, False) for frame in (4, 8)
    ):
        diagnosis = "catastrophic_forgetting_of_solved_views"
    elif exact_pass and not any(exact_pass.values()):
        diagnosis = "shared_fine_adapter_capacity_limit"
    elif sum(not exact_pass.get(frame, False) for frame in (2, 5)) == 1:
        diagnosis = "view_specific_tail_failure"
    elif any(
        float(row.get("aux_coordinate_p95_mm", math.inf)) > 1.0 + 1e-6
        and exact_pass.get(int(row["frame_id"]), False)
        for row in rows
    ):
        diagnosis = "raw_coordinate_threshold_not_equal_to_surface_path_failure"
    elif any(
        exact_pass.get(frame, False) and not k16_pass.get(frame, False)
        for frame in exact_pass
    ):
        diagnosis = "shortlist_failure"
    if diagnosis is None and any(not gate["passed"] for gate in gates.values()):
        diagnosis = "unclassified_active_path_failure"
    return {
        "diagnosis": diagnosis,
        "selective_unfreeze_allowed": False,
        "recommended_fallback": (
            "inspect per-frame active metrics; selective unfreeze requires an explicit external decision"
            if diagnosis is not None
            else None
        ),
        "stop_after_analysis": diagnosis is not None,
    }


def diagnose_eight_view_failure(
    rows: Sequence[Mapping[str, Any]], *, best_epoch: int | None = None,
    last_evaluation_epoch: int | None = None,
) -> dict[str, Any]:
    practical_gates = {
        int(row["frame_id"]): practical_sample_gate(row) for row in rows
    }


def diagnose_ten_view_scratch_failure(
    rows: Sequence[Mapping[str, Any]], *, best_epoch: int | None = None,
    last_evaluation_epoch: int | None = None,
) -> dict[str, Any]:
    gates = ten_view_stage_gates(rows)
    pose_count = sum(item["passed"] for item in gates["pose_placement_gate"]["per_sample"])
    practical_count = sum(item["passed"] for item in gates["practical_surface_gate"]["per_sample"])
    diagnosis = None
    if any(bool(row.get("active_nonfinite_detected", False)) for row in rows):
        diagnosis = "numerical_training_failure"
    else:
        summaries = [row.get("correspondence_prediction_summary") for row in rows]
        if summaries and all(
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            for value in summaries
        ):
            values = torch.as_tensor(summaries, dtype=torch.float64)
            if len(values) > 1 and float(torch.pdist(values).mean()) <= 1e-6:
                diagnosis = "input_conditioning_collapse"
    if diagnosis is None and pose_count == len(rows) and practical_count < len(rows):
        diagnosis = "pose_good_surface_correspondence_tail_failure"
    elif diagnosis is None and 0 < practical_count < len(rows):
        diagnosis = "view_specific_optimization_failure"
    elif diagnosis is None and practical_count == 0:
        worst = max(
            float(row["exact_global_projected_correspondence_p95_mm"])
            for row in rows
        )
        diagnosis = (
            "scratch_optimization_failure" if worst > 20.0
            else "shared_representation_capacity_limit"
        )
    if (
        diagnosis is None and best_epoch is not None
        and last_evaluation_epoch is not None and best_epoch == last_evaluation_epoch
    ):
        diagnosis = "undertrained_but_improving"
    return {
        "diagnosis": diagnosis,
        "warmstart_allowed": False,
        "recommended_action": "STOP and externally inspect compact diagnostics" if diagnosis else None,
        "stop_after_analysis": diagnosis is not None,
    }
    practical = {frame: gate["passed"] for frame, gate in practical_gates.items()}
    exact_practical_pass = {
        frame: all(
            passed for name, passed in gate["checks"].items()
            if name not in {"k16_exact_global_recall", "k16_zero_fallback"}
        )
        for frame, gate in practical_gates.items()
    }
    old = (4, 5, 2, 8)
    new = (0, 1, 6, 9)
    diagnosis = None
    if any(
        exact_practical_pass.get(int(row["frame_id"]), False)
        and (
            not practical_gates[int(row["frame_id"])]["checks"]["k16_exact_global_recall"]
            or not practical_gates[int(row["frame_id"])]["checks"]["k16_zero_fallback"]
        )
        for row in rows
    ):
        diagnosis = "shortlist_failure"
    elif any(
        float(row.get("aux_coordinate_p95_mm", math.inf)) > 1.5 + 1e-6
        and exact_practical_pass.get(int(row["frame_id"]), False)
        for row in rows
    ):
        diagnosis = "raw_coordinate_error_not_blocking_surface_path"
    elif any(not practical.get(frame, False) for frame in old) and sum(
        practical.get(frame, False) for frame in new
    ) >= 3:
        diagnosis = "catastrophic_forgetting_of_four_view_solution"
    elif practical and not any(practical.values()):
        diagnosis = "shared_fine_adapter_capacity_limit"
    elif 1 <= sum(not practical.get(frame, False) for frame in new) <= 2:
        diagnosis = "view_specific_tail_failure"
    elif (
        best_epoch is not None and last_evaluation_epoch is not None
        and best_epoch == last_evaluation_epoch
    ):
        diagnosis = "undertrained_but_improving"
    if diagnosis is None and not all(practical.values()):
        diagnosis = "unclassified_active_path_failure"
    return {
        "diagnosis": diagnosis,
        "selective_unfreeze_allowed": False,
        "recommended_fallback": (
            "STOP and inspect strict/practical per-frame metrics; selective "
            "unfreeze requires an explicit external decision"
            if diagnosis else None
        ),
        "stop_after_analysis": diagnosis is not None,
    }


__all__ = [
    "ACTIVE_EXACT_PREFIX",
    "ACTIVE_K16_PREFIX",
    "active_row",
    "active_sample_gate",
    "active_values_are_finite",
    "active_world_metrics",
    "active_world_pairwise_matrices",
    "diagnose_eight_view_failure",
    "diagnose_ten_view_scratch_failure",
    "diagnose_four_view_failure",
    "evaluate_active_sample",
    "four_view_stage_gate",
    "practical_sample_gate",
    "practical_sample_score",
    "practical_surface_sample_gate",
    "pose_placement_sample_gate",
    "strict_surface_sample_gate",
    "strict_and_practical_stage_gates",
    "ten_view_sample_score",
    "ten_view_stage_gates",
    "worst_ten_view_sample_score",
    "worst_sample_practical_score",
    "worst_sample_projection_score",
]
