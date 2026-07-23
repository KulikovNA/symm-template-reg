"""Compact best-evaluation artifacts and physical stage gate for the joint baseline."""

from __future__ import annotations

import csv
import ast
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from symm_template_reg.evaluation.patch_stage import (
    candidate_stage_gate,
    top1_quality_gate,
)
from symm_template_reg.evaluation.local_stage import check_local_substage
from symm_template_reg.evaluation.fine_stage import (
    fine_coordinate_gate,
    fine_triangle_gate,
)
from symm_template_reg.evaluation.active_coordinate import (
    active_world_pairwise_matrices,
    active_world_metrics,
    diagnose_eight_view_failure,
    diagnose_ten_view_scratch_failure,
    diagnose_four_view_failure,
    four_view_stage_gate,
    strict_and_practical_stage_gates,
    ten_view_stage_gates,
    worst_sample_practical_score,
    worst_sample_projection_score,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.models.symmetry.metadata import load_symmetry_metadata
from symm_template_reg.engine.multifragment_overfit import (
    aggregate_metrics,
    diagnose_multifragment_failure,
    multifragment_stage_gates,
    worst_multifragment_sample_score,
)

WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1.0
    if text in {"false", "no"}:
        return 0.0
    return float(value)


def materialize_best_evaluation(run_dir: str | Path, best_epoch: int) -> Path:
    run = Path(run_dir).expanduser().resolve()
    source = run / "evaluations" / f"epoch_{int(best_epoch):04d}"
    destination = run / "best_evaluation"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    resolved = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    clean_active_only = bool(
        resolved.get("active_coordinate_path", {}).get("clean_active_only", False)
    )
    shutil.copy2(source / "per_sample_metrics.csv", destination / "per_sample_metrics.csv")
    conditioning = source / "context_conditioning_diagnostics.json"
    if conditioning.is_file():
        shutil.copy2(conditioning, destination / "context_conditioning_diagnostics.json")
    with (source / "per_sample_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        all_rows = list(csv.DictReader(stream))
    for filename, prefixes in (
        ("exact_global_metrics.csv", ("exact_global_", "aux_coordinate_")),
        ("k16_metrics.csv", ("k16_",)),
    ):
        identity = {"sample_id", "frame_id", "num_shell_points"}
        fields = [
            key for key in all_rows[0]
            if key in identity or any(key.startswith(prefix) for prefix in prefixes)
        ]
        with (destination / filename).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in fields} for row in all_rows)
    _json(destination / "evaluation_summary.json", metrics)
    categories = {
        "loss_breakdown.json": ("loss_", "weighted_loss", "raw_"),
        "correspondence_metrics.json": ("correspondence", "surface", "weight"),
        "pose_metrics.json": ("rotation", "translation", "pose_success", "physical"),
        "alignment_metrics.json": ("alignment",),
        "symmetry_selection.json": ("symmetry",),
        "procrustes_diagnostics.json": ("procrustes", "determinant", "orthogonality", "rank"),
        "triangle_classifier_metrics.json": ("triangle", "local_fine"),
        "random_baseline.json": ("random", "uniform"),
        "barycentric_metrics.json": ("barycentric",),
        "canonical_coordinate_metrics.json": ("correspondence",),
    }
    if clean_active_only:
        categories = {
            name: tokens
            for name, tokens in categories.items()
            if name not in {
                "triangle_classifier_metrics.json", "barycentric_metrics.json",
                "random_baseline.json",
            }
        }
    for filename, tokens in categories.items():
        aggregate = {key: value for key, value in metrics.items() if any(token in key for token in tokens)}
        rows = []
        with (source / "per_sample_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append({key: value for key, value in row.items() if key in {"sample_id", "frame_id"} or any(token in key for token in tokens)})
        _json(destination / filename, {**WARNING_FLAGS, "epoch": int(best_epoch), "aggregate": aggregate, "per_sample": rows})
    return destination


def check_joint_stage(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    best = json.loads((run / "checkpoints" / "best_metrics.json").read_text(encoding="utf-8"))
    best_epoch = int(best["epoch"])
    best_eval = run / "best_evaluation"
    if not best_eval.is_dir():
        best_eval = materialize_best_evaluation(run, best_epoch)
    with (best_eval / "per_sample_metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    resolved = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    audit_path = resolved.get("target_leakage_policy", {}).get("audit_path")
    audit_verified = bool(audit_path) and Path(str(audit_path)).is_file()
    leakage = None
    if audit_verified:
        audit = json.loads(Path(str(audit_path)).read_text(encoding="utf-8"))
        leakage = bool(audit.get("target_leakage_detected", True))
    active_path_config = resolved.get("active_coordinate_path", {})
    if bool(active_path_config.get("enabled", False)):
        expected_frames = tuple(
            map(int, active_path_config.get("expected_frames", (4, 5, 2, 8)))
        )
        active_rows = []
        for row in rows:
            converted = dict(row)
            for key, value in list(converted.items()):
                if key in {
                    "sample_id",
                    "exact_global_T_C_from_O",
                    "k16_T_C_from_O",
                    "exact_global_T_W_from_O",
                    "k16_T_W_from_O",
                }:
                    continue
                try:
                    converted[key] = _number(value)
                except (TypeError, ValueError):
                    pass
            # CSV serialization stores matrices as Python list strings.  The
            # active world metrics are already materialized by the evaluator;
            # stage gating itself only consumes physical scalar fields.
            active_rows.append(converted)
        dual_gates = bool(active_path_config.get("dual_gates", False))
        ten_gates = bool(active_path_config.get("ten_view_gates", False))
        multifragment_gates_enabled = bool(
            active_path_config.get("multifragment_gates", False)
        )
        pose_gate = None
        if multifragment_gates_enabled:
            combined = multifragment_stage_gates(active_rows)
            strict_gate = combined["strict_surface_gate"]
            practical_gate = combined["practical_surface_gate"]
            pose_gate = combined["pose_placement_gate"]
            gate = practical_gate
        elif ten_gates:
            combined = ten_view_stage_gates(active_rows, expected_frames=expected_frames)
            strict_gate = combined["strict_surface_gate"]
            practical_gate = combined["practical_surface_gate"]
            pose_gate = combined["pose_placement_gate"]
            gate = practical_gate
        elif dual_gates:
            combined = strict_and_practical_stage_gates(
                active_rows, expected_frames=expected_frames
            )
            strict_gate = combined["strict_submillimetre_gate"]
            practical_gate = combined["practical_pose_first_gate"]
            gate = practical_gate
        else:
            combined = None
            strict_gate = practical_gate = pose_gate = None
            gate = four_view_stage_gate(active_rows, expected_frames=expected_frames)
        if not audit_verified or leakage is not False:
            gate["stage_passed"] = False
            gate["next_stage_allowed"] = False
            gate["failures"].append("target_leakage_audit")
        nonfinite = any(
            bool(_number(row.get("active_nonfinite_detected", 1.0)))
            for row in active_rows
        )
        if nonfinite:
            gate["stage_passed"] = False
            gate["next_stage_allowed"] = False
            if "active_path_nonfinite" not in gate["failures"]:
                gate["failures"].append("active_path_nonfinite")
        active_world = {
            key.removeprefix("eval/active/world/"): value
            for key, value in best.get("metrics", {}).items()
            if key.startswith("eval/active/world/")
        }
        pairwise = None
        if dual_gates or ten_gates:
            run_manifest = json.loads(
                (run / "run_manifest.json").read_text(encoding="utf-8")
            )
            metadata = load_symmetry_metadata(run_manifest["symmetry_sidecar_path"])
            manifest_payload = json.loads(
                Path(run_manifest["train_manifest"]).read_text(encoding="utf-8")
            )
            effective_group = parse_rotation_group(
                manifest_payload["samples"][0]["effective_symmetry_group"]
            )
            if metadata is None:
                raise ValueError("active eight-view world metrics require symmetry metadata")
            matrix_rows = []
            for raw, converted in zip(rows, active_rows):
                item = dict(converted)
                for key in ("exact_global_T_W_from_O", "k16_T_W_from_O"):
                    item[key] = ast.literal_eval(raw[key])
                matrix_rows.append(item)
            pairwise = active_world_pairwise_matrices(
                matrix_rows, metadata, effective_group
            )
            frame_ids = [int(row["frame_id"]) for row in active_rows]
            for name, matrix in pairwise.items():
                with (run / f"{name}.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["frame_id", *frame_ids])
                    for frame, values in zip(frame_ids, matrix):
                        writer.writerow([frame, *values])
        multifragment_world = []
        multifragment_pairwise = {}
        if multifragment_gates_enabled:
            run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
            metadata = load_symmetry_metadata(run_manifest["symmetry_sidecar_path"])
            manifest_payload = json.loads(Path(run_manifest["train_manifest"]).read_text(encoding="utf-8"))
            matrix_rows = []
            for raw, converted in zip(rows, active_rows):
                item = dict(converted)
                for key in ("exact_global_T_W_from_O", "k16_T_W_from_O"):
                    item[key] = ast.literal_eval(raw[key])
                matrix_rows.append(item)
            for fragment_id in sorted({int(row["fragment_id"]) for row in active_rows}):
                selected = [row for row in matrix_rows if int(row["fragment_id"]) == fragment_id]
                manifest_row = next(row for row in manifest_payload["samples"] if int(row["fragment_id"]) == fragment_id)
                group = parse_rotation_group(manifest_row["effective_symmetry_group"])
                values = active_world_metrics(selected, metadata, group)
                multifragment_world.append({"fragment_id": fragment_id, **values})
                matrices = active_world_pairwise_matrices(selected, metadata, group)
                multifragment_pairwise[str(fragment_id)] = matrices
                frame_ids = [int(row["frame_id"]) for row in selected]
                for name, matrix in matrices.items():
                    path = best_eval / f"fragment_{fragment_id:04d}_{name}.csv"
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream); writer.writerow(["frame_id", *frame_ids])
                        for frame, values_row in zip(frame_ids, matrix):
                            writer.writerow([frame, *values_row])
            with (best_eval / "world_metrics_per_fragment.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(multifragment_world[0]))
                writer.writeheader(); writer.writerows(multifragment_world)
        report = {
            **WARNING_FLAGS,
            **gate,
            "best_epoch": best_epoch,
            "sample_count": len(active_rows),
            "active_path_only": True,
            "best_metric_name": "eval/active/exact_global/worst_sample_projection_score",
            "worst_sample_projection_score": worst_sample_projection_score(active_rows),
            "target_leakage_audit_path": audit_path,
            "target_leakage_verified": audit_verified,
            "target_leakage_detected": leakage,
            "nonfinite_detected": nonfinite,
            "active_world_metrics": active_world,
            **({} if ten_gates or multifragment_gates_enabled else {"inactive_namespaces": {
                "legacy_triangle": "inactive_ignored",
                "legacy_barycentric": "inactive_ignored",
                "legacy_pose_query": "inactive_ignored",
                "regions": "inactive_ignored",
                "ranking": "inactive_ignored",
            }}),
            "failure_diagnosis": (
                diagnose_multifragment_failure(active_rows)
                if multifragment_gates_enabled else diagnose_ten_view_scratch_failure(active_rows, best_epoch=best_epoch)
                if ten_gates else diagnose_eight_view_failure(active_rows, best_epoch=best_epoch)
                if dual_gates else diagnose_four_view_failure(active_rows)
            ),
        }
        if multifragment_gates_enabled:
            assert combined is not None and strict_gate is not None
            assert practical_gate is not None and pose_gate is not None
            if not audit_verified or leakage is not False or nonfinite:
                for selected_gate in (strict_gate, practical_gate, pose_gate):
                    selected_gate["stage_passed"] = False
                    selected_gate["next_stage_allowed"] = False
                    reason = "target_leakage_audit" if not audit_verified or leakage is not False else "active_path_nonfinite"
                    if reason not in selected_gate["failures"]:
                        selected_gate["failures"].append(reason)
            per_fragment = aggregate_metrics(active_rows, "fragment_id")
            per_frame = aggregate_metrics(active_rows, "frame_id")
            for path, values in (
                (best_eval / "per_fragment_metrics.csv", per_fragment),
                (best_eval / "per_frame_metrics.csv", per_frame),
            ):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                    writer.writeheader(); writer.writerows(values)
            report.update(
                strict_surface_gate=strict_gate,
                practical_surface_gate=practical_gate,
                pose_placement_gate=pose_gate,
                strict_stage_passed=strict_gate["stage_passed"],
                practical_stage_passed=practical_gate["stage_passed"],
                pose_placement_passed=pose_gate["stage_passed"],
                stage_passed=practical_gate["stage_passed"],
                next_stage_allowed=practical_gate["stage_passed"],
                transition_policy="practical_surface_gate",
                best_metric_name="eval/active/worst_sample_multifragment_score",
                worst_sample_multifragment_score=worst_multifragment_sample_score(active_rows),
                per_fragment_metrics=per_fragment,
                per_frame_metrics=per_frame,
                world_metrics_per_fragment=multifragment_world,
                world_pairwise_matrices_per_fragment=multifragment_pairwise,
            )
            _json(run / "strict_surface_gate.json", {**WARNING_FLAGS, **strict_gate})
            _json(run / "practical_surface_gate.json", {**WARNING_FLAGS, **practical_gate})
            _json(run / "pose_placement_gate.json", {**WARNING_FLAGS, **pose_gate})
        elif ten_gates:
            assert combined is not None and strict_gate is not None
            assert practical_gate is not None and pose_gate is not None
            if not audit_verified or leakage is not False or nonfinite:
                for selected_gate in (strict_gate, practical_gate, pose_gate):
                    selected_gate["stage_passed"] = False
                    selected_gate["next_stage_allowed"] = False
                    reason = "target_leakage_audit" if not audit_verified or leakage is not False else "active_path_nonfinite"
                    if reason not in selected_gate["failures"]:
                        selected_gate["failures"].append(reason)
            report.update(
                strict_surface_gate=strict_gate,
                practical_surface_gate=practical_gate,
                pose_placement_gate=pose_gate,
                strict_stage_passed=strict_gate["stage_passed"],
                practical_stage_passed=practical_gate["stage_passed"],
                pose_placement_passed=pose_gate["stage_passed"],
                stage_passed=practical_gate["stage_passed"],
                next_stage_allowed=practical_gate["stage_passed"],
                transition_policy="practical_surface_gate",
                world_pairwise_matrices=pairwise,
                best_metric_name="eval/active/worst_sample_score",
            )
            _json(run / "strict_surface_gate.json", {**WARNING_FLAGS, **strict_gate})
            _json(run / "practical_surface_gate.json", {**WARNING_FLAGS, **practical_gate})
            _json(run / "pose_placement_gate.json", {**WARNING_FLAGS, **pose_gate})
        elif dual_gates:
            assert combined is not None and strict_gate is not None and practical_gate is not None
            # Leakage/nonfinite are active checks in both gates.  Apply the
            # external audit result without allowing practical pass to rewrite
            # a strict failure.
            if not audit_verified or leakage is not False or nonfinite:
                for selected_gate in (strict_gate, practical_gate):
                    selected_gate["stage_passed"] = False
                    selected_gate["next_stage_allowed"] = False
                    reason = (
                        "target_leakage_audit"
                        if not audit_verified or leakage is not False
                        else "active_path_nonfinite"
                    )
                    if reason not in selected_gate["failures"]:
                        selected_gate["failures"].append(reason)
            report.update(
                strict_submillimetre_gate=strict_gate,
                practical_pose_first_gate=practical_gate,
                strict_stage_passed=strict_gate["stage_passed"],
                practical_stage_passed=practical_gate["stage_passed"],
                stage_passed=practical_gate["stage_passed"],
                next_stage_allowed=practical_gate["stage_passed"],
                transition_policy="practical_pose_first_gate",
                worst_sample_practical_score=worst_sample_practical_score(active_rows),
                world_pairwise_matrices=pairwise,
                best_metric_name="eval/active/worst_sample_practical_score",
            )
            _json(run / "strict_stage_gate.json", {**WARNING_FLAGS, **strict_gate})
            _json(run / "practical_stage_gate.json", {**WARNING_FLAGS, **practical_gate})
        _json(run / "stage_gate.json", report)
        if not report["stage_passed"]:
            _json(run / "diagnostic_failure.json", report)
        return report
    dependencies = resolved.get("stage_gate_dependencies", {})
    local_substage = dependencies.get("local_substage")
    fine_gate_config = resolved.get("fine_stage_gate")
    thresholds = {
        "rotation_error_deg": 2.0,
        "translation_total_mm": 2.0,
        "correspondence_p95_mm": 2.0,
        "visible_alignment_p95_mm": 2.0,
        "predicted_to_template_surface_p95_mm": 1.0,
    }
    thresholds.update(dict(dependencies.get("physical_thresholds", {})))
    patch_only_gate = bool(dependencies.get("patch_only_gate", False))
    if patch_only_gate:
        thresholds = {
            "candidate_stage": {
                "valid_patch_set_top4_recall": 0.995,
                "valid_patch_set_in_candidate_set_fraction": 0.995,
                "unique_predicted_patches": "> 1",
                "most_popular_patch_fraction": "< 0.8",
            },
            "top1_quality_diagnostic": {
                "valid_patch_set_top1_accuracy": 0.95,
            },
        }
    elif (local_substage and str(local_substage).upper() != "B4") or fine_gate_config:
        # B1/B2/B3 use their isolated local gate below.  In particular B1/B2
        # must not inherit pose/correspondence readiness thresholds.
        thresholds = {}
    failures: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    nonfinite = False
    for row in rows:
        for value in row.values():
            try:
                numeric = _number(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric):
                nonfinite = True
        for name, threshold in (
            () if patch_only_gate or local_substage or fine_gate_config else thresholds.items()
        ):
            try:
                value = _number(row[name])
            except (KeyError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                nonfinite = True
            ratio = value / threshold if math.isfinite(value) else math.inf
            if worst is None or ratio > worst["threshold_ratio"]:
                worst = {"sample_id": row.get("sample_id"), "component": name, "value": value, "threshold": threshold, "threshold_ratio": ratio}
            if not math.isfinite(value) or value > threshold:
                failures.append({"sample_id": row.get("sample_id"), "component": name, "value": value, "threshold": threshold})
        for name, expected in (
            ()
            if patch_only_gate or local_substage or fine_gate_config
            else (("procrustes_rank_valid", 1.0), ("effective_correspondence_fraction", 1.0))
        ):
            try:
                value = _number(row[name])
            except (KeyError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                nonfinite = True
            if not math.isfinite(value) or abs(value - expected) > 1e-6:
                failures.append({"sample_id": row.get("sample_id"), "component": name, "value": value, "expected": expected})
        if bool(dependencies.get("teacher_forced_local_gate", False)):
            for name, expected in (
                ("gt_triangle_in_local_candidates_fraction", 1.0),
                ("correspondence_rank", 3.0),
                ("procrustes_rank_valid", 1.0),
            ):
                value = _number(row.get(name, math.nan))
                if not math.isfinite(value) or abs(value - expected) > 1e-6:
                    failures.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "component": name,
                            "value": value,
                            "expected": expected,
                        }
                    )
        if bool(dependencies.get("require_no_correspondence_collapse", False)):
            rank = _number(row.get("correspondence_geometry_rank", math.nan))
            unique = _number(row.get("attention_unique_selected_anchors", math.nan))
            if not math.isfinite(rank) or rank < 3:
                failures.append({"sample_id": row.get("sample_id"), "component": "correspondence_geometry_rank", "value": rank, "minimum": 3})
            if not math.isfinite(unique) or unique < 3:
                failures.append({"sample_id": row.get("sample_id"), "component": "attention_unique_selected_anchors", "value": unique, "minimum": 3})
        if bool(dependencies.get("require_no_excessive_attention_diffusion", False)):
            entropy = _number(row.get("attention_normalized_entropy", math.nan))
            if not math.isfinite(entropy) or entropy > 0.8:
                failures.append({"sample_id": row.get("sample_id"), "component": "attention_normalized_entropy", "value": entropy, "maximum": 0.8})
        if bool(dependencies.get("require_patch_gate", False)) and not patch_only_gate:
            for name, minimum in (
                ("coarse_patch_top1_accuracy", 0.95),
                ("coarse_patch_top4_recall", 0.995),
                ("gt_patch_in_candidate_set_fraction", 0.995),
            ):
                value = _number(row.get(name, math.nan))
                if not math.isfinite(value) or value < minimum:
                    failures.append(
                        {
                            "sample_id": row.get("sample_id"),
                            "component": name,
                            "value": value,
                            "minimum": minimum,
                        }
                    )
    dependency_reports: dict[str, Any] = {}
    capacity_audit_passed = not bool(
        dependencies.get("require_parameterization_capacity", False)
    )
    for required_key, path_key, passed_key in (
        ("require_point_contract_audit", "point_contract_audit_path", "audit_passed"),
        ("require_parameterization_capacity", "parameterization_capacity_path", "free_capacity_passed"),
    ):
        if not bool(dependencies.get(required_key, False)):
            continue
        configured = dependencies.get(path_key)
        if required_key == "require_parameterization_capacity":
            passed_key = str(
                dependencies.get("parameterization_capacity_required_field", passed_key)
            )
        dependency_paths = configured if isinstance(configured, list) else [configured]
        report_rows = []
        for dependency_path in dependency_paths:
            dependency_payload = None
            if dependency_path and Path(str(dependency_path)).is_file():
                dependency_payload = json.loads(Path(str(dependency_path)).read_text(encoding="utf-8"))
            report_rows.append({"path": dependency_path, "payload": dependency_payload})
            dependency_passed = (
                dependency_payload is not None
                and bool(dependency_payload.get(passed_key, False))
            )
            if required_key == "require_parameterization_capacity":
                capacity_audit_passed = capacity_audit_passed or dependency_passed
            if not dependency_passed:
                failures.append({"component": required_key, "path": dependency_path, "required_field": passed_key, "value": None if dependency_payload is None else dependency_payload.get(passed_key)})
        dependency_reports[path_key] = report_rows
    if not audit_verified or leakage is not False:
        failures.append({"component": "target_leakage_detected", "value": leakage, "required": False, "audit_verified": audit_verified})
    candidate_gate = None
    top1_gate = None
    local_gate = None
    fine_gate = None
    if patch_only_gate:
        def aggregate(name: str, fallback: str | None, mode: str) -> float:
            values = []
            for row in rows:
                raw = row.get(name)
                if (raw is None or raw == "") and fallback is not None:
                    raw = row.get(fallback)
                values.append(_number(raw if raw is not None else math.nan))
            return (min(values) if mode == "min" else max(values)) if values else math.nan

        patch_metrics = {
            "single_owner_top1_accuracy": aggregate(
                "coarse_patch_top1_accuracy", None, "min"
            ),
            "valid_patch_set_top1_accuracy": aggregate(
                "valid_patch_set_top1_accuracy", "coarse_patch_top1_accuracy", "min"
            ),
            "valid_patch_set_top4_recall": aggregate(
                "valid_patch_set_top4_recall", "coarse_patch_top4_recall", "min"
            ),
            "valid_patch_set_in_candidate_set_fraction": aggregate(
                "valid_patch_set_in_candidate_set_fraction",
                "gt_patch_in_candidate_set_fraction",
                "min",
            ),
            "unique_predicted_patches": aggregate(
                "unique_predicted_patches", None, "min"
            ),
            "most_popular_patch_fraction": aggregate(
                "most_popular_patch_fraction", None, "max"
            ),
        }
        candidate_gate = candidate_stage_gate(
            patch_metrics,
            nonfinite_detected=nonfinite,
            target_leakage_detected=leakage,
            capacity_audit_passed=capacity_audit_passed,
        )
        if failures:
            candidate_gate["checks"]["required_dependencies_passed"] = False
            candidate_gate["failures"].append("required_dependencies_passed")
            candidate_gate["dependency_failures"] = failures
            candidate_gate["candidate_stage_passed"] = False
            candidate_gate["stage_passed"] = False
            candidate_gate["next_stage_allowed"] = False
            candidate_gate["authorizes_teacher_forced_local_stage_b"] = False
        top1_gate = top1_quality_gate(patch_metrics)
        _json(run / "candidate_stage_gate.json", candidate_gate)
        _json(run / "top1_quality_gate.json", top1_gate)
        passed = bool(candidate_gate["candidate_stage_passed"])
    elif fine_gate_config:
        def fine_aggregate(name: str, mode: str, fallback: float) -> float:
            values = []
            for row in rows:
                try:
                    values.append(_number(row.get(name, fallback)))
                except (TypeError, ValueError):
                    values.append(float(fallback))
            return (max(values) if mode == "max" else min(values)) if values else float(fallback)

        fine_metrics = {
            "aux_coordinate_p95_mm": fine_aggregate("aux_coordinate_p95_mm", "max", math.inf),
            "aux_coordinate_rmse_mm": fine_aggregate("aux_coordinate_rmse_mm", "max", math.inf),
            "fine_feature_variance": fine_aggregate("fine_feature_variance", "min", 0.0),
            "fine_feature_effective_rank": fine_aggregate("fine_feature_effective_rank", "min", 0.0),
            "fine_feature_pairwise_distance": fine_aggregate("fine_feature_pairwise_distance", "min", 0.0),
            "fine_feature_collision_fraction": fine_aggregate("fine_feature_collision_fraction", "max", 1.0),
            "fine_candidate_logit_variance": fine_aggregate("fine_candidate_logit_variance", "min", 0.0),
            "valid_triangle_set_top1_accuracy": fine_aggregate("valid_triangle_set_top1_accuracy", "min", -math.inf),
            "valid_triangle_set_top4_recall": fine_aggregate("valid_triangle_set_top4_recall", "min", -math.inf),
            "candidate_recall": fine_aggregate("candidate_recall", "min", -math.inf),
            "triangle_target_index_mismatch_fraction": fine_aggregate("triangle_target_index_mismatch_fraction", "max", math.inf),
            "correspondence_p95_mm": fine_aggregate("correspondence_p95_mm", "max", math.inf),
            "barycentric_reconstruction_p95_mm": fine_aggregate("barycentric_reconstruction_p95_mm", "max", math.inf),
            "correspondence_rank": fine_aggregate("correspondence_rank", "min", -math.inf),
            "procrustes_rank_valid": fine_aggregate("procrustes_rank_valid", "min", 0.0),
            "target_leakage_detected": bool(leakage is not False),
        }
        stage_name = str(resolved.get("stage", {}).get("name", ""))
        if stage_name.startswith("TWO_VIEW_"):
            per_sample = []
            for row in rows:
                checks = {
                    "projected_correspondence_p95_mm": _number(row.get("exact_global_projected_correspondence_p95_mm", math.inf)) <= 1.0 + 1e-6,
                    "alignment_p95_mm": _number(row.get("exact_global_projection_alignment_p95_mm", math.inf)) <= 1.0 + 1e-6,
                    "rotation_error_deg": _number(row.get("exact_global_projection_rotation_error_deg", math.inf)) <= 1.0 + 1e-6,
                    "translation_error_mm": _number(row.get("exact_global_projection_translation_error_mm", math.inf)) <= 1.0 + 1e-6,
                    "rank_three": _number(row.get("exact_global_projection_rank", -math.inf)) >= 3 - 1e-6,
                    "surface_membership_p95_mm": _number(row.get("exact_global_surface_membership_p95_mm", math.inf)) <= .1 + 1e-6,
                    "k16_exact_global_recall": _number(row.get("k16_exact_global_triangle_recall", -math.inf)) >= .995 - 1e-6,
                    "k16_zero_fallback": _number(row.get("k16_fallback_fraction", math.inf)) <= 1e-6,
                }
                per_sample.append({"sample_id": row.get("sample_id"), "frame_id": row.get("frame_id"), "checks": checks, "passed": all(checks.values())})
            all_passed = bool(per_sample) and all(item["passed"] for item in per_sample)
            fine_gate = {"passed": all_passed, "checks": {"each_sample_passed": all_passed}, "failures": [] if all_passed else ["one_or_more_samples_failed"], "per_sample": per_sample}
        elif stage_name.startswith("F1_"):
            fine_gate = fine_coordinate_gate(
                fine_metrics,
                minimum_feature_variance=float(
                    fine_gate_config.get("minimum_feature_variance", 1e-8)
                ),
            )
        elif stage_name.startswith("F2_"):
            fine_gate = fine_triangle_gate(fine_metrics)
        else:
            f3_checks = {
                "correspondence_p95": fine_metrics["correspondence_p95_mm"] <= .5 + 1e-6,
                "barycentric_p95": fine_metrics["barycentric_reconstruction_p95_mm"] <= .5 + 1e-6,
                "rank_three": fine_metrics["correspondence_rank"] >= 3.0 - 1e-6,
                "procrustes_valid": fine_metrics["procrustes_rank_valid"] >= 1.0 - 1e-6,
            }
            fine_gate = {
                "passed": all(f3_checks.values()) and not failures and not nonfinite,
                "checks": f3_checks,
            }
        gate_failures = list(fine_gate.get("failures", []))
        if failures:
            fine_gate["checks"]["required_dependencies_passed"] = False
            fine_gate["passed"] = False
            gate_failures.append("required_dependencies_passed")
        fine_gate.update(
            metrics=fine_metrics,
            failures=gate_failures,
            dependency_failures=failures,
        )
        passed = bool(fine_gate["passed"])
        _json(run / "coordinate_metrics.json", {"epoch": best_epoch, **fine_metrics})
        _json(
            run / "fine_feature_metrics.json",
            {
                "epoch": best_epoch,
                **{key: value for key, value in fine_metrics.items() if key.startswith("fine_")},
            },
        )
    elif local_substage:
        metric_names = (
            "valid_triangle_set_top1",
            "valid_triangle_set_top4",
            "valid_triangle_candidate_recall",
            "local_triangle_set_ce",
            "local_triangle_random_ce",
            "triangle_target_index_mismatch_fraction",
            "mean_valid_triangle_count",
            "fraction_with_multiple_valid_triangles",
            "mean_local_candidate_count",
            "min_local_candidate_count",
            "max_local_candidate_count",
            "invalid_candidate_count_fraction",
            "duplicate_local_candidate_fraction",
            "teacher_forcing_selected_symmetry_element",
            "barycentric_reconstruction_p95_mm",
            "correspondence_p95_mm",
            "predicted_to_template_surface_p95_mm",
            "visible_alignment_p95_mm",
            "rotation_error_deg",
            "translation_total_mm",
            "correspondence_rank",
            "procrustes_rank_valid",
        )
        integer_metrics = {
            "min_local_candidate_count",
            "max_local_candidate_count",
        }
        local_metrics = {}
        for name in metric_names:
            raw = rows[0].get(name, math.nan) if rows else math.nan
            if name in integer_metrics:
                try:
                    local_metrics[name] = int(str(raw))
                except (TypeError, ValueError):
                    local_metrics[name] = None
            else:
                local_metrics[name] = _number(raw)
        local_gate = check_local_substage(
            str(local_substage),
            local_metrics,
            nonfinite_detected=nonfinite,
            target_leakage_detected=leakage,
        )
        if failures:
            local_gate["checks"]["required_dependencies_passed"] = False
            local_gate["failures"].append("required_dependencies_passed")
            local_gate["dependency_failures"] = failures
            local_gate["stage_passed"] = False
            local_gate["next_stage_allowed"] = False
        passed = bool(local_gate["stage_passed"])
    else:
        passed = not failures and not nonfinite
    report = {
        **WARNING_FLAGS, "stage_passed": passed, "best_epoch": best_epoch,
        "sample_count": len(rows), "thresholds": thresholds,
        "target_leakage_audit_path": audit_path, "target_leakage_verified": audit_verified,
        "target_leakage_detected": leakage, "nonfinite_detected": nonfinite,
        "dependency_reports": dependency_reports,
        "worst_sample_component": worst, "failures": failures,
        "next_stage_allowed": passed,
    }
    if patch_only_gate:
        report.update(
            candidate_stage_passed=passed,
            top1_quality_passed=bool(top1_gate["top1_quality_passed"]),
            candidate_stage_gate_path=str(run / "candidate_stage_gate.json"),
            top1_quality_gate_path=str(run / "top1_quality_gate.json"),
            top1_quality_gate_blocks_stage_b=False,
            failures=candidate_gate["failures"],
        )
    elif local_gate is not None:
        report.update(local_gate)
        report["best_epoch"] = best_epoch
        report["sample_count"] = len(rows)
        report["target_leakage_audit_path"] = audit_path
        report["dependency_reports"] = dependency_reports
        report["nonfinite_detected"] = nonfinite
        substage = str(local_substage).upper()
        if substage == "B1":
            _json(
                run / "triangle_target_contract.json",
                {
                    "triangle_target_index_mismatch_fraction": local_gate["metrics"].get(
                        "triangle_target_index_mismatch_fraction"
                    ),
                    "valid_triangle_candidate_recall": local_gate["metrics"].get(
                        "valid_triangle_candidate_recall"
                    ),
                    "duplicate_local_candidate_fraction": local_gate["metrics"].get(
                        "duplicate_local_candidate_fraction"
                    ),
                    "min_local_candidate_count": local_gate["metrics"].get(
                        "min_local_candidate_count"
                    ),
                    "max_local_candidate_count": local_gate["metrics"].get(
                        "max_local_candidate_count"
                    ),
                    "invalid_candidate_count_fraction": local_gate["metrics"].get(
                        "invalid_candidate_count_fraction"
                    ),
                    "teacher_forcing_selected_symmetry_element": local_gate["metrics"].get(
                        "teacher_forcing_selected_symmetry_element"
                    ),
                },
            )
            _json(
                run / "triangle_ambiguity.json",
                {
                    "mean_valid_triangle_count": local_gate["metrics"].get(
                        "mean_valid_triangle_count"
                    ),
                    "fraction_with_multiple_valid_triangles": local_gate["metrics"].get(
                        "fraction_with_multiple_valid_triangles"
                    ),
                    "triangle_target_tolerance_m": dependencies.get(
                        "local_triangle_target_tolerance_m", 0.00015
                    ),
                },
            )
            _json(
                run / "triangle_classifier_metrics.json",
                {"epoch": best_epoch, **local_gate["metrics"]},
            )
            _json(
                run / "random_baseline.json",
                {
                    "local_triangle_set_ce": local_gate["metrics"].get(
                        "local_triangle_set_ce"
                    ),
                    "random_cross_entropy": local_gate["metrics"].get(
                        "local_triangle_random_ce"
                    ),
                    "warning": (
                        "local_triangle_classifier_worse_than_uniform"
                        if not local_gate["checks"].get("loss_below_random", False)
                        else None
                    ),
                },
            )
        elif substage == "B2":
            _json(
                run / "barycentric_metrics.json",
                {"epoch": best_epoch, **local_gate["metrics"]},
            )
            _json(
                run / "canonical_coordinate_metrics.json",
                {"epoch": best_epoch, **local_gate["metrics"]},
            )
    elif fine_gate is not None:
        report["thresholds"] = fine_gate.get("thresholds", fine_gate_config)
        report.update(
            fine_stage_gate=fine_gate,
            stage_passed=bool(fine_gate["passed"]),
            next_stage_allowed=bool(fine_gate["passed"]),
            failures=fine_gate["failures"],
        )
    _json(run / "stage_gate.json", report)
    if not passed:
        _json(run / "diagnostic_failure.json", report)
        plateau = {
            **WARNING_FLAGS, "best_epoch": best_epoch,
            "diagnosis": (
                "candidate-set readiness gate failed; inspect candidate_stage_gate.json"
                if patch_only_gate
                else "physical stage gate failed; inspect the worst sample/component and best-evaluation files"
            ),
            "worst_sample_component": worst,
            "files_for_analysis": [path for path in [
                str(run / "best_evaluation" / "evaluation_summary.json"),
                str(run / "best_evaluation" / "per_sample_metrics.csv"),
                str(run / "stage_gate.json"),
                str(run / "candidate_stage_gate.json") if patch_only_gate else None,
                str(run / "top1_quality_gate.json") if patch_only_gate else None,
                str(run / "history" / "history.jsonl"),
            ] if path is not None],
        }
        _json(run / "plateau_analysis.json", plateau)
    return report


__all__ = ["check_joint_stage", "materialize_best_evaluation"]
