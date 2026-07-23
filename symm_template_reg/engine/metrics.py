"""Symmetry-aware accepted-sample registration metrics."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from symm_template_reg.models.pose.metrics import (
    rotation_error_deg,
    symmetry_aware_pose_errors,
)
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.losses.pose_query_ranking_loss import PoseQueryRankingLoss
from symm_template_reg.models.losses.correspondence_confidence_loss import (
    correspondence_confidence_diagnostics,
)
from symm_template_reg.models.losses.joint_correspondence_pose_loss import (
    JointCorrespondencePoseLoss,
)
from symm_template_reg.models.losses.joint_surface_correspondence_pose_loss_v3 import (
    JointSurfaceCorrespondencePoseLossV3,
)
from symm_template_reg.models.losses.clean_coordinate_pose_loss_v3 import (
    CleanCoordinatePoseLossV3,
)
from symm_template_reg.models.losses.symmetry_aware_correspondence_loss import (
    SymmetryAwareCorrespondenceLoss,
)
from symm_template_reg.evaluation.correspondence_diagnostics import (
    attention_distribution_metrics,
    covariance_geometry,
    local_rigidity_errors,
)
from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.geometry.patch_targets import (
    single_owner_patch_ids,
    valid_patch_mask,
    valid_set_topk_hits,
)
from symm_template_reg.models.geometry.triangle_targets import (
    local_valid_triangle_mask,
    triangle_target_sets,
)
from symm_template_reg.engine.single_fragment import ranking_diagnostics
from symm_template_reg.models.symmetry.groups import (
    SO2Group,
    group_to_dict,
    parse_rotation_group,
)
from symm_template_reg.models.symmetry.region_assignment import (
    effective_group_from_regions,
)
from symm_template_reg.models.symmetry.pose_conditioned_resolver import (
    PoseConditionedSymmetryResolver,
)


def _duplicate_fraction(poses: Tensor) -> float:
    if len(poses) < 2:
        return 0.0
    translation = torch.cdist(poses[:, :3, 3], poses[:, :3, 3])
    rotation = poses[:, :3, :3]
    relative = rotation[:, None].transpose(-1, -2) @ rotation[None]
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1, 1)
    angle = torch.acos(cosine)
    upper = torch.triu(torch.ones_like(translation, dtype=torch.bool), diagonal=1)
    duplicates = upper & (translation < 1e-4) & (angle < torch.deg2rad(angle.new_tensor(0.1)))
    pairs = int(upper.sum())
    return float(duplicates.sum()) / max(pairs, 1)


def score_pose_cost_spearman(scores: Tensor, costs: Tensor) -> float:
    """Spearman(pose_logit, -pose_cost); positive correlation is good."""

    if scores.numel() < 2:
        return 1.0
    score_ranks = torch.argsort(torch.argsort(scores)).to(torch.float32)
    quality_ranks = torch.argsort(torch.argsort(-costs)).to(torch.float32)
    score_ranks = score_ranks - score_ranks.mean()
    quality_ranks = quality_ranks - quality_ranks.mean()
    denominator = torch.linalg.vector_norm(score_ranks) * torch.linalg.vector_norm(
        quality_ranks
    )
    if float(denominator) <= 0.0:
        return 0.0
    return float(torch.dot(score_ranks, quality_ranks) / denominator)


def physical_normalized_score(
    rotation_error_deg: float,
    translation_total_mm: float,
    correspondence_p95_mm: float,
    visible_alignment_p95_mm: float,
    predicted_to_template_surface_p95_mm: float,
) -> float:
    return (
        rotation_error_deg / 2.0
        + translation_total_mm / 2.0
        + correspondence_p95_mm / 2.0
        + visible_alignment_p95_mm / 2.0
        + predicted_to_template_surface_p95_mm
    )


def _binary_region_metrics(
    predicted: Tensor, expected: Tensor, prefix: str
) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    f1_values = []
    for region_index in range(len(expected)):
        p = predicted[region_index : region_index + 1]
        t = expected[region_index : region_index + 1]
        tp = float((p & t).sum())
        fp = float((p & ~t).sum())
        fn = float((~p & t).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result[f"{prefix}_region_{region_index}_precision"] = precision
        result[f"{prefix}_region_{region_index}_recall"] = recall
        result[f"{prefix}_region_{region_index}_f1"] = f1
        f1_values.append(f1)
    result[f"{prefix}_macro_f1"] = sum(f1_values) / max(len(f1_values), 1)
    result[f"{prefix}_exact_match"] = bool(torch.equal(predicted, expected))
    return result


def batch_pose_metric_rows(
    prediction: Any,
    batch: Mapping[str, Any],
    *,
    translation_cost_weight: float = 10.0,
    rotation_cost_weight: float = 1.0,
    ranking_config: Mapping[str, Any] | None = None,
    joint_loss_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt = batch["gt"]["T_C_from_O"]
    metadata_list = batch["template_symmetry_metadata"]
    groups = batch["gt"]["effective_symmetry_group"]
    correspondence_pose = getattr(prediction, "correspondence_pose", None)
    joint_diagnostics = None
    if joint_loss_config is not None and bool(
        joint_loss_config.get("enabled", False)
    ):
        target_payload = batch["gt"]["points_O_corresponding"]
        target_points = (
            target_payload.to_padded()["points"]
            if hasattr(target_payload, "to_padded")
            else target_payload
        )
        template_payload = batch["template"]
        if hasattr(template_payload, "to_padded"):
            template_padded = template_payload.to_padded()
            template_surface_points = template_padded["points"]
            template_surface_mask = template_padded["valid_mask"]
        else:
            template_surface_points = template_payload.get(
                "points_O", template_payload.get("points")
            )
            template_surface_mask = template_payload["valid_mask"]
        joint_cfg = dict(joint_loss_config)
        joint_cfg.pop("enabled", None)
        clean_active_only = bool(joint_cfg.pop("clean_active_only", False))
        runtime_epoch = int(joint_cfg.pop("_runtime_epoch", 0))
        common_arguments = dict(
            predicted_points_O=prediction.correspondence_points_O,
            predicted_pose_T_C_from_O=prediction.correspondence_pose,
            gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
            observed_points_C=(
                batch["observed"].to_padded()["points"]
                if hasattr(batch["observed"], "to_padded")
                else batch["observed"].get("points_C", batch["observed"].get("points"))
            ),
            target_points_O=target_points,
            valid_mask=prediction.observed_valid_mask,
            template_surface_points_O=template_surface_points,
            template_valid_mask=template_surface_mask,
            symmetry_metadata=metadata_list,
            effective_symmetry_groups=groups,
        )
        if clean_active_only:
            if prediction.correspondence_auxiliary is None:
                raise ValueError("clean V3 evaluation requires normalized q_aux")
            joint_diagnostics = CleanCoordinatePoseLossV3(
                current_epoch=runtime_epoch, **joint_cfg
            )(
                predicted_normalized_O=prediction.correspondence_auxiliary[
                    "fine_aux_coordinate_normalized"
                ],
                observed_points_C=common_arguments["observed_points_C"],
                target_points_O=target_points,
                valid_mask=prediction.observed_valid_mask,
                gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
                symmetry_metadata=metadata_list,
                effective_symmetry_groups=groups,
                template_mesh_vertices_O=batch["template_mesh_vertices_O"],
            )
        elif "lambda_corr_mean" in joint_cfg:
            if prediction.correspondence_auxiliary is None:
                raise ValueError("V3 evaluation requires correspondence auxiliary outputs")
            pose_diagnostics = prediction.correspondence_pose_diagnostics or {}
            joint_diagnostics = JointSurfaceCorrespondencePoseLossV3(**joint_cfg)(
                **common_arguments,
                correspondence_auxiliary=prediction.correspondence_auxiliary,
                template_mesh_vertices_O=batch["template_mesh_vertices_O"],
                template_mesh_faces=batch["template_mesh_faces"],
                pose_rank_valid=pose_diagnostics.get("rank_valid"),
            )
        else:
            joint_diagnostics = JointCorrespondencePoseLoss(**joint_cfg)(
                **common_arguments
            )
    observed_payload = batch.get("observed")
    if observed_payload is not None and hasattr(observed_payload, "to_padded"):
        observed_dense = observed_payload.to_padded()
        observed_points_C = observed_dense["points"]
        observed_valid_mask = observed_dense["valid_mask"]
    elif observed_payload is not None:
        observed_points_C = observed_payload.get(
            "points_C", observed_payload.get("points")
        )
        observed_valid_mask = observed_payload["valid_mask"]
    else:
        observed_points_C = None
        observed_valid_mask = None
    activity_configs = [dict(item.get("symmetry_region_activity", {})) for item in batch["meta"]]
    if (
        observed_points_C is not None
        and activity_configs
        and all(value == activity_configs[0] for value in activity_configs)
    ):
        pose_conditioned = PoseConditionedSymmetryResolver().resolve(
            observed_points_C,
            observed_valid_mask,
            prediction.pose_hypotheses,
            metadata_list,
            activity_configs[0],
        )
    else:
        pose_conditioned = None
    for index in range(len(gt)):
        poses = prediction.pose_hypotheses[index]
        top_index = int(torch.argmax(prediction.pose_logits[index]))
        errors = symmetry_aware_pose_errors(
            poses,
            gt[index],
            metadata_list[index],
            effective_group=groups[index],
        )
        pose_costs = (
            float(translation_cost_weight) * errors["translation_m"]
            + float(rotation_cost_weight) * errors["rotation_rad"]
        )
        oracle_index = int(torch.argmin(pose_costs))
        top_rotation_deg = float(errors["rotation_deg"][top_index])
        top_translation_m = float(errors["translation_m"][top_index])
        oracle_rotation_deg = float(errors["rotation_deg"][oracle_index])
        oracle_translation_m = float(errors["translation_m"][oracle_index])
        ranking_type = str((ranking_config or {}).get("type", "matched_categorical"))
        ranking_loss = PoseQueryRankingLoss(
            type=ranking_type,
            temperature=float((ranking_config or {}).get("temperature", 0.25)),
            cost_normalization=str(
                (ranking_config or {}).get("cost_normalization", "minmax")
            ),
            detach_pose_cost=True,
        )
        if ranking_type == "soft_quality":
            target_distribution = ranking_loss.target_distribution(
                pose_costs.unsqueeze(0)
            )[0]
        else:
            target_distribution = torch.nn.functional.one_hot(
                torch.argmin(pose_costs), num_classes=len(pose_costs)
            ).to(pose_costs.dtype)
        ranking_stats = ranking_diagnostics(
            prediction.pose_logits[index].unsqueeze(0),
            pose_costs.unsqueeze(0),
            target_distribution.unsqueeze(0),
        )
        axis_O = torch.as_tensor(
            metadata_list[index].axis.direction,
            dtype=poses.dtype,
            device=poses.device,
        )
        axis_C = gt[index, :3, :3] @ axis_O
        translation_delta = poses[top_index, :3, 3] - gt[index, :3, 3]
        along = torch.abs(torch.dot(translation_delta, axis_C))
        perpendicular = torch.linalg.vector_norm(
            translation_delta - torch.dot(translation_delta, axis_C) * axis_C
        )
        fragment = batch["meta"][index]["fragment_mesh"]
        frame_ids = batch.get("frame_id")
        frame_id = int(frame_ids[index]) if frame_ids is not None else -1
        row = {
            "sample_id": batch["sample_id"][index],
            "scene_id": batch["scene_id"][index],
            "fragment_id": int(batch["fragment_id"][index]),
            # Older synthetic unit batches predate frame_id.  Keep their
            # metric contract usable while real dataset rows remain explicit.
            "frame_id": frame_id,
            "top1_query_index": top_index,
            "oracle_query_index": oracle_index,
            "query_pose_costs": pose_costs.detach().cpu().tolist(),
            "query_rotation_error_deg": errors["rotation_deg"].detach().cpu().tolist(),
            "query_translation_error_mm": (
                errors["translation_m"] * 1000.0
            ).detach().cpu().tolist(),
            "query_T_C_from_O": poses.detach().cpu().tolist(),
            "query_classification_accuracy": top_index == oracle_index,
            "top1_query_is_oracle": top_index == oracle_index,
            "oracle_best_pose_cost": float(pose_costs[oracle_index]),
            "top1_scored_pose_cost": float(pose_costs[top_index]),
            "ranking_regret": float(pose_costs[top_index] - pose_costs[oracle_index]),
            "score_pose_cost_spearman": score_pose_cost_spearman(
                prediction.pose_logits[index], pose_costs
            ),
            "score_vs_negative_pose_cost_spearman": score_pose_cost_spearman(
                prediction.pose_logits[index], pose_costs
            ),
            "ranking_target_distribution": target_distribution.detach().cpu().tolist(),
            **{
                key: float(value[0]) for key, value in ranking_stats.items()
            },
            "query_oracle_probability": float(
                torch.softmax(prediction.pose_logits[index], dim=-1)[oracle_index]
            ),
            "top1_rotation_error_deg": top_rotation_deg,
            "oracle_topk_rotation_error_deg": oracle_rotation_deg,
            "translation_total_mm": top_translation_m * 1000.0,
            "oracle_translation_total_mm": oracle_translation_m * 1000.0,
            "translation_along_axis_mm": float(along) * 1000.0,
            "translation_perpendicular_axis_mm": float(perpendicular) * 1000.0,
            "success_2deg_2mm": top_rotation_deg <= 2.0 and top_translation_m <= 0.002,
            "success_5deg_5mm": top_rotation_deg <= 5.0 and top_translation_m <= 0.005,
            "success_10deg_10mm": top_rotation_deg <= 10.0 and top_translation_m <= 0.010,
            "oracle_topk_success_5deg_5mm": (
                oracle_rotation_deg <= 5.0 and oracle_translation_m <= 0.005
            ),
            "oracle_topk_success_2deg_2mm": (
                oracle_rotation_deg <= 2.0 and oracle_translation_m <= 0.002
            ),
            "oracle_topk_success_10deg_10mm": (
                oracle_rotation_deg <= 10.0 and oracle_translation_m <= 0.010
            ),
            "duplicate_pose_query_fraction": _duplicate_fraction(poses),
            "fragment_num_faces": fragment["num_faces"],
            "fragment_surface_area_m2": fragment["surface_area_m2"],
            "fragment_bbox_diagonal_m": fragment["bbox_diagonal_m"],
            "gt_T_C_from_O": gt[index].detach().cpu().tolist(),
        }
        correspondence_points = getattr(prediction, "correspondence_points_O", None)
        observed_valid_mask = getattr(prediction, "observed_valid_mask", None)
        if correspondence_points is not None and observed_valid_mask is not None:
            valid_correspondence = correspondence_points[
                index, observed_valid_mask[index]
            ]
            row["correspondence_prediction_summary"] = torch.cat(
                (
                    valid_correspondence.mean(dim=0),
                    valid_correspondence.std(dim=0, unbiased=False),
                )
            ).detach().cpu().tolist()
            logits = getattr(prediction, "correspondence_logits", None)
            if isinstance(logits, Tensor):
                sharpness = attention_distribution_metrics(logits[index, observed_valid_mask[index]])
                row.update(
                    attention_entropy=float(sharpness["entropy"].mean()),
                    attention_normalized_entropy=float(sharpness["normalized_entropy"].mean()),
                    attention_top1_mass=float(sharpness["top1_mass"].mean()),
                    attention_top5_mass=float(sharpness["top5_mass"].mean()),
                    attention_effective_candidate_count=float(sharpness["effective_candidate_count"].mean()),
                    attention_unique_selected_anchors=int(sharpness["unique_argmax_anchors"]),
                    attention_anchor_collision_ratio=float(sharpness["collision_ratio"]),
                    attention_most_popular_anchor_fraction=float(sharpness["most_popular_anchor_fraction"]),
                )
            geometry = covariance_geometry(valid_correspondence)
            eigenvalues = geometry["covariance_eigenvalues"]
            row.update(
                correspondence_geometry_rank=int(geometry["rank"]),
                correspondence_covariance_min_max_ratio=float(
                    eigenvalues.min() / eigenvalues.max().clamp_min(1e-12)
                ),
            )
            if observed_points_C is not None:
                rigidity = local_rigidity_errors(
                    valid_correspondence,
                    observed_points_C[index, observed_valid_mask[index]],
                    8,
                )
                row["local_rigidity_p95_mm"] = float(torch.quantile(rigidity, 0.95) * 1000.0)
        base_pose = getattr(prediction, "base_pose", None)
        if base_pose is not None:
            base_errors = symmetry_aware_pose_errors(
                base_pose[index].unsqueeze(0),
                gt[index],
                metadata_list[index],
                effective_group=groups[index],
            )
            row.update(
                {
                    "base_T_C_from_O": base_pose[index]
                    .detach()
                    .cpu()
                    .tolist(),
                    "base_pose_rotation_error_deg": float(
                        base_errors["rotation_deg"][0]
                    ),
                    "base_pose_translation_error_mm": float(
                        base_errors["translation_m"][0] * 1000.0
                    ),
                }
            )
            context_diagnostics = getattr(prediction, "context_diagnostics", None)
            if context_diagnostics is not None:
                for name in (
                    "observed_context",
                    "template_context",
                    "sample_context",
                    "rotation_context",
                    "translation_context",
                ):
                    value = context_diagnostics.get(name)
                    if isinstance(value, Tensor):
                        row[name] = value[index].detach().cpu().tolist()
                for name in (
                    "observed_context_norm",
                    "template_context_norm",
                    "sample_context_norm",
                    "rotation_context_norm",
                    "translation_context_norm",
                ):
                    value = context_diagnostics.get(name)
                    if isinstance(value, Tensor):
                        row[name] = float(value[index])
            residual_transforms = getattr(prediction, "residual_transforms", None)
            if isinstance(residual_transforms, Tensor):
                row["residual_T_camera"] = (
                    residual_transforms[index].detach().cpu().tolist()
                )
            correction = getattr(prediction, "base_correction_transform", None)
            if isinstance(correction, Tensor):
                correction_rotation = rotation_error_deg(
                    correction[index],
                    torch.eye(4, dtype=correction.dtype, device=correction.device),
                )
                row["hybrid_residual_rotation_deg"] = float(correction_rotation)
                row["hybrid_residual_translation_mm"] = float(
                    torch.linalg.vector_norm(correction[index, :3, 3]) * 1000.0
                )
                row["hybrid_correction_T"] = correction[index].detach().cpu().tolist()
        T_W_from_C = batch["gt"].get("T_W_from_C")
        if isinstance(T_W_from_C, Tensor):
            query_world = T_W_from_C[index].unsqueeze(0) @ poses
            top_world = T_W_from_C[index] @ poses[top_index]
            oracle_world = T_W_from_C[index] @ poses[oracle_index]
            gt_world = T_W_from_C[index] @ gt[index]
            row.update(
                {
                    "top1_T_W_from_O": top_world.detach().cpu().tolist(),
                    "oracle_T_W_from_O": oracle_world.detach().cpu().tolist(),
                    "gt_T_W_from_O": gt_world.detach().cpu().tolist(),
                    "query_T_W_from_O": query_world.detach().cpu().tolist(),
                    "symmetry_axis_O": list(metadata_list[index].axis.direction),
                    "effective_symmetry_group": groups[index],
                }
            )
            if correspondence_pose is not None:
                row["correspondence_T_W_from_O"] = (
                    T_W_from_C[index] @ correspondence_pose[index]
                ).detach().cpu().tolist()
        if pose_conditioned is not None:
            resolved_group = pose_conditioned.effective_group_per_pose[index][top_index]
            gt_group = parse_rotation_group(groups[index])
            resolved_payload = (
                group_to_dict(resolved_group) if resolved_group is not None else None
            )
            row["pose_conditioned_effective_group_accuracy"] = (
                resolved_payload == group_to_dict(gt_group)
            )
            resolved_count = (
                1
                if resolved_group is None
                else -1
                if isinstance(resolved_group, SO2Group)
                else resolved_group.order
            )
            gt_count = -1 if isinstance(gt_group, SO2Group) else gt_group.order
            row["pose_conditioned_hypothesis_count_accuracy"] = (
                resolved_count == gt_count
            )
            row["pose_conditioned_out_of_bounds_fraction"] = (
                pose_conditioned.out_of_sidecar_bounds_fraction[index][top_index]
            )
            row["pose_conditioned_group_unresolved"] = (
                pose_conditioned.unresolved_flags[index][top_index]
            )
        active_target = batch["gt"].get("active_symmetry_regions")
        active_valid = batch["gt"].get("active_symmetry_regions_valid_mask")
        if (
            prediction.active_region_logits is not None
            and active_target is not None
            and active_valid is not None
        ):
            valid = active_valid[index]
            width = active_target.shape[-1]
            valid = valid[:width]
            if bool(valid.any()):
                predicted_active = prediction.active_region_logits[
                    index, :width
                ].gt(0)[valid]
                expected_active = active_target[index, :width].bool()[valid]
                true_positive = (predicted_active & expected_active).sum().float()
                precision = true_positive / predicted_active.sum().clamp_min(1)
                recall = true_positive / expected_active.sum().clamp_min(1)
                row.update(
                    {
                        "active_region_accuracy": float(
                            (predicted_active == expected_active).float().mean()
                        ),
                        "active_region_precision": float(precision),
                        "active_region_recall": float(recall),
                        "active_region_exact_match": bool(
                            torch.equal(predicted_active, expected_active)
                        ),
                    }
                )
                row.update(
                    _binary_region_metrics(
                        predicted_active, expected_active, "active_region"
                    )
                )
                predicted_full = torch.zeros(
                    width, dtype=torch.bool, device=predicted_active.device
                )
                predicted_full[valid] = predicted_active
                predicted_group = effective_group_from_regions(
                    metadata_list[index], predicted_full
                )
                expected_group = parse_rotation_group(groups[index])
                row["learned_effective_group_accuracy"] = (
                    group_to_dict(predicted_group) == group_to_dict(expected_group)
                )
                predicted_count = -1 if isinstance(predicted_group, SO2Group) else predicted_group.order
                expected_count = -1 if isinstance(expected_group, SO2Group) else expected_group.order
                row["learned_hypothesis_count_accuracy"] = (
                    predicted_count == expected_count
                )
        point_target = batch["gt"].get("point_symmetry_region_indices")
        point_valid = batch["gt"].get("point_symmetry_region_valid_mask")
        if (
            prediction.observed_region_logits is not None
            and point_target is not None
            and point_valid is not None
            and active_valid is not None
        ):
            region_width = int(active_valid[index].sum())
            logits = prediction.observed_region_logits[index].clone()
            logits[:, region_width:] = -torch.inf
            usable = (
                point_valid[index]
                & prediction.observed_valid_mask[index]
                & point_target[index].ge(0)
                & point_target[index].lt(region_width)
            )
            row["observed_region_ignored_point_count"] = int(
                prediction.observed_valid_mask[index].sum() - usable.sum()
            )
            if bool(usable.any()):
                predicted_ids = logits.argmax(dim=-1)[usable]
                expected_ids = point_target[index][usable]
                row["observed_region_point_accuracy"] = float(
                    predicted_ids.eq(expected_ids).float().mean()
                )
                f1_values = []
                for true_region in range(region_width):
                    target_is_region = expected_ids.eq(true_region)
                    predicted_is_region = predicted_ids.eq(true_region)
                    tp = float((target_is_region & predicted_is_region).sum())
                    fp = float((~target_is_region & predicted_is_region).sum())
                    fn = float((target_is_region & ~predicted_is_region).sum())
                    precision = tp / max(tp + fp, 1.0)
                    recall = tp / max(tp + fn, 1.0)
                    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
                    row[f"observed_region_{true_region}_precision"] = precision
                    row[f"observed_region_{true_region}_recall"] = recall
                    row[f"observed_region_{true_region}_f1"] = f1
                    f1_values.append(f1)
                    for predicted_region in range(region_width):
                        row[
                            f"observed_region_confusion_true_{true_region}_pred_{predicted_region}"
                        ] = int(
                            (
                                target_is_region
                                & predicted_ids.eq(predicted_region)
                            ).sum()
                        )
                row["observed_region_macro_f1"] = sum(f1_values) / max(
                    len(f1_values), 1
                )
        if batch["gt"].get("points_O_corresponding") is not None:
            target = batch["gt"]["points_O_corresponding"]
            if hasattr(target, "to_padded"):
                target = target.to_padded()["points"]
            correspondence_match = SymmetryAwareCorrespondenceLoss().forward_with_diagnostics(
                prediction.correspondence_points_O[index : index + 1],
                target[index : index + 1],
                prediction.observed_valid_mask[index : index + 1],
                [metadata_list[index]],
                [groups[index]],
                prediction.correspondence_confidence[index : index + 1],
            )
            matched_target = correspondence_match["matched_target_points_O"][0]
            row["selected_shared_symmetry_element"] = int(
                correspondence_match["selected_shared_symmetry_element"][0]
            )
            mask = prediction.observed_valid_mask[index]
            distance = torch.linalg.vector_norm(
                prediction.correspondence_points_O[index, mask] - matched_target[mask],
                dim=-1,
            )
            point_rmse = float(
                torch.sqrt(torch.mean(distance * distance)) * 1000.0
            )
            row["correspondence_rmse_mm"] = point_rmse
            row["correspondence_point_rmse_mm"] = point_rmse
            row["correspondence_point_p95_mm"] = float(
                torch.quantile(distance, 0.95) * 1000.0
            )
            confidence_values = correspondence_confidence_diagnostics(
                prediction.correspondence_confidence[index : index + 1],
                prediction.observed_valid_mask[index : index + 1],
            )
            row["confidence_entropy"] = float(confidence_values["entropy"][0])
            row["effective_correspondence_count"] = float(
                confidence_values["effective_count"][0]
            )
            row["maximum_normalized_correspondence_weight"] = float(
                confidence_values["max_normalized_weight"][0]
            )
        if correspondence_pose is not None:
            corr_pose = correspondence_pose[index]
            corr_errors = symmetry_aware_pose_errors(
                corr_pose.unsqueeze(0), gt[index], metadata_list[index],
                effective_group=groups[index],
            )
            corr_rotation = float(corr_errors["rotation_deg"][0])
            corr_translation = float(corr_errors["translation_m"][0] * 1000.0)
            row.update(
                {
                    "correspondence_T_C_from_O": corr_pose.detach().cpu().tolist(),
                    "correspondence_pose_rotation_error_deg": corr_rotation,
                    "correspondence_pose_translation_error_mm": corr_translation,
                    "correspondence_pose_success_2deg_2mm": corr_rotation <= 2.0 and corr_translation <= 2.0,
                    "correspondence_pose_success_5deg_5mm": corr_rotation <= 5.0 and corr_translation <= 5.0,
                }
            )
            if prediction.base_pose is not None:
                row["direct_vs_correspondence_rotation_deg"] = float(
                    rotation_error_deg(prediction.base_pose[index], corr_pose)
                )
                row["direct_vs_correspondence_translation_mm"] = float(
                    torch.linalg.vector_norm(
                        prediction.base_pose[index, :3, 3] - corr_pose[:3, 3]
                    ) * 1000.0
                )
            pose_diagnostics = getattr(
                prediction, "correspondence_pose_diagnostics", None
            )
            if pose_diagnostics is not None:
                row["correspondence_pose_source_rank"] = int(
                    pose_diagnostics["source_rank"][index]
                )
                row["correspondence_pose_target_rank"] = int(
                    pose_diagnostics["target_rank"][index]
                )
                row["correspondence_pose_rank_valid"] = bool(
                    pose_diagnostics["rank_valid"][index]
                )
                row["correspondence_pose_valid_solution"] = bool(
                    pose_diagnostics["valid_solution"][index]
                )
        if joint_diagnostics is not None:
            mask = prediction.observed_valid_mask[index]
            selected = int(
                joint_diagnostics["selected_shared_symmetry_element"][index]
            )
            matched_target = joint_diagnostics["matched_target_points_O"][index]
            matched_pose = joint_diagnostics["matched_gt_pose_T_C_from_O"][index]
            correspondence_distance = torch.linalg.vector_norm(
                prediction.correspondence_points_O[index, mask]
                - matched_target[mask],
                dim=-1,
            )
            vertices = batch["template_mesh_vertices_O"][index].to(
                prediction.correspondence_points_O
            )
            faces = batch["template_mesh_faces"][index].to(
                device=vertices.device, dtype=torch.long
            )
            surface_distance = closest_points_on_triangle_mesh(
                prediction.correspondence_points_O[index, mask], vertices, faces
            )["distances"]
            visible_patch_to_predicted = torch.cdist(
                matched_target[mask].unsqueeze(0).float(),
                prediction.correspondence_points_O[index, mask].unsqueeze(0).float(),
            )[0].amin(-1)
            predicted_to_visible_patch = torch.cdist(
                prediction.correspondence_points_O[index, mask].unsqueeze(0).float(),
                matched_target[mask].unsqueeze(0).float(),
            )[0].amin(-1)
            alignment_distance = joint_diagnostics["alignment_distance_m"][
                index, mask
            ]
            rotation_error = float(
                torch.rad2deg(
                    rotation_geodesic_distance(
                        prediction.correspondence_pose[index, :3, :3],
                        matched_pose[:3, :3],
                    )
                )
            )
            translation_delta = (
                prediction.correspondence_pose[index, :3, 3]
                - matched_pose[:3, 3]
            )
            translation_mm = float(
                torch.linalg.vector_norm(translation_delta) * 1000.0
            )
            axis_O = torch.as_tensor(
                metadata_list[index].axis.direction,
                dtype=matched_pose.dtype,
                device=matched_pose.device,
            )
            axis_C = matched_pose[:3, :3] @ axis_O
            along_mm = float(torch.abs(torch.dot(translation_delta, axis_C)) * 1000.0)
            perpendicular_mm = float(
                torch.linalg.vector_norm(
                    translation_delta
                    - torch.dot(translation_delta, axis_C) * axis_C
                )
                * 1000.0
            )
            weights = prediction.correspondence_confidence[index, mask]
            count = int(mask.sum())
            diagnostics = prediction.correspondence_pose_diagnostics
            row.update(
                {
                    "selected_shared_symmetry_element": selected,
                    "loss_by_symmetry_element": joint_diagnostics[
                        "loss_by_symmetry_element"
                    ][index].detach().cpu().tolist(),
                    "rotation_error_by_symmetry_element": joint_diagnostics[
                        "rotation_error_by_symmetry_element_deg"
                    ][index].detach().cpu().tolist(),
                    "translation_error_by_symmetry_element": joint_diagnostics[
                        "translation_error_by_symmetry_element_mm"
                    ][index].detach().cpu().tolist(),
                    "correspondence_error_by_symmetry_element": joint_diagnostics[
                        "correspondence_error_by_symmetry_element_mm"
                    ][index].detach().cpu().tolist(),
                    "correspondence_rmse_mm": float(torch.sqrt((correspondence_distance.square()).mean()) * 1000.0),
                    "correspondence_p50_mm": float(torch.quantile(correspondence_distance, 0.50) * 1000.0),
                    "correspondence_p95_mm": float(torch.quantile(correspondence_distance, 0.95) * 1000.0),
                    "correspondence_max_mm": float(correspondence_distance.max() * 1000.0),
                    "predicted_to_template_surface_p50_mm": float(torch.quantile(surface_distance, 0.50) * 1000.0),
                    "predicted_to_template_surface_p95_mm": float(torch.quantile(surface_distance, 0.95) * 1000.0),
                    "predicted_to_template_surface_max_mm": float(surface_distance.max() * 1000.0),
                    "template_visible_patch_to_predicted_p95_mm": float(torch.quantile(visible_patch_to_predicted, 0.95) * 1000.0),
                    "symmetric_chamfer_p95_mm": float(max(torch.quantile(visible_patch_to_predicted, 0.95), torch.quantile(predicted_to_visible_patch, 0.95)) * 1000.0),
                    "rotation_error_deg": rotation_error,
                    "translation_total_mm": translation_mm,
                    "translation_along_axis_mm": along_mm,
                    "translation_perpendicular_to_axis_mm": perpendicular_mm,
                    "visible_alignment_rmse_mm": float(torch.sqrt(alignment_distance.square().mean()) * 1000.0),
                    "visible_alignment_p95_mm": float(torch.quantile(alignment_distance, 0.95) * 1000.0),
                    "visible_alignment_max_mm": float(alignment_distance.max() * 1000.0),
                    "pose_success_1deg_1mm": rotation_error <= 1.0 and translation_mm <= 1.0,
                    "pose_success_2deg_2mm": rotation_error <= 2.0 and translation_mm <= 2.0,
                    "pose_success_5deg_5mm": rotation_error <= 5.0 and translation_mm <= 5.0,
                    "weighting_mode": getattr(prediction, "weighting_mode", None),
                    "effective_correspondence_count": float(1.0 / weights.square().sum().clamp_min(1e-12)),
                    "effective_correspondence_fraction": float((1.0 / weights.square().sum().clamp_min(1e-12)) / max(count, 1)),
                    "max_correspondence_weight": float(weights.max()),
                    "valid_point_count": count,
                    "procrustes_rank": int(diagnostics["rank"][index]),
                    "procrustes_rank_valid": bool(diagnostics["rank_valid"][index]),
                    "procrustes_determinant": float(diagnostics["determinant"][index]),
                    "procrustes_orthogonality_error": float(diagnostics["orthogonality_error"][index]),
                    "procrustes_reflection_corrected": bool(diagnostics["reflection_corrected"][index]),
                    "triangle_target_index_mismatch_fraction": float(
                        joint_diagnostics.get(
                            "triangle_target_index_mismatch_fraction",
                            torch.as_tensor(0.0, device=vertices.device),
                        )
                    ),
                    "covariance_penalty_active": float(
                        joint_diagnostics.get(
                            "covariance_penalty_active",
                            torch.as_tensor(0.0, device=vertices.device),
                        )
                    ),
                    "predicted_eigenvalues": joint_diagnostics.get(
                        "predicted_covariance_eigenvalues",
                        torch.empty(0, device=vertices.device),
                    )[index].detach().cpu().tolist(),
                    "gt_eigenvalues": joint_diagnostics.get(
                        "gt_covariance_eigenvalues",
                        torch.empty(0, device=vertices.device),
                    )[index].detach().cpu().tolist(),
                    "eigenvalue_ratios": joint_diagnostics.get(
                        "selected_eigenvalue_ratios",
                        torch.empty(0, device=vertices.device),
                    )[index].detach().cpu().tolist(),
                    "rank_margin_m2": float(
                        joint_diagnostics.get(
                            "rank_margin_m2",
                            torch.zeros(len(prediction.correspondence_points_O), device=vertices.device),
                        )[index]
                    ),
                }
            )
            rank_valid = bool(diagnostics["rank_valid"][index])
            predicted_geometry = covariance_geometry(
                prediction.correspondence_points_O[index, mask]
            )
            target_geometry = covariance_geometry(matched_target[mask])
            predicted_eigenvalues = predicted_geometry[
                "covariance_eigenvalues"
            ].clamp_min(0)
            target_eigenvalues = target_geometry[
                "covariance_eigenvalues"
            ].clamp_min(0)
            row.update(
                predicted_covariance_eigenvalues=predicted_eigenvalues.detach().cpu().tolist(),
                gt_covariance_eigenvalues=target_eigenvalues.detach().cpu().tolist(),
                eigenvalue_ratio=(
                    predicted_eigenvalues / target_eigenvalues.clamp_min(1e-12)
                ).detach().cpu().tolist(),
                correspondence_rank=int(predicted_geometry["rank"]),
                rank_invalid_fraction=float(not rank_valid),
            )
            if not rank_valid:
                row["pose_success_1deg_1mm"] = False
                row["pose_success_2deg_2mm"] = False
                row["pose_success_5deg_5mm"] = False
                row["correspondence_pose_success_2deg_2mm"] = False
                row["correspondence_pose_success_5deg_5mm"] = False

            auxiliary = getattr(prediction, "correspondence_auxiliary", None)
            if auxiliary is not None:
                for diagnostic_name in (
                    "fine_feature_variance",
                    "fine_feature_effective_rank",
                    "fine_feature_pairwise_distance",
                    "fine_feature_collision_fraction",
                    "fine_candidate_logit_variance",
                ):
                    diagnostic = auxiliary.get(diagnostic_name)
                    if diagnostic is not None:
                        row[diagnostic_name] = float(diagnostic.float().mean())
                for diagnostic_name in (
                    "aux_coordinate_rmse_mm",
                    "aux_coordinate_p95_mm",
                ):
                    diagnostic = joint_diagnostics.get(diagnostic_name)
                    if diagnostic is not None:
                        row[diagnostic_name] = float(diagnostic)
            if auxiliary is not None and all(
                key in auxiliary
                for key in (
                    "coarse_patch_logits",
                    "selected_topk_patch_ids",
                    "selected_triangle_ids",
                    "candidate_triangle_ids",
                    "patch_points_O",
                )
            ):
                q_gt = matched_target[mask]
                all_candidate_ids = auxiliary.get("all_candidate_triangle_ids")
                coarse_logits = auxiliary["coarse_patch_logits"][index, mask]
                predicted_patch = coarse_logits.argmax(-1)
                topk = auxiliary["selected_topk_patch_ids"][index, mask]
                cached_valid_global = auxiliary.get(
                    "teacher_forcing_valid_triangle_global_mask"
                )
                cached_gt_triangle = auxiliary.get(
                    "teacher_forcing_gt_triangle_ids"
                )
                if cached_valid_global is not None and cached_gt_triangle is not None:
                    triangle_targets = {
                        "face_ids": cached_gt_triangle[index, mask],
                        "valid_triangle_mask": cached_valid_global[index, mask],
                    }
                else:
                    triangle_targets = triangle_target_sets(
                        q_gt,
                        vertices,
                        faces,
                        tolerance_m=float(
                            (joint_loss_config or {}).get(
                                "triangle_target_tolerance_m", 0.00015
                            )
                        ),
                        point_chunk_size=256,
                    )
                gt_triangle = triangle_targets["face_ids"]
                if all_candidate_ids is not None:
                    valid_targets = valid_patch_mask(
                        gt_triangle, all_candidate_ids[index]
                    )
                    owners = auxiliary.get("face_owner_patch_ids")
                    gt_patch = (
                        single_owner_patch_ids(gt_triangle, owners[index])
                        if owners is not None
                        else valid_targets.to(torch.int64).argmax(-1)
                    )
                else:
                    patch_points = auxiliary["patch_points_O"][index]
                    gt_patch = torch.cdist(
                        q_gt.float(), patch_points.float()
                    ).argmin(-1)
                    valid_targets = torch.nn.functional.one_hot(
                        gt_patch, num_classes=coarse_logits.shape[-1]
                    ).bool()
                selected_triangle = auxiliary["selected_triangle_ids"][index, mask]
                candidate_triangle = auxiliary["candidate_triangle_ids"][index, mask]
                candidate_triangle_mask = auxiliary.get("candidate_triangle_mask")
                candidate_triangle_mask = (
                    candidate_triangle.ge(0)
                    if candidate_triangle_mask is None
                    else candidate_triangle_mask[index, mask]
                )
                valid_local_triangle = local_valid_triangle_mask(
                    candidate_triangle, triangle_targets["valid_triangle_mask"]
                ) & candidate_triangle_mask
                fine_logits_for_metrics = auxiliary["fine_local_logits"][index, mask]
                fine_top1 = fine_logits_for_metrics.argmax(-1, keepdim=True)
                valid_triangle_top1 = valid_local_triangle.gather(
                    -1, fine_top1
                ).squeeze(-1)
                fine_top4 = fine_logits_for_metrics.topk(
                    min(4, fine_logits_for_metrics.shape[-1]), dim=-1
                ).indices
                valid_triangle_top4 = valid_local_triangle.gather(
                    -1, fine_top4
                ).any(-1)
                valid_triangle_count = triangle_targets[
                    "valid_triangle_mask"
                ].sum(-1)
                candidate_count = candidate_triangle_mask.sum(-1)
                sorted_candidate = candidate_triangle.masked_fill(
                    ~candidate_triangle_mask, -1
                ).sort(-1).values
                unique_candidate = sorted_candidate.ge(0)
                if sorted_candidate.shape[-1] > 1:
                    unique_candidate[:, 1:] &= sorted_candidate[:, 1:].ne(
                        sorted_candidate[:, :-1]
                    )
                has_duplicate_candidate = unique_candidate.sum(-1).ne(
                    candidate_count
                )
                local_triangle_loss = -torch.logsumexp(
                    torch.log_softmax(fine_logits_for_metrics, -1).masked_fill(
                        ~valid_local_triangle, float("-inf")
                    ),
                    dim=-1,
                ).mean()
                random_local_ce = candidate_count.float().log().mean()
                reconstruction = torch.linalg.vector_norm(
                    prediction.correspondence_points_O[index, mask] - q_gt, dim=-1
                )
                patch_count = int(coarse_logits.shape[-1])
                confusion = torch.zeros(
                    (patch_count, patch_count), dtype=torch.long,
                    device=coarse_logits.device,
                )
                confusion.index_put_(
                    (gt_patch, predicted_patch),
                    torch.ones_like(gt_patch, dtype=torch.long), accumulate=True,
                )
                unique_patch, patch_frequency = predicted_patch.unique(
                    return_counts=True
                )
                unique_triangle, triangle_frequency = selected_triangle.unique(
                    return_counts=True
                )
                valid_count = valid_targets.sum(-1)
                valid_top1 = valid_targets.gather(
                    -1, predicted_patch[:, None]
                ).squeeze(-1)
                valid_top4 = valid_set_topk_hits(
                    topk, valid_targets, 4, already_topk=True
                )
                valid_top8 = valid_set_topk_hits(coarse_logits, valid_targets, 8)
                row.update(
                    coarse_patch_top1_accuracy=float(
                        predicted_patch.eq(gt_patch).float().mean()
                    ),
                    coarse_patch_top4_recall=float(
                        topk[:, : min(4, topk.shape[-1])]
                        .eq(gt_patch[:, None]).any(-1).float().mean()
                    ),
                    coarse_patch_top8_recall=float(
                        coarse_logits.topk(min(8, patch_count), -1).indices
                        .eq(gt_patch[:, None]).any(-1).float().mean()
                    ),
                    gt_patch_in_candidate_set_fraction=float(
                        topk.eq(gt_patch[:, None]).any(-1).float().mean()
                    ),
                    valid_patch_set_top1_accuracy=float(valid_top1.float().mean()),
                    valid_patch_set_top4_recall=float(valid_top4.float().mean()),
                    valid_patch_set_top8_recall=float(valid_top8.float().mean()),
                    valid_patch_set_in_candidate_set_fraction=float(
                        valid_top4.float().mean()
                    ),
                    mean_valid_patch_count=float(valid_count.float().mean()),
                    max_valid_patch_count=int(valid_count.max()),
                    fraction_with_multiple_valid_patches=float(
                        valid_count.gt(1).float().mean()
                    ),
                    wrong_top1_but_same_triangle_available_fraction=float(
                        (predicted_patch.ne(gt_patch) & valid_top1).float().mean()
                    ),
                    triangle_top1_accuracy=float(
                        selected_triangle.eq(gt_triangle).float().mean()
                    ),
                    single_owner_triangle_top1=float(
                        selected_triangle.eq(gt_triangle).float().mean()
                    ),
                    valid_triangle_set_top1=float(
                        valid_triangle_top1.float().mean()
                    ),
                    valid_triangle_set_top1_accuracy=float(
                        valid_triangle_top1.float().mean()
                    ),
                    valid_triangle_set_top4=float(
                        valid_triangle_top4.float().mean()
                    ),
                    valid_triangle_set_top4_recall=float(
                        valid_triangle_top4.float().mean()
                    ),
                    mean_valid_triangle_count=float(
                        valid_triangle_count.float().mean()
                    ),
                    fraction_with_multiple_valid_triangles=float(
                        valid_triangle_count.gt(1).float().mean()
                    ),
                    valid_triangle_candidate_recall=float(
                        valid_local_triangle.any(-1).float().mean()
                    ),
                    candidate_recall=float(
                        valid_local_triangle.any(-1).float().mean()
                    ),
                    local_triangle_set_ce=float(local_triangle_loss),
                    local_triangle_random_ce=float(random_local_ce),
                    local_triangle_classifier_worse_than_uniform=bool(
                        local_triangle_loss > random_local_ce
                    ),
                    mean_local_candidate_count=float(
                        candidate_count.float().mean()
                    ),
                    min_local_candidate_count=int(candidate_count.min()),
                    max_local_candidate_count=int(candidate_count.max()),
                    invalid_candidate_count_fraction=float(
                        candidate_count.ne(candidate_triangle.shape[-1])
                        .to(torch.float64)
                        .mean()
                    ),
                    duplicate_local_candidate_fraction=float(
                        has_duplicate_candidate.float().mean()
                    ),
                    teacher_forcing_selected_symmetry_element=float(
                        auxiliary.get(
                            "teacher_forcing_selected_symmetry_element",
                            torch.full(
                                (len(prediction.correspondence_points_O),),
                                -1,
                                device=vertices.device,
                            ),
                        )[index]
                    ),
                    gt_triangle_in_local_candidates_fraction=float(
                        candidate_triangle.eq(gt_triangle[:, None]).any(-1).float().mean()
                    ),
                    barycentric_reconstruction_p50_mm=float(
                        torch.quantile(reconstruction, 0.50) * 1000.0
                    ),
                    barycentric_reconstruction_p95_mm=float(
                        torch.quantile(reconstruction, 0.95) * 1000.0
                    ),
                    unique_predicted_patches=int(unique_patch.numel()),
                    unique_predicted_triangles=int(unique_triangle.numel()),
                    most_popular_patch_fraction=float(
                        patch_frequency.max() / max(len(predicted_patch), 1)
                    ),
                    most_popular_triangle_fraction=float(
                        triangle_frequency.max() / max(len(selected_triangle), 1)
                    ),
                    patch_confusion_matrix=confusion.detach().cpu().tolist(),
                )
                if "coarse_points_O" in auxiliary:
                    coarse_points = auxiliary["coarse_points_O"][index, mask]
                    coarse_row = torch.linalg.vector_norm(coarse_points - q_gt, dim=-1)
                    coarse_surface = closest_points_on_triangle_mesh(
                        coarse_points, vertices, faces
                    )["distances"]
                    row.update(
                        coarse_q_row_p50_mm=float(torch.quantile(coarse_row, .50) * 1000.),
                        coarse_q_row_p95_mm=float(torch.quantile(coarse_row, .95) * 1000.),
                        coarse_q_surface_p50_mm=float(torch.quantile(coarse_surface, .50) * 1000.),
                        coarse_q_surface_p95_mm=float(torch.quantile(coarse_surface, .95) * 1000.),
                        coarse_correspondence_rank=int(covariance_geometry(coarse_points)["rank"]),
                        refined_correspondence_rank=int(covariance_geometry(
                            prediction.correspondence_points_O[index, mask]
                        )["rank"]),
                    )
            row["physical_normalized_score"] = physical_normalized_score(
                row["rotation_error_deg"], row["translation_total_mm"],
                row["correspondence_p95_mm"], row["visible_alignment_p95_mm"],
                row["predicted_to_template_surface_p95_mm"],
            )
        rows.append(row)
    return rows


def aggregate_metric_rows(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, bool)) and key != "fragment_id"
        }
    )
    return {
        key: float(sum(float(row[key]) for row in rows if key in row) / max(sum(key in row for row in rows), 1))
        for key in numeric_keys
    }


__all__ = [
    "aggregate_metric_rows",
    "batch_pose_metric_rows",
    "physical_normalized_score",
    "score_pose_cost_spearman",
]
