"""Reproducible CPU/CUDA debug trainer for the current pure-PyTorch baseline."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from symm_template_reg.datasets.fragment_mesh_filter import sha256_file
from symm_template_reg.config import validate_data_policy
from symm_template_reg.engine.checkpoint import load_checkpoint, save_checkpoint
from symm_template_reg.engine.evaluator import evaluate_model, move_to_device
from symm_template_reg.engine.logging import MetricLogger
from symm_template_reg.engine.manifest import (
    load_and_validate_manifest,
    validate_debug_training_flags,
)
from symm_template_reg.engine.seed import seed_everything
from symm_template_reg.models import build_loss, build_model, register_all_modules
from symm_template_reg.models.losses import (
    ConditionedMultiHypothesisPoseLoss,
    CorrespondenceLoss,
    CorrespondenceConfidenceRegularizationLoss,
    DirectCorrespondencePoseConsistencyLoss,
    CrossViewWorldPoseLoss,
    OverlapLoss,
    PointConfidenceLoss,
    PoseQueryRankingLoss,
    PairwisePoseResponseLoss,
    symmetry_aware_pose_costs,
    SymmetryAwareCorrespondenceLoss,
    JointCorrespondencePoseLoss,
    JointSurfaceCorrespondencePoseLossV3,
    CleanCoordinatePoseLossV3,
)
from symm_template_reg.models.pose.metrics import symmetry_aware_pose_errors
from symm_template_reg.models.losses.region_loss import (
    active_region_binary_loss,
    aggregate_point_region_activity,
    masked_point_region_cross_entropy,
)
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg


def resolve_device(name: str) -> torch.device:
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: CUDA is unavailable; --device auto selected CPU")
    return torch.device("cpu")


def _unique_work_dir(base: str | Path) -> Path:
    root = Path(base).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        candidate = root / (stamp if suffix == 0 else f"{stamp}_{suffix:03d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate unique run directory below {root}")


def _build_dataset(config: Mapping[str, Any]) -> Any:
    dataset_cfg = deepcopy(dict(config["dataset"]))
    data_cfg = config.get("data", {})
    if isinstance(data_cfg, Mapping):
        if "fragment_mesh_filter" in data_cfg:
            dataset_cfg["fragment_mesh_filter"] = deepcopy(
                data_cfg["fragment_mesh_filter"]
            )
        if "observed_filter" in data_cfg:
            dataset_cfg["observed_filter"] = deepcopy(data_cfg["observed_filter"])
        if "symmetry_region_activity" in data_cfg:
            dataset_cfg["symmetry_region_activity"] = deepcopy(
                data_cfg["symmetry_region_activity"]
            )
    return build_from_cfg(dataset_cfg, DATASETS)


def _normalize_data_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Make the resolved config reflect the authoritative ``data`` section."""

    resolved = deepcopy(dict(config))
    data_cfg = resolved.get("data", {})
    if not isinstance(data_cfg, Mapping):
        return resolved
    dataset_cfg = deepcopy(dict(resolved.get("dataset", {})))
    for key in ("fragment_mesh_filter", "observed_filter", "symmetry_region_activity"):
        if key in data_cfg:
            value = deepcopy(data_cfg[key])
            dataset_cfg[key] = value
            resolved[key] = deepcopy(value)
    resolved["dataset"] = dataset_cfg
    return resolved


def _padded_correspondence(batch: Mapping[str, Any]) -> torch.Tensor | None:
    target = batch["gt"].get("points_O_corresponding")
    if target is None:
        return None
    if hasattr(target, "to_padded"):
        return target.to_padded()["points"]
    return target


def _padded_overlap(batch: Mapping[str, Any], prediction: Any) -> torch.Tensor | None:
    labels = batch["gt"].get("overlap_labels")
    if labels is None or labels.ndim == 2:
        return labels
    padded = torch.zeros_like(prediction.observed_valid_mask)
    start = 0
    for index, length in enumerate(prediction.observed_valid_mask.sum(-1).tolist()):
        padded[index, :length] = labels[start : start + length]
        start += length
    return padded


def _padded_points(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(value, "to_padded"):
        padded = value.to_padded()
        return padded["points"], padded["valid_mask"]
    points = value.get("points_C", value.get("points_O", value.get("points")))
    return points, value["valid_mask"]


def _joint_correspondence_training_losses(
    prediction: Any,
    batch: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = dict(training_config.get("joint_correspondence_pose", {}))
    if not bool(cfg.pop("enabled", False)):
        raise ValueError("joint model requires joint_correspondence_pose.enabled=true")
    target = _padded_correspondence(batch)
    if target is None:
        raise ValueError("joint loss requires row-aligned canonical correspondences")
    observed_points, observed_mask = _padded_points(batch["observed"])
    template_points, template_mask = _padded_points(batch["template"])
    if not torch.equal(observed_mask, prediction.observed_valid_mask):
        raise ValueError("joint observed masks disagree")
    losses = JointCorrespondencePoseLoss(**cfg)(
        predicted_points_O=prediction.correspondence_points_O,
        predicted_pose_T_C_from_O=prediction.correspondence_pose,
        gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
        observed_points_C=observed_points,
        target_points_O=target,
        valid_mask=prediction.observed_valid_mask,
        template_surface_points_O=template_points,
        template_valid_mask=template_mask,
        symmetry_metadata=batch["template_symmetry_metadata"],
        effective_symmetry_groups=batch["gt"]["effective_symmetry_group"],
    )
    zero = losses["loss_total"].new_zeros(())
    losses["loss_symmetry_pose"] = (
        losses["weighted_loss_rotation"] + losses["weighted_loss_translation"]
    )
    losses["weighted_loss_symmetry_pose"] = losses["loss_symmetry_pose"]
    losses["loss_pose_classification"] = zero
    losses["weighted_loss_pose_classification"] = zero
    return losses["loss_total"], losses


def _joint_surface_correspondence_training_losses(
    prediction: Any,
    batch: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = dict(training_config.get("joint_surface_correspondence_pose_v3", {}))
    if not bool(cfg.pop("enabled", False)):
        raise ValueError("surface V2 model requires joint_surface_correspondence_pose_v3.enabled=true")
    if prediction.correspondence_auxiliary is None:
        raise ValueError("V3 loss requires SurfaceConstrainedCorrespondenceHeadV2 auxiliary outputs")
    target = _padded_correspondence(batch)
    if target is None:
        raise ValueError("V3 loss requires row-aligned canonical correspondences")
    observed_points, observed_mask = _padded_points(batch["observed"])
    template_points, template_mask = _padded_points(batch["template"])
    if not torch.equal(observed_mask, prediction.observed_valid_mask):
        raise ValueError("V3 observed masks disagree")
    if bool(cfg.pop("clean_active_only", False)):
        current_epoch = int(cfg.pop("_runtime_epoch", 0))
        auxiliary = prediction.correspondence_auxiliary
        q_normalized = auxiliary.get("fine_aux_coordinate_normalized")
        if q_normalized is None:
            raise ValueError("clean V3 loss requires normalized q_aux")
        losses = CleanCoordinatePoseLossV3(
            current_epoch=current_epoch, **cfg
        )(
            predicted_normalized_O=q_normalized,
            observed_points_C=observed_points,
            target_points_O=target,
            valid_mask=prediction.observed_valid_mask,
            gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
            symmetry_metadata=batch["template_symmetry_metadata"],
            effective_symmetry_groups=batch["gt"]["effective_symmetry_group"],
            template_mesh_vertices_O=batch["template_mesh_vertices_O"],
        )
        zero = losses["loss_total"].new_zeros(())
        losses["loss_symmetry_pose"] = (
            losses["weighted_loss_rotation"] + losses["weighted_loss_translation"]
        )
        losses["weighted_loss_symmetry_pose"] = losses["loss_symmetry_pose"]
        losses["loss_pose_classification"] = zero
        losses["weighted_loss_pose_classification"] = zero
        return losses["loss_total"], losses
    losses = JointSurfaceCorrespondencePoseLossV3(**cfg)(
        predicted_points_O=prediction.correspondence_points_O,
        predicted_pose_T_C_from_O=prediction.correspondence_pose,
        gt_pose_T_C_from_O=batch["gt"]["T_C_from_O"],
        observed_points_C=observed_points,
        target_points_O=target,
        valid_mask=prediction.observed_valid_mask,
        template_surface_points_O=template_points,
        template_valid_mask=template_mask,
        symmetry_metadata=batch["template_symmetry_metadata"],
        effective_symmetry_groups=batch["gt"]["effective_symmetry_group"],
        correspondence_auxiliary=prediction.correspondence_auxiliary,
        template_mesh_vertices_O=batch["template_mesh_vertices_O"],
        template_mesh_faces=batch["template_mesh_faces"],
        pose_rank_valid=prediction.correspondence_pose_diagnostics.get("rank_valid"),
    )
    zero = losses["loss_total"].new_zeros(())
    losses["loss_symmetry_pose"] = (
        losses["weighted_loss_rotation"] + losses["weighted_loss_translation"]
    )
    losses["weighted_loss_symmetry_pose"] = losses["loss_symmetry_pose"]
    losses["loss_pose_classification"] = zero
    losses["weighted_loss_pose_classification"] = zero
    return losses["loss_total"], losses


def _conditioned_training_losses(
    prediction: Any,
    batch: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    config = dict(training_config.get("conditioned_pose_loss", {}))
    criterion = ConditionedMultiHypothesisPoseLoss(
        base_pose_weight=float(config.get("base_pose_weight", 1.0)),
        best_residual_pose_weight=float(
            config.get("best_residual_pose_weight", 1.0)
        ),
        residual_regularization_weight=float(
            config.get("residual_regularization_weight", 0.01)
        ),
        translation_weight=float(training_config.get("translation_cost_weight", 10.0)),
        rotation_weight=float(training_config.get("rotation_cost_weight", 1.0)),
    )
    conditioned = criterion(
        prediction.base_pose,
        prediction.pose_hypotheses,
        batch["gt"]["T_C_from_O"],
        batch["template_symmetry_metadata"],
        batch["gt"]["effective_symmetry_group"],
        prediction.residual_pose_parameters,
    )
    losses = dict(conditioned)
    total = conditioned["loss_conditioned_pose"]
    zero = total.new_zeros(())
    pose_supervision = (
        float(config.get("base_pose_weight", 1.0)) * conditioned["loss_base_pose"]
        + float(config.get("best_residual_pose_weight", 1.0))
        * conditioned["loss_best_residual_pose"]
    )
    losses.update(
        {
            "loss_symmetry_pose": pose_supervision,
            "weighted_loss_symmetry_pose": pose_supervision,
            "loss_translation": zero,
            "loss_rotation": zero,
            "loss_pose_classification": zero,
            "weighted_loss_pose_classification": zero,
            "loss_pose_auxiliary": zero,
            "weighted_loss_pose_auxiliary": zero,
            "assigned_query_indices": conditioned[
                "conditioned_best_query_indices"
            ],
        }
    )
    auxiliary_weight = float(
        training_config.get("pose_decoder_auxiliary_weight", 0.0)
    )
    if auxiliary_weight and prediction.auxiliary_outputs:
        auxiliary_values = []
        for output in prediction.auxiliary_outputs:
            auxiliary_values.append(
                criterion(
                    prediction.base_pose,
                    output["pose_hypotheses"],
                    batch["gt"]["T_C_from_O"],
                    batch["template_symmetry_metadata"],
                    batch["gt"]["effective_symmetry_group"],
                    output.get("residual_pose_parameters"),
                )["loss_conditioned_pose"]
            )
        auxiliary = torch.stack(auxiliary_values).mean()
        losses["loss_pose_auxiliary"] = auxiliary
        losses["weighted_loss_pose_auxiliary"] = auxiliary_weight * auxiliary
        total = total + losses["weighted_loss_pose_auxiliary"]
    correspondence_config = dict(training_config.get("correspondence_loss", {}))
    if bool(correspondence_config.get("enabled", False)):
        target = _padded_correspondence(batch)
        if target is None:
            raise ValueError("enabled correspondence loss requires row-aligned points_O")
        correspondence_result = SymmetryAwareCorrespondenceLoss(
            robust_type=str(correspondence_config.get("robust_type", "smooth_l1")),
            beta=float(correspondence_config.get("beta", 0.01)),
            use_shared_symmetry_element=bool(
                correspondence_config.get("use_shared_symmetry_element", True)
            ),
        ).forward_with_diagnostics(
            prediction.correspondence_points_O,
            target,
            prediction.observed_valid_mask,
            batch["template_symmetry_metadata"],
            batch["gt"]["effective_symmetry_group"],
            prediction.correspondence_confidence,
        )
        correspondence = correspondence_result["loss"]
        losses["loss_symmetry_aware_correspondence"] = correspondence
        losses["selected_shared_symmetry_element"] = correspondence_result[
            "selected_shared_symmetry_element"
        ]
        losses["weighted_loss_symmetry_aware_correspondence"] = float(
            correspondence_config.get("weight", 1.0)
        ) * correspondence
        total = total + losses["weighted_loss_symmetry_aware_correspondence"]
        confidence_config = dict(
            correspondence_config.get("confidence_regularization", {})
        )
        if bool(confidence_config.get("enabled", False)):
            confidence_values = CorrespondenceConfidenceRegularizationLoss(
                minimum_effective_point_count=float(
                    confidence_config.get("minimum_effective_point_count", 16.0)
                ),
                minimum_weight_sum=float(
                    confidence_config.get("minimum_weight_sum", 1e-3)
                ),
            )(
                prediction.correspondence_confidence,
                prediction.observed_valid_mask,
            )
            losses.update(confidence_values)
            losses["weighted_loss_confidence_regularization"] = float(
                confidence_config.get("weight", 0.01)
            ) * confidence_values["loss_confidence_regularization"]
            total = total + losses["weighted_loss_confidence_regularization"]
    if prediction.correspondence_pose is not None:
        corr_costs = []
        for index in range(len(prediction.correspondence_pose)):
            error = symmetry_aware_pose_errors(
                prediction.correspondence_pose[index].unsqueeze(0),
                batch["gt"]["T_C_from_O"][index],
                batch["template_symmetry_metadata"][index],
                effective_group=batch["gt"]["effective_symmetry_group"][index],
            )
            corr_costs.append(
                float(training_config.get("translation_cost_weight", 10.0))
                * error["translation_m"][0]
                + float(training_config.get("rotation_cost_weight", 1.0))
                * error["rotation_rad"][0]
            )
        correspondence_pose_loss = torch.stack(corr_costs).mean()
        losses["loss_correspondence_pose"] = correspondence_pose_loss
        pose_weight = float(training_config.get("correspondence_pose_loss_weight", 0.0))
        losses["weighted_loss_correspondence_pose"] = (
            pose_weight * correspondence_pose_loss
        )
        total = total + losses["weighted_loss_correspondence_pose"]
        consistency = DirectCorrespondencePoseConsistencyLoss(
            translation_weight=float(
                training_config.get("translation_cost_weight", 10.0)
            )
        )(prediction.base_pose, prediction.correspondence_pose)
        losses["loss_direct_correspondence_pose_consistency"] = consistency
        consistency_weight = float(
            training_config.get(
                "direct_vs_correspondence_pose_consistency_weight", 0.0
            )
        )
        losses["weighted_loss_direct_correspondence_pose_consistency"] = (
            consistency_weight * consistency
        )
        total = total + losses[
            "weighted_loss_direct_correspondence_pose_consistency"
        ]
    hybrid_config = dict(training_config.get("hybrid_direct_residual", {}))
    correction = getattr(prediction, "base_correction_transform", None)
    if bool(hybrid_config.get("enabled", False)):
        if correction is None:
            raise ValueError("hybrid_direct_residual requires a bounded base correction")
        identity = torch.eye(3, dtype=correction.dtype, device=correction.device)
        relative = correction[:, :3, :3]
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
        rotation = torch.acos(cosine.clamp(-1.0, 1.0))
        translation = torch.linalg.vector_norm(correction[:, :3, 3], dim=-1)
        regularization = (
            rotation.square().mean()
            + float(hybrid_config.get("translation_scale", 100.0))
            * translation.square().mean()
        )
        losses["loss_hybrid_residual_regularization"] = regularization
        losses["weighted_loss_hybrid_residual_regularization"] = float(
            hybrid_config.get("regularization_weight", 0.01)
        ) * regularization
        total = total + losses["weighted_loss_hybrid_residual_regularization"]
    cross_view_config = dict(
        training_config.get("cross_view_world_consistency", {})
    )
    if bool(cross_view_config.get("enabled", False)):
        T_W_from_C = batch["gt"].get("T_W_from_C")
        if not isinstance(T_W_from_C, torch.Tensor):
            raise ValueError("cross-view consistency requires GT T_W_from_C")
        cross_view = CrossViewWorldPoseLoss(
            rotation_weight=float(cross_view_config.get("rotation_weight", 1.0)),
            translation_weight=float(
                cross_view_config.get("translation_weight", 10.0)
            ),
            reference_mode=str(
                cross_view_config.get("reference_mode", "pairwise_medoid")
            ),
        )(
            prediction.base_pose,
            T_W_from_C,
            batch["template_symmetry_metadata"],
            batch["gt"]["effective_symmetry_group"],
        )
        losses.update(cross_view)
        cross_weight = float(cross_view_config.get("weight", 1.0))
        losses["weighted_loss_cross_view_world_pose"] = (
            cross_weight * cross_view["cross_view_world_pose_loss"]
        )
        total = total + losses["weighted_loss_cross_view_world_pose"]
    pairwise_config = dict(training_config.get("pairwise_pose_response", {}))
    if bool(pairwise_config.get("enabled", False)):
        pairwise = PairwisePoseResponseLoss(
            rotation_weight=float(pairwise_config.get("rotation_weight", 0.25)),
            translation_weight=float(
                pairwise_config.get("translation_weight", 0.25)
            ),
        )(prediction.base_pose, batch["gt"]["T_C_from_O"])
        losses.update(pairwise)
        pairwise_weight = float(pairwise_config.get("weight", 1.0))
        losses["weighted_loss_pairwise_pose_response"] = (
            pairwise_weight * pairwise["pairwise_pose_response_loss"]
        )
        total = total + losses["weighted_loss_pairwise_pose_response"]
    losses["loss_total"] = total
    return total, losses


def compute_training_losses(
    prediction: Any,
    batch: Mapping[str, Any],
    pose_criterion: torch.nn.Module,
    training_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    enabled_primary = sum(
        bool(dict(training_config.get(name, {})).get("enabled", False))
        for name in ("joint_correspondence_pose", "joint_surface_correspondence_pose_v3")
    )
    if enabled_primary > 1:
        raise ValueError("primary joint losses are mutually exclusive and total loss must be computed once")
    if bool(
        dict(training_config.get("joint_surface_correspondence_pose_v3", {})).get(
            "enabled", False
        )
    ):
        return _joint_surface_correspondence_training_losses(
            prediction, batch, training_config
        )
    if bool(
        dict(training_config.get("joint_correspondence_pose", {})).get(
            "enabled", False
        )
    ):
        return _joint_correspondence_training_losses(
            prediction, batch, training_config
        )
    if getattr(prediction, "base_pose", None) is not None:
        return _conditioned_training_losses(prediction, batch, training_config)
    use_decoder_auxiliary = bool(training_config.get("pose_decoder_auxiliary_loss", True))
    pose_losses = pose_criterion(
        prediction.pose_hypotheses,
        prediction.pose_logits,
        batch["gt"]["T_C_from_O"],
        prediction.auxiliary_outputs if use_decoder_auxiliary else None,
        symmetry_metadata=batch["template_symmetry_metadata"],
        effective_symmetry_groups=batch["gt"]["effective_symmetry_group"],
    )
    losses = dict(pose_losses)
    symmetry_pose = (
        float(getattr(pose_criterion, "translation_weight", 10.0))
        * pose_losses["loss_translation"]
        + float(getattr(pose_criterion, "rotation_weight", 1.0))
        * pose_losses["loss_rotation"]
    )
    losses["loss_symmetry_pose"] = symmetry_pose
    symmetry_weight = float(
        training_config.get(
            "symmetry_pose_weight", getattr(pose_criterion, "symmetry_pose_weight", 1.0)
        )
    )
    weighted_symmetry = symmetry_weight * symmetry_pose
    losses["weighted_loss_symmetry_pose"] = weighted_symmetry
    classification_weight = float(
        training_config.get(
            "pose_query_classification_weight",
            getattr(pose_criterion, "classification_weight", 0.0),
        )
    )
    weighted_classification = classification_weight * pose_losses[
        "loss_pose_classification"
    ]
    losses["weighted_loss_pose_classification"] = weighted_classification
    auxiliary_weight = float(getattr(pose_criterion, "auxiliary_weight", 0.0))
    weighted_decoder_auxiliary = auxiliary_weight * pose_losses["loss_pose_auxiliary"]
    losses["weighted_loss_pose_auxiliary"] = weighted_decoder_auxiliary
    total = weighted_symmetry + weighted_classification + weighted_decoder_auxiliary

    ranking_config = training_config.get("pose_query_ranking")
    if isinstance(ranking_config, Mapping) and float(ranking_config.get("weight", 0.0)) != 0.0:
        pose_costs = symmetry_aware_pose_costs(
            prediction.pose_hypotheses,
            batch["gt"]["T_C_from_O"],
            batch["template_symmetry_metadata"],
            batch["gt"]["effective_symmetry_group"],
            translation_weight=float(
                training_config.get("translation_cost_weight", 10.0)
            ),
            rotation_weight=float(training_config.get("rotation_cost_weight", 1.0)),
        )
        ranking = PoseQueryRankingLoss(
            type=str(ranking_config.get("type", "soft_quality")),
            temperature=float(ranking_config.get("temperature", 0.25)),
            cost_normalization=str(
                ranking_config.get("cost_normalization", "minmax")
            ),
            detach_pose_cost=bool(ranking_config.get("detach_pose_cost", True)),
        )(prediction.pose_logits, pose_costs)
        losses["loss_pose_query_ranking"] = ranking["loss_pose_query_ranking"]
        ranking_weight = float(ranking_config["weight"])
        losses["weighted_loss_pose_query_ranking"] = (
            ranking_weight * losses["loss_pose_query_ranking"]
        )
        total = total + losses["weighted_loss_pose_query_ranking"]

    auxiliary_enabled = bool(
        training_config.get("auxiliary_registration_losses", False)
    )
    if auxiliary_enabled:
        def auxiliary_weight_for(name: str) -> float:
            return float(training_config[name]) if name in training_config else 1.0

        correspondence_target = _padded_correspondence(batch)
        correspondence_weight = auxiliary_weight_for("correspondence_weight")
        if correspondence_target is not None and correspondence_weight != 0.0:
            losses["loss_correspondence"] = CorrespondenceLoss()(
                prediction.correspondence_points_O,
                correspondence_target,
                prediction.observed_valid_mask,
            )
            losses["weighted_loss_correspondence"] = (
                correspondence_weight * losses["loss_correspondence"]
            )
            total = total + losses["weighted_loss_correspondence"]
        overlap_target = _padded_overlap(batch, prediction)
        overlap_weight = auxiliary_weight_for("overlap_weight")
        point_weight = auxiliary_weight_for("point_weight_weight")
        if overlap_target is not None and (overlap_weight != 0.0 or point_weight != 0.0):
            losses["loss_overlap"] = OverlapLoss()(
                prediction.observed_overlap_logits,
                overlap_target,
                prediction.observed_valid_mask,
            )
            confidence_logits = torch.logit(
                prediction.correspondence_confidence.clamp(1e-5, 1.0 - 1e-5)
            )
            losses["loss_point_confidence"] = PointConfidenceLoss()(
                confidence_logits,
                overlap_target,
                prediction.observed_valid_mask,
            )
            losses["weighted_loss_overlap"] = overlap_weight * losses["loss_overlap"]
            losses["weighted_loss_point_confidence"] = (
                point_weight * losses["loss_point_confidence"]
            )
            total = (
                total
                + losses["weighted_loss_overlap"]
                + losses["weighted_loss_point_confidence"]
            )
        point_region_target = batch["gt"].get("point_symmetry_region_indices")
        point_region_valid = batch["gt"].get("point_symmetry_region_valid_mask")
        active_target = batch["gt"].get("active_symmetry_regions")
        active_valid = batch["gt"].get("active_symmetry_regions_valid_mask")
        observed_region_weight = auxiliary_weight_for("observed_region_weight")
        if (
            observed_region_weight != 0.0
            and prediction.observed_region_logits is not None
            and point_region_target is not None
            and point_region_valid is not None
            and active_valid is not None
        ):
            point_loss_config = dict(
                training_config.get("observed_region_loss", {})
            )
            losses["loss_observed_regions"] = masked_point_region_cross_entropy(
                prediction.observed_region_logits,
                point_region_target,
                point_region_valid & prediction.observed_valid_mask,
                active_valid,
                class_weights=point_loss_config.get("class_weights"),
            )
            losses["weighted_loss_observed_regions"] = (
                observed_region_weight * losses["loss_observed_regions"]
            )
            total = total + losses["weighted_loss_observed_regions"]
        active_region_weight = auxiliary_weight_for("active_region_weight")
        if (
            active_region_weight != 0.0
            and
            prediction.active_region_logits is not None
            and active_target is not None
            and active_valid is not None
        ):
            active_loss_config = dict(training_config.get("active_region_loss", {}))
            losses["loss_active_regions"] = active_region_binary_loss(
                prediction.active_region_logits,
                active_target,
                active_valid,
                loss_type=str(active_loss_config.get("type", "bce")),
                focal_gamma=float(active_loss_config.get("focal_gamma", 2.0)),
                pos_weight=active_loss_config.get("pos_weight"),
            )
            losses["weighted_loss_active_regions"] = (
                active_region_weight * losses["loss_active_regions"]
            )
            total = total + losses["weighted_loss_active_regions"]
        consistency_weight = auxiliary_weight_for("region_consistency_weight")
        if (
            consistency_weight != 0.0
            and prediction.observed_region_logits is not None
            and prediction.active_region_logits is not None
            and active_valid is not None
        ):
            consistency_config = dict(
                training_config.get("region_consistency", {})
            )
            aggregate = aggregate_point_region_activity(
                prediction.observed_region_logits,
                prediction.observed_valid_mask,
                active_valid,
                aggregation=str(consistency_config.get("aggregation", "topk_mean")),
                topk=int(consistency_config.get("topk", 16)),
            )
            width = min(aggregate.shape[-1], active_valid.shape[-1])
            active_probability = torch.sigmoid(
                prediction.active_region_logits[:, :width]
            )
            valid_slots = active_valid[:, :width]
            raw_consistency = (active_probability - aggregate[:, :width]).square()
            losses["loss_region_consistency"] = (
                raw_consistency * valid_slots.to(raw_consistency.dtype)
            ).sum() / valid_slots.sum().clamp_min(1)
            losses["region_consistency_mean_abs_error"] = (
                (active_probability - aggregate[:, :width]).abs()
                * valid_slots.to(active_probability.dtype)
            ).sum() / valid_slots.sum().clamp_min(1)
            losses["weighted_loss_region_consistency"] = (
                consistency_weight * losses["loss_region_consistency"]
            )
            total = total + losses["weighted_loss_region_consistency"]
    losses["loss_total"] = total
    return total, losses


def run_training(
    config: Mapping[str, Any],
    *,
    device_name: str = "auto",
    max_steps_override: int | None = None,
    work_dir_override: str | Path | None = None,
    manifest_override: str | Path | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    config = _normalize_data_config(config)
    validate_data_policy(config)
    validate_debug_training_flags(config)
    device = resolve_device(device_name)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    register_all_modules()
    dataset = _build_dataset(config)
    manifest_path = Path(
        manifest_override or config.get("sample_manifest", "")
    ).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest, manifest_digest = load_and_validate_manifest(
        manifest_path, config, dataset
    )
    record_index = {
        record.sample_id: index for index, record in enumerate(dataset.sample_records)
    }
    selected_indices = [record_index[str(sample["sample_id"])] for sample in manifest["samples"]]
    if not selected_indices:
        raise ValueError("training manifest contains no samples")
    selected_dataset = Subset(dataset, selected_indices)
    collate = build_from_cfg(
        config.get("collate", {"type": "FragmentTemplateCollator", "mode": "packed"}),
        COLLATE_FUNCTIONS,
    )
    dataloader_cfg = config.get("dataloader", {})
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        selected_dataset,
        batch_size=int(dataloader_cfg.get("batch_size", 4)),
        shuffle=bool(dataloader_cfg.get("shuffle", True)),
        num_workers=int(dataloader_cfg.get("num_workers", 0)),
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )
    model = build_model(config["model"]).to(device)
    optimizer_cfg = config.get("optimizer", {})
    if optimizer_cfg.get("type", "AdamW") != "AdamW":
        raise ValueError("current debug trainer supports AdamW only")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
    )
    training_cfg = config.get("training", {})
    amp = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    # The modern API avoids a deprecation warning on current PyTorch while the
    # fallback keeps the debug trainer usable on older supported installations.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch only
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_step = 0
    epoch = 0
    best_metric = math.inf
    if resume is not None:
        resumed = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            strict=True,
        )
        start_step = int(resumed["step"])
        epoch = int(resumed["epoch"])
        best_metric = float(resumed["best_metric"])
    base_work = work_dir_override or config.get("work_dir", "work_dirs/debug_training")
    work_dir = _unique_work_dir(base_work)
    resolved_config_path = work_dir / "resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    dataset.write_filter_artifacts(work_dir / "data_filter")
    accepted_csv = work_dir / "data_filter" / "accepted_fragments.csv"
    first_record = dataset.sample_records[selected_indices[0]]
    template = dataset.template_repository.get(first_record.object_model_id)
    template_path = Path(str(template["mesh_path"]))
    sidecar_path = (
        Path(str(template["symmetry_sidecar_path"]))
        if template.get("symmetry_sidecar_path")
        else None
    )
    checkpoint_manifest = {
        "debug_training_on_test_split": True,
        "results_are_not_final_evaluation": True,
        "resolved_config": config,
        "resolved_config_path": str(resolved_config_path),
        "dataset_root": str(dataset.dataset_root),
        "sample_manifest_path": str(manifest_path.resolve()),
        "sample_manifest_sha256": manifest_digest,
        "fragment_face_threshold": dataset.fragment_mesh_filter.config.get("min_num_faces"),
        "accepted_physical_fragments": dataset.index_report["accepted_physical_fragments"],
        "rejected_physical_fragments": dataset.index_report["rejected_physical_fragments"],
        "accepted_observations": dataset.index_report[
            "accepted_frame_observations_before_max_samples"
        ],
        "rejected_observations": (
            dataset.index_report["rejected_because_physical_fragment"]
            + dataset.index_report["rejected_observed_points_too_few"]
        ),
        "accepted_fragments_csv_sha256": sha256_file(accepted_csv),
        "template_path": str(template_path),
        "template_sha256": sha256_file(template_path),
        "symmetry_sidecar_path": str(sidecar_path) if sidecar_path else None,
        "symmetry_sidecar_sha256": sha256_file(sidecar_path) if sidecar_path else None,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": optimizer_cfg,
        "device": str(device),
        "amp": amp,
    }
    (work_dir / "run_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger = MetricLogger(work_dir)
    filter_log = {
        "phase": "data_filter",
        "physical_fragments_total": dataset.index_report[
            "total_physical_fragments"
        ],
        "physical_fragments_accepted": dataset.index_report[
            "accepted_physical_fragments"
        ],
        "physical_fragments_rejected": dataset.index_report[
            "rejected_physical_fragments"
        ],
        "frame_observations_total": dataset.index_report[
            "total_frame_observations"
        ],
        "frame_observations_accepted": dataset.index_report[
            "accepted_frame_observations_before_max_samples"
        ],
        "frame_observations_rejected_physical_fragment": dataset.index_report[
            "rejected_because_physical_fragment"
        ],
        "frame_observations_rejected_observed_points": dataset.index_report[
            "rejected_observed_points_too_few"
        ],
    }
    logger.log(filter_log)
    print(json.dumps(filter_log, sort_keys=True))
    pose_criterion = build_loss(config.get("loss", {"type": "PoseSetLoss"}))
    max_steps = int(
        max_steps_override
        if max_steps_override is not None
        else training_cfg.get("max_steps", 3000)
    )
    accumulation = int(training_cfg.get("gradient_accumulation_steps", 1))
    gradient_clip = float(training_cfg.get("gradient_clip_norm", 1.0))
    eval_interval = int(training_cfg.get("eval_interval", 100))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 250))
    patience_limit = int(training_cfg.get("early_stopping_patience", 10))
    patience = 0
    step = start_step
    last_evaluated_step = -1
    stop_training = False
    optimizer.zero_grad(set_to_none=True)
    model.train()
    while step < max_steps and not stop_training:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break
            moved = move_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp):
                prediction = model(moved)
                total, losses = compute_training_losses(
                    prediction, moved, pose_criterion, training_cfg
                )
                scaled_loss = total / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = ((step + 1) % accumulation) == 0 or (step + 1) == max_steps
            gradient_norm = float("nan")
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            row = {
                "phase": "train",
                "step": step,
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
                **{
                    key: float(value.detach())
                    for key, value in losses.items()
                    if isinstance(value, torch.Tensor) and value.ndim == 0
                },
            }
            logger.log(row)
            print(json.dumps(row, sort_keys=True))
            evaluated = eval_interval > 0 and step % eval_interval == 0
            is_best = False
            if evaluated:
                metrics, _ = evaluate_model(model, dataloader, device)
                metric = metrics.get("top1_rotation_error_deg", math.inf) + metrics.get(
                    "translation_total_mm", math.inf
                )
                is_best = metric < best_metric
                if is_best:
                    best_metric = metric
                    patience = 0
                else:
                    patience += 1
                logger.log({"phase": "eval", "step": step, **metrics})
                last_evaluated_step = step
            if checkpoint_interval > 0 and step % checkpoint_interval == 0:
                save_checkpoint(
                    work_dir / "checkpoints" / f"step_{step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    epoch=epoch,
                    best_metric=best_metric,
                    manifest=checkpoint_manifest,
                    is_best=is_best,
                )
            if evaluated and patience_limit > 0 and patience >= patience_limit:
                stop_training = True
                break
    final_is_best = False
    if step > 0 and last_evaluated_step != step:
        metrics, _ = evaluate_model(model, dataloader, device)
        metric = metrics.get("top1_rotation_error_deg", math.inf) + metrics.get(
            "translation_total_mm", math.inf
        )
        final_is_best = metric < best_metric
        if final_is_best:
            best_metric = metric
        logger.log({"phase": "eval", "step": step, **metrics})
    final_checkpoint = save_checkpoint(
        work_dir / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        epoch=epoch,
        best_metric=best_metric,
        manifest=checkpoint_manifest,
        is_best=(
            final_is_best
            or not (work_dir / "checkpoints" / "best.pt").exists()
        ),
    )
    logger.write_csv()
    summary = {
        "debug_training_on_test_split": True,
        "results_are_not_final_evaluation": True,
        "status": "ok",
        "work_dir": str(work_dir),
        "device": str(device),
        "steps": step,
        "epochs": epoch,
        "best_metric": best_metric,
        "last_checkpoint": str(final_checkpoint),
    }
    (work_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "compute_training_losses",
    "resolve_device",
    "run_training",
]
