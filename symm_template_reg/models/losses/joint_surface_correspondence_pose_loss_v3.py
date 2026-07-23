"""Surface-constrained correspondence loss with one shared symmetry choice."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from symm_template_reg.evaluation.correspondence_diagnostics import local_rigidity_errors
from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.geometry.patch_targets import (
    PATCH_TARGET_MODES,
    multi_positive_softmax_loss,
    single_owner_patch_ids,
    valid_patch_mask,
)
from symm_template_reg.models.geometry.triangle_targets import (
    closest_barycentric_on_triangles,
    local_valid_triangle_mask,
    triangle_target_sets,
)
from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.registry import LOSSES


def top_tail_mean(values: Tensor, fraction: float = 0.10) -> Tensor:
    """Mean of the largest ``ceil(fraction * N)`` values."""

    if values.numel() == 0:
        raise ValueError("tail loss requires at least one value")
    count = max(1, int(math.ceil(values.numel() * float(fraction))))
    return values.flatten().topk(count).values.mean()


def _normalized(raw: Tensor, scale: float) -> Tensor:
    value = raw / max(float(scale), 1e-12)
    return F.smooth_l1_loss(value, torch.zeros_like(value), reduction="none")


def _pad(rows: list[Tensor]) -> Tensor:
    width = max(row.numel() for row in rows)
    output = rows[0].new_full((len(rows), width), float("nan"))
    for index, row in enumerate(rows):
        output[index, : row.numel()] = row
    return output


def covariance_collapse_penalties(
    predicted_points: Tensor, target_points: Tensor, min_eigenvalue_m2: float
) -> dict[str, Tensor]:
    """Differentiable covariance/eigenvalue diagnostics for one point cloud."""

    def geometry(points: Tensor) -> tuple[Tensor, Tensor]:
        centered = points - points.mean(0)
        covariance = centered.transpose(0, 1) @ centered / max(len(points) - 1, 1)
        return covariance, torch.linalg.eigvalsh(covariance).clamp_min(0)

    predicted_covariance, predicted_eigenvalues = geometry(predicted_points)
    target_covariance, target_eigenvalues = geometry(target_points)
    threshold = max(float(min_eigenvalue_m2), 1e-12)
    return {
        "covariance_error_m2": torch.linalg.matrix_norm(
            predicted_covariance - target_covariance
        ),
        "min_eigenvalue_penalty": (
            threshold - predicted_eigenvalues[0]
        ).clamp_min(0) / threshold,
        "predicted_eigenvalues": predicted_eigenvalues,
        "target_eigenvalues": target_eigenvalues,
    }


def conditional_covariance_penalty(
    covariance_errors: Tensor,
    predicted_eigenvalues: Tensor,
    target_eigenvalues: Tensor,
    *,
    min_eigenvalue_m2: float,
    minimum_eigenvalue_ratio: float = 0.10,
) -> dict[str, Tensor]:
    """Mask covariance regularization once rank and eigenvalue ratios are healthy."""

    threshold = max(float(min_eigenvalue_m2), 1e-12)
    ratios = predicted_eigenvalues.unsqueeze(0) / target_eigenvalues.clamp_min(1e-12)
    rank_margin = predicted_eigenvalues[0] - threshold
    active = (rank_margin <= 0) | ratios.amin(-1).lt(float(minimum_eigenvalue_ratio))
    return {
        "penalty": covariance_errors * active.to(covariance_errors.dtype),
        "active": active,
        "eigenvalue_ratios": ratios,
        "rank_margin_m2": rank_margin,
    }


def coordinate_mean_and_tail_loss(
    predicted_normalized: Tensor,
    target_normalized: Tensor,
    tail_fraction: float = 0.10,
) -> tuple[Tensor, Tensor]:
    """Return per-target mean and worst-point SmoothL1 coordinate losses."""

    if predicted_normalized.ndim == target_normalized.ndim - 1:
        predicted_normalized = predicted_normalized.unsqueeze(0).expand_as(
            target_normalized
        )
    per_coordinate = F.smooth_l1_loss(
        predicted_normalized, target_normalized, reduction="none"
    )
    per_point = per_coordinate.mean(-1)
    count = max(1, math.ceil(per_point.shape[-1] * float(tail_fraction)))
    return per_point.mean(-1), per_point.topk(count, dim=-1).values.mean(-1)


def per_sample_mean_then_batch_mean(sample_losses: Sequence[Tensor]) -> Tensor:
    """Give every sample equal weight after its own point-wise reduction."""

    if not sample_losses:
        raise ValueError("at least one sample loss is required")
    reduced = [value.mean() if value.ndim else value for value in sample_losses]
    return torch.stack(reduced).mean()


@LOSSES.register_module()
class JointSurfaceCorrespondencePoseLossV3(nn.Module):
    """Joint V3 objective; every symmetry-dependent term is evaluated for the same S."""

    def __init__(
        self,
        correspondence_scale_m: float = 0.002,
        rotation_scale_deg: float = 2.0,
        translation_scale_m: float = 0.002,
        alignment_scale_m: float = 0.002,
        template_surface_scale_m: float = 0.001,
        tail_fraction: float = 0.10,
        local_rigidity_k: int = 8,
        lambda_patch_ce: float = 1.0,
        lambda_local_fine: float = 1.0,
        lambda_barycentric: float = 0.0,
        lambda_corr_mean: float = 1.0,
        lambda_corr_tail: float = 1.0,
        lambda_rot: float = 1.0,
        lambda_trans: float = 1.0,
        lambda_align_mean: float = 0.5,
        lambda_align_tail: float = 0.5,
        lambda_surface: float = 0.0,
        lambda_local_rigidity: float = 0.25,
        lambda_covariance: float = 0.5,
        lambda_min_eigenvalue: float = 0.5,
        lambda_patch_diversity: float = 0.25,
        fine_coordinate_aux_weight: float = 0.0,
        fine_coordinate_tail_weight: float = 0.0,
        raw_pose_rotation_weight: float = 0.0,
        raw_pose_translation_weight: float = 0.0,
        raw_alignment_weight: float = 0.0,
        raw_pose_rotation_scale_deg: float = 1.0,
        raw_pose_translation_scale_m: float = 0.001,
        raw_alignment_scale_m: float = 0.001,
        min_eigenvalue_m2: float = 1e-6,
        so2_samples: int = 36,
        patch_target_mode: str = "single_owner",
        triangle_target_mode: str = "single_owner",
        triangle_target_tolerance_m: float = 0.00015,
        require_exact_triangle_candidate: bool = False,
        use_teacher_forcing_shared_symmetry_element: bool = False,
        conditional_covariance: bool = True,
        covariance_minimum_eigenvalue_ratio: float = 0.10,
        loss_reduction: str = "per_sample_mean_then_batch_mean",
    ) -> None:
        super().__init__()
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if float(lambda_surface) != 0.0:
            raise ValueError(
                "exact barycentric surface output has identically-zero surface loss; "
                "lambda_surface must be 0"
            )
        if patch_target_mode not in PATCH_TARGET_MODES:
            raise ValueError(
                f"patch_target_mode must be one of {sorted(PATCH_TARGET_MODES)}, "
                f"got {patch_target_mode!r}"
            )
        self.patch_target_mode = patch_target_mode
        if triangle_target_mode not in PATCH_TARGET_MODES:
            raise ValueError(
                "triangle_target_mode must be 'single_owner' or "
                "'multi_valid_patch_set'"
            )
        self.triangle_target_mode = triangle_target_mode
        self.triangle_target_tolerance_m = float(triangle_target_tolerance_m)
        self.require_exact_triangle_candidate = bool(require_exact_triangle_candidate)
        self.use_teacher_forcing_shared_symmetry_element = bool(
            use_teacher_forcing_shared_symmetry_element
        )
        self.conditional_covariance = bool(conditional_covariance)
        self.covariance_minimum_eigenvalue_ratio = float(
            covariance_minimum_eigenvalue_ratio
        )
        if loss_reduction != "per_sample_mean_then_batch_mean":
            raise ValueError(
                "JointSurfaceCorrespondencePoseLossV3 requires "
                "loss_reduction='per_sample_mean_then_batch_mean'"
            )
        self.loss_reduction = loss_reduction
        self.scales = {
            "corr": float(correspondence_scale_m),
            "rot": math.radians(float(rotation_scale_deg)),
            "trans": float(translation_scale_m),
            "align": float(alignment_scale_m),
            "surface": float(template_surface_scale_m),
            "local": float(alignment_scale_m),
            "raw_aux_rot": math.radians(float(raw_pose_rotation_scale_deg)),
            "raw_aux_trans": float(raw_pose_translation_scale_m),
            "raw_aux_align": float(raw_alignment_scale_m),
        }
        self.weights = {
            "patch_ce": float(lambda_patch_ce),
            "local_fine": float(lambda_local_fine),
            "barycentric": float(lambda_barycentric),
            "corr_mean": float(lambda_corr_mean),
            "corr_tail": float(lambda_corr_tail),
            "rotation": float(lambda_rot),
            "translation": float(lambda_trans),
            "align_mean": float(lambda_align_mean),
            "align_tail": float(lambda_align_tail),
            "surface": float(lambda_surface),
            "local_rigidity": float(lambda_local_rigidity),
            "covariance": float(lambda_covariance),
            "min_eigenvalue": float(lambda_min_eigenvalue),
            "patch_diversity": float(lambda_patch_diversity),
            "fine_coordinate_aux": float(fine_coordinate_aux_weight),
            "fine_coordinate_aux_tail": float(fine_coordinate_tail_weight),
            "raw_aux_rotation": float(raw_pose_rotation_weight),
            "raw_aux_translation": float(raw_pose_translation_weight),
            "raw_aux_alignment": float(raw_alignment_weight),
        }
        self.aux_procrustes = WeightedProcrustes()
        self.tail_fraction = float(tail_fraction)
        self.local_rigidity_k = int(local_rigidity_k)
        self.min_eigenvalue_m2 = float(min_eigenvalue_m2)
        self.so2_samples = int(so2_samples)

    def forward(
        self,
        predicted_points_O: Tensor,
        predicted_pose_T_C_from_O: Tensor,
        gt_pose_T_C_from_O: Tensor,
        observed_points_C: Tensor,
        target_points_O: Tensor,
        valid_mask: Tensor,
        template_surface_points_O: Tensor,
        template_valid_mask: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
        correspondence_auxiliary: dict[str, Tensor],
        template_mesh_vertices_O: Sequence[Tensor],
        template_mesh_faces: Sequence[Tensor],
        pose_rank_valid: Tensor | None = None,
    ) -> dict[str, Any]:
        del template_surface_points_O, template_valid_mask  # V2 uses its exact selected face.
        reconstructed = torch.einsum(
            "bij,bnj->bni", predicted_pose_T_C_from_O[:, :3, :3], predicted_points_O
        ) + predicted_pose_T_C_from_O[:, None, :3, 3]
        alignment_distance = torch.linalg.vector_norm(reconstructed - observed_points_C, dim=-1)

        names = (
            "patch_ce", "local_fine", "barycentric", "corr_mean", "corr_tail", "rotation",
            "translation", "align_mean", "align_tail", "surface", "local_rigidity",
            "covariance", "min_eigenvalue", "patch_diversity",
            "fine_coordinate_aux",
            "fine_coordinate_aux_tail",
            "raw_aux_rotation", "raw_aux_translation", "raw_aux_alignment",
        )
        selected_values: dict[str, list[Tensor]] = {name: [] for name in names}
        selected_raw: dict[str, list[Tensor]] = {name: [] for name in names}
        selected_totals: list[Tensor] = []
        totals_by_element: list[Tensor] = []
        corr_by_element: list[Tensor] = []
        rot_by_element: list[Tensor] = []
        trans_by_element: list[Tensor] = []
        selected_indices: list[int] = []
        matched_targets: list[Tensor] = []
        matched_poses: list[Tensor] = []
        surface_rows: list[Tensor] = []
        predicted_eigenvalues: list[Tensor] = []
        target_eigenvalues: list[Tensor] = []
        rank_invalid_rows: list[Tensor] = []
        covariance_active_rows: list[Tensor] = []
        eigenvalue_ratio_rows: list[Tensor] = []
        rank_margin_rows: list[Tensor] = []
        triangle_target_mismatch_rows: list[Tensor] = []
        auxiliary_coordinate_distance_rows: list[Tensor] = []

        for batch_index in range(len(predicted_points_O)):
            mask = valid_mask[batch_index]
            if not bool(mask.any()):
                raise ValueError("V3 loss requires at least one valid observed point")
            metadata = symmetry_metadata[batch_index]
            group = parse_rotation_group(effective_symmetry_groups[batch_index])
            symmetries = symmetry_transforms(
                group,
                metadata.axis.direction,
                metadata.axis.origin,
                so2_num_samples=self.so2_samples if group.type == "SO2" else None,
                dtype=predicted_points_O.dtype,
                device=predicted_points_O.device,
            )
            targets = transform_points(
                invert_transform(symmetries), target_points_O[batch_index].unsqueeze(0)
            )
            equivalent_poses = gt_pose_T_C_from_O[batch_index].unsqueeze(0) @ symmetries
            q = predicted_points_O[batch_index, mask]
            p = observed_points_C[batch_index, mask]
            target_rows = targets[:, mask]

            corr_distance = torch.linalg.vector_norm(q.unsqueeze(0) - target_rows, dim=-1)
            corr_mean_raw = corr_distance.mean(-1)
            corr_tail_raw = torch.stack(
                [top_tail_mean(row, self.tail_fraction) for row in corr_distance]
            )
            rotation_raw = rotation_geodesic_distance(
                predicted_pose_T_C_from_O[batch_index, :3, :3].unsqueeze(0),
                equivalent_poses[:, :3, :3],
            )
            translation_raw = torch.linalg.vector_norm(
                predicted_pose_T_C_from_O[batch_index, :3, 3].unsqueeze(0)
                - equivalent_poses[:, :3, 3], dim=-1,
            )
            align_points = alignment_distance[batch_index, mask]
            align_mean_raw = align_points.mean()
            align_tail_raw = top_tail_mean(align_points, self.tail_fraction)
            local_raw = local_rigidity_errors(q, p, self.local_rigidity_k).mean()

            q_centered = q - q.mean(0)
            q_covariance = q_centered.transpose(0, 1) @ q_centered / max(len(q) - 1, 1)
            q_eigenvalues = torch.linalg.eigvalsh(q_covariance).clamp_min(0)
            target_covariances = []
            target_eigenvalue_rows = []
            for target in target_rows:
                centered = target - target.mean(0)
                covariance = centered.transpose(0, 1) @ centered / max(len(target) - 1, 1)
                target_covariances.append(covariance)
                target_eigenvalue_rows.append(torch.linalg.eigvalsh(covariance).clamp_min(0))
            target_eigenvalue_tensor = torch.stack(target_eigenvalue_rows)
            covariance_unmasked = torch.stack(
                [torch.linalg.matrix_norm(q_covariance - value) for value in target_covariances]
            )
            conditional = conditional_covariance_penalty(
                covariance_unmasked,
                q_eigenvalues,
                target_eigenvalue_tensor,
                min_eigenvalue_m2=self.min_eigenvalue_m2,
                minimum_eigenvalue_ratio=self.covariance_minimum_eigenvalue_ratio,
            )
            covariance_raw = (
                conditional["penalty"]
                if self.conditional_covariance
                else covariance_unmasked
            )
            covariance_normalized = _normalized(
                covariance_raw, self.scales["corr"] ** 2
            )
            threshold = max(self.min_eigenvalue_m2, 1e-12)
            min_eigenvalue_raw = (threshold - q_eigenvalues[0]).clamp_min(0)
            min_eigenvalue_normalized = (min_eigenvalue_raw / threshold).expand(
                len(symmetries)
            )

            vertices = template_mesh_vertices_O[batch_index].to(q)
            faces = template_mesh_faces[batch_index].to(device=q.device, dtype=torch.long)
            auxiliary_normalized = correspondence_auxiliary.get(
                "fine_aux_coordinate_normalized"
            )
            if auxiliary_normalized is None:
                if (
                    self.weights["fine_coordinate_aux"] != 0.0
                    or self.weights["fine_coordinate_aux_tail"] != 0.0
                ):
                    raise ValueError(
                        "fine_coordinate_aux_weight requires "
                        "FineCanonicalCoordinateAuxiliaryHead"
                    )
                auxiliary_raw = q.new_zeros((len(symmetries),))
                auxiliary_tail_raw = q.new_zeros((len(symmetries),))
                auxiliary_distance = q.new_zeros((len(symmetries), len(q)))
                aux_rotation_raw = q.new_zeros((len(symmetries),))
                aux_translation_raw = q.new_zeros((len(symmetries),))
                aux_alignment_raw = q.new_zeros((len(symmetries),))
            else:
                predicted_normalized = auxiliary_normalized[batch_index, mask]
                bbox_min = vertices.amin(0)
                bbox_max = vertices.amax(0)
                bbox_extent = (bbox_max - bbox_min).clamp_min(1e-8)
                target_normalized = (
                    2.0 * (target_rows - bbox_min) / bbox_extent - 1.0
                )
                auxiliary_raw, auxiliary_tail_raw = coordinate_mean_and_tail_loss(
                    predicted_normalized, target_normalized, .10
                )
                decoded_auxiliary = (
                    0.5 * (predicted_normalized + 1.0) * bbox_extent + bbox_min
                )
                auxiliary_distance = torch.linalg.vector_norm(
                    decoded_auxiliary.unsqueeze(0) - target_rows, dim=-1
                )
                aux_solution = self.aux_procrustes.solve(
                    decoded_auxiliary.unsqueeze(0), p.unsqueeze(0),
                    decoded_auxiliary.new_ones((1, len(decoded_auxiliary))),
                    torch.ones((1, len(decoded_auxiliary)), dtype=torch.bool, device=q.device),
                )
                aux_pose = aux_solution["transform"][0]
                aux_rotation_raw = rotation_geodesic_distance(
                    aux_pose[:3, :3].unsqueeze(0), equivalent_poses[:, :3, :3]
                )
                aux_translation_raw = torch.linalg.vector_norm(
                    aux_pose[:3, 3].unsqueeze(0) - equivalent_poses[:, :3, 3], dim=-1
                )
                aux_reconstructed = transform_points(
                    aux_pose.unsqueeze(0), decoded_auxiliary.unsqueeze(0)
                )[0]
                aux_alignment_raw = torch.linalg.vector_norm(
                    aux_reconstructed - p, dim=-1
                ).mean().expand(len(symmetries))
            chosen_faces = correspondence_auxiliary["selected_triangle_ids"][batch_index, mask]
            chosen_triangles = vertices[faces[chosen_faces]]
            barycentric = correspondence_auxiliary["predicted_barycentric"][batch_index, mask]
            same_face_point = (barycentric[..., None] * chosen_triangles).sum(1)
            surface_points = torch.linalg.vector_norm(q - same_face_point, dim=-1)
            if float(surface_points.detach().max()) > 1e-5:
                raise AssertionError(
                    "surface-constrained head output is not on its selected triangle"
                )
            surface_rows.append(surface_points)
            surface_raw = surface_points.mean()

            patch_points = correspondence_auxiliary["patch_points_O"][batch_index]
            coarse_logits = correspondence_auxiliary["coarse_patch_logits"][batch_index, mask]
            fine_logits = correspondence_auxiliary["fine_local_logits"][batch_index, mask]
            centroids = vertices[faces].mean(1)
            candidate_ids = correspondence_auxiliary["candidate_triangle_ids"][batch_index, mask]
            candidate_mask = correspondence_auxiliary.get("candidate_triangle_mask")
            candidate_mask = (
                candidate_ids.ge(0)
                if candidate_mask is None
                else candidate_mask[batch_index, mask]
            )
            candidate_centroids = centroids[candidate_ids.clamp_min(0)]
            all_candidate_ids = correspondence_auxiliary.get(
                "all_candidate_triangle_ids"
            )
            face_owner_ids = correspondence_auxiliary.get("face_owner_patch_ids")
            patch_ce_rows: list[Tensor] = []
            fine_ce_rows: list[Tensor] = []
            barycentric_loss_rows: list[Tensor] = []
            patch_diversity_rows: list[Tensor] = []
            target_mismatch_by_element: list[Tensor] = []
            predicted_occupancy = torch.softmax(coarse_logits, -1).mean(0)
            forced_elements = correspondence_auxiliary.get(
                "teacher_forcing_selected_symmetry_element"
            )
            forced_selected: int | None = None
            if self.use_teacher_forcing_shared_symmetry_element:
                if forced_elements is None:
                    raise AssertionError(
                        "shared-S local loss requires teacher-forcing symmetry metadata"
                    )
                forced_selected = int(forced_elements[batch_index])
                if not 0 <= forced_selected < len(target_rows):
                    raise AssertionError("invalid teacher-forcing symmetry element")
            cached_valid_global = correspondence_auxiliary.get(
                "teacher_forcing_valid_triangle_global_mask"
            )
            cached_gt_triangle = correspondence_auxiliary.get(
                "teacher_forcing_gt_triangle_ids"
            )
            for element_index, target in enumerate(target_rows):
                if forced_selected is not None and element_index != forced_selected:
                    # Candidate construction belongs to one shared S.  Local
                    # losses for other S values cannot be meaningful and are
                    # never selected, so avoid repeating the expensive exact
                    # point-to-all-triangles projection for them.
                    zero = fine_logits.sum() * 0.0
                    patch_ce_rows.append(zero)
                    fine_ce_rows.append(zero)
                    barycentric_loss_rows.append(zero)
                    patch_diversity_rows.append(zero)
                    target_mismatch_by_element.append(zero + 1.0)
                    continue
                if (
                    forced_selected is not None
                    and cached_valid_global is not None
                    and cached_gt_triangle is not None
                ):
                    gt_triangle = cached_gt_triangle[batch_index, mask]
                    valid_triangle_global = cached_valid_global[batch_index, mask]
                else:
                    target_set = triangle_target_sets(
                        target,
                        vertices,
                        faces,
                        tolerance_m=self.triangle_target_tolerance_m,
                        point_chunk_size=256,
                    )
                    gt_triangle = target_set["face_ids"]
                    valid_triangle_global = target_set["valid_triangle_mask"]
                selected_face_target = closest_barycentric_on_triangles(
                    target, chosen_triangles
                )
                barycentric_loss_rows.append(
                    F.smooth_l1_loss(
                        barycentric,
                        selected_face_target["barycentric"].to(barycentric),
                        reduction="none",
                    ).sum(-1).mean()
                )
                if all_candidate_ids is not None:
                    valid_targets = valid_patch_mask(
                        gt_triangle, all_candidate_ids[batch_index]
                    )
                    patch_target = (
                        single_owner_patch_ids(gt_triangle, face_owner_ids[batch_index])
                        if face_owner_ids is not None
                        else valid_targets.to(torch.int64).argmax(-1)
                    )
                else:
                    patch_target = torch.cdist(
                        target.float(), patch_points.float()
                    ).argmin(-1)
                    valid_targets = F.one_hot(
                        patch_target, num_classes=coarse_logits.shape[-1]
                    ).bool()
                if self.patch_target_mode == "multi_valid_patch_set":
                    patch_ce_rows.append(
                        multi_positive_softmax_loss(coarse_logits, valid_targets)
                    )
                else:
                    patch_ce_rows.append(F.cross_entropy(coarse_logits, patch_target))
                target_occupancy = torch.bincount(
                    patch_target, minlength=coarse_logits.shape[-1]
                ).to(coarse_logits.dtype)
                target_occupancy = target_occupancy / target_occupancy.sum().clamp_min(1)
                patch_diversity_rows.append(
                    (predicted_occupancy - target_occupancy).square().mean()
                    * coarse_logits.shape[-1]
                )
                exact_fine_valid = candidate_ids.eq(gt_triangle[:, None]) & candidate_mask
                fine_valid = (
                    local_valid_triangle_mask(
                        candidate_ids, valid_triangle_global
                    )
                    & candidate_mask
                    if self.triangle_target_mode == "multi_valid_patch_set"
                    else exact_fine_valid
                )
                target_mismatch_by_element.append(
                    (~fine_valid.any(-1)).float().mean()
                )
                has_valid = fine_valid.any(-1)
                if bool(has_valid.all()):
                    fine_ce_rows.append(
                        multi_positive_softmax_loss(fine_logits, fine_valid)
                    )
                else:
                    # Predicted-only Stage A may miss the exact triangle.  Keep a
                    # finite diagnostic/local objective; Stage B probability=1
                    # guarantees the exact branch and therefore takes the path above.
                    fine_target_distance = torch.linalg.vector_norm(
                        candidate_centroids - target[:, None], dim=-1
                    ).masked_fill(~candidate_mask, float("inf"))
                    fine_target = fine_target_distance.argmin(-1)
                    fine_ce_rows.append(F.cross_entropy(fine_logits, fine_target))
            patch_ce_raw = torch.stack(patch_ce_rows)
            fine_ce_raw = torch.stack(fine_ce_rows)
            barycentric_raw = torch.stack(barycentric_loss_rows)
            patch_diversity_raw = torch.stack(patch_diversity_rows)

            rank_valid = (
                torch.as_tensor(True, device=q.device)
                if pose_rank_valid is None
                else pose_rank_valid[batch_index].to(device=q.device, dtype=torch.bool)
            )
            rank_invalid_rows.append(~rank_valid)
            pose_multiplier = rank_valid.to(q.dtype)

            normalized = {
                "patch_ce": patch_ce_raw,
                "local_fine": fine_ce_raw,
                "barycentric": barycentric_raw,
                "corr_mean": _normalized(corr_mean_raw, self.scales["corr"]),
                "corr_tail": _normalized(corr_tail_raw, self.scales["corr"]),
                "rotation": _normalized(rotation_raw, self.scales["rot"]) * pose_multiplier,
                "translation": _normalized(translation_raw, self.scales["trans"]) * pose_multiplier,
                "align_mean": _normalized(align_mean_raw, self.scales["align"]).expand(len(symmetries)),
                "align_tail": _normalized(align_tail_raw, self.scales["align"]).expand(len(symmetries)),
                "surface": _normalized(surface_raw, self.scales["surface"]).expand(len(symmetries)),
                "local_rigidity": _normalized(local_raw, self.scales["local"]).expand(len(symmetries)),
                "covariance": covariance_normalized,
                "min_eigenvalue": min_eigenvalue_normalized,
                "patch_diversity": patch_diversity_raw,
                "fine_coordinate_aux": auxiliary_raw,
                "fine_coordinate_aux_tail": auxiliary_tail_raw,
                "raw_aux_rotation": _normalized(aux_rotation_raw, self.scales["raw_aux_rot"]),
                "raw_aux_translation": _normalized(aux_translation_raw, self.scales["raw_aux_trans"]),
                "raw_aux_alignment": _normalized(aux_alignment_raw, self.scales["raw_aux_align"]),
            }
            raw = {
                "patch_ce": patch_ce_raw,
                "local_fine": fine_ce_raw,
                "barycentric": barycentric_raw,
                "corr_mean": corr_mean_raw,
                "corr_tail": corr_tail_raw,
                "rotation": rotation_raw,
                "translation": translation_raw,
                "align_mean": align_mean_raw.expand(len(symmetries)),
                "align_tail": align_tail_raw.expand(len(symmetries)),
                "surface": surface_raw.expand(len(symmetries)),
                "local_rigidity": local_raw.expand(len(symmetries)),
                "covariance": covariance_raw,
                "min_eigenvalue": min_eigenvalue_raw.expand(len(symmetries)),
                "patch_diversity": patch_diversity_raw,
                "fine_coordinate_aux": auxiliary_raw,
                "fine_coordinate_aux_tail": auxiliary_tail_raw,
                "raw_aux_rotation": aux_rotation_raw,
                "raw_aux_translation": aux_translation_raw,
                "raw_aux_alignment": aux_alignment_raw,
            }
            total = sum(self.weights[name] * normalized[name] for name in names)
            if self.use_teacher_forcing_shared_symmetry_element:
                assert forced_selected is not None
                selected = forced_selected
            else:
                selected = int(total.argmin().detach())
            selected_target_mismatch = torch.stack(target_mismatch_by_element)[
                selected
            ]
            if self.require_exact_triangle_candidate and bool(
                selected_target_mismatch > 0
            ):
                raise AssertionError(
                    "candidate_global_ids[local_target_index] != exact GT triangle "
                    "for the selected shared symmetry element"
                )
            selected_indices.append(selected)
            selected_totals.append(total[selected])
            totals_by_element.append(total)
            corr_by_element.append(corr_mean_raw * 1000.0)
            rot_by_element.append(torch.rad2deg(rotation_raw))
            trans_by_element.append(translation_raw * 1000.0)
            matched_targets.append(targets[selected])
            matched_poses.append(equivalent_poses[selected])
            predicted_eigenvalues.append(q_eigenvalues)
            target_eigenvalues.append(target_eigenvalue_tensor[selected])
            covariance_active_rows.append(conditional["active"][selected])
            eigenvalue_ratio_rows.append(
                conditional["eigenvalue_ratios"][selected]
            )
            rank_margin_rows.append(conditional["rank_margin_m2"])
            triangle_target_mismatch_rows.append(
                selected_target_mismatch
            )
            auxiliary_coordinate_distance_rows.append(auxiliary_distance[selected])
            for name in names:
                selected_values[name].append(normalized[name][selected])
                selected_raw[name].append(raw[name][selected])

        values = {name: torch.stack(rows).mean() for name, rows in selected_values.items()}
        raw_values = {name: torch.stack(rows).mean() for name, rows in selected_raw.items()}
        selected_tensor = torch.tensor(selected_indices, dtype=torch.long, device=predicted_points_O.device)
        result: dict[str, Any] = {
            "loss_total": per_sample_mean_then_batch_mean(selected_totals),
            "per_sample_loss_total": torch.stack(selected_totals),
            "selected_shared_symmetry_element": selected_tensor,
            "selected_shared_symmetry_element_mean": selected_tensor.float().mean(),
            "v3_loss_by_symmetry_element": _pad(totals_by_element),
            "loss_by_symmetry_element": _pad(totals_by_element),
            "correspondence_error_by_symmetry_element_mm": _pad(corr_by_element),
            "rotation_error_by_symmetry_element_deg": _pad(rot_by_element),
            "translation_error_by_symmetry_element_mm": _pad(trans_by_element),
            "matched_target_points_O": torch.stack(matched_targets),
            "matched_gt_pose_T_C_from_O": torch.stack(matched_poses),
            "alignment_distance_m": alignment_distance,
            "template_surface_distance_m": torch.nn.utils.rnn.pad_sequence(
                surface_rows, batch_first=True
            ),
            "reconstructed_visible_points_C": reconstructed,
            "predicted_covariance_eigenvalues": torch.stack(predicted_eigenvalues),
            "gt_covariance_eigenvalues": torch.stack(target_eigenvalues),
            "eigenvalue_ratio": torch.stack(predicted_eigenvalues)
            / torch.stack(target_eigenvalues).clamp_min(1e-12),
            "correspondence_rank": torch.stack(
                [(row > self.min_eigenvalue_m2).sum() for row in predicted_eigenvalues]
            ),
            "rank_invalid_fraction": torch.stack(rank_invalid_rows).float().mean(),
            "covariance_penalty_active": torch.stack(
                covariance_active_rows
            ).float().mean(),
            "selected_eigenvalue_ratios": torch.stack(eigenvalue_ratio_rows),
            "rank_margin_m2": torch.stack(rank_margin_rows),
            "triangle_target_index_mismatch_fraction": torch.stack(
                triangle_target_mismatch_rows
            ).mean(),
            "aux_coordinate_distance_m": torch.nn.utils.rnn.pad_sequence(
                auxiliary_coordinate_distance_rows, batch_first=True
            ),
        }
        auxiliary_distances = torch.cat(auxiliary_coordinate_distance_rows)
        result["aux_coordinate_rmse_mm"] = (
            auxiliary_distances.square().mean().sqrt() * 1000.0
        )
        result["aux_coordinate_p95_mm"] = (
            torch.quantile(auxiliary_distances.float(), 0.95).to(predicted_points_O)
            * 1000.0
        )
        for name in names:
            result[f"loss_{name}_normalized"] = values[name]
            result[f"raw_{name}"] = raw_values[name]
            result[f"weighted_loss_{name}"] = self.weights[name] * values[name]
            result[f"per_sample_loss_{name}_normalized"] = torch.stack(
                selected_values[name]
            )
            result[f"per_sample_raw_{name}"] = torch.stack(selected_raw[name])
            result[f"per_sample_weighted_loss_{name}"] = (
                self.weights[name] * torch.stack(selected_values[name])
            )
        # Stable aliases consumed by existing logging/evaluation code.
        result.update(
            loss_correspondence_normalized=values["corr_mean"],
            loss_rotation_normalized=values["rotation"],
            loss_translation_normalized=values["translation"],
            loss_alignment_normalized=values["align_mean"],
            loss_template_surface_normalized=values["surface"],
            weighted_loss_correspondence=self.weights["corr_mean"] * values["corr_mean"],
            weighted_loss_rotation=self.weights["rotation"] * values["rotation"],
            weighted_loss_translation=self.weights["translation"] * values["translation"],
            weighted_loss_alignment=self.weights["align_mean"] * values["align_mean"],
            weighted_loss_template_surface=self.weights["surface"] * values["surface"],
            raw_correspondence_error_m=raw_values["corr_mean"],
            raw_rotation_error_rad=raw_values["rotation"],
            raw_translation_error_m=raw_values["translation"],
            raw_alignment_error_m=raw_values["align_mean"],
            raw_template_surface_error_m=raw_values["surface"],
            rotation_error_deg=torch.rad2deg(raw_values["rotation"]),
            translation_total_mm=raw_values["translation"] * 1000.0,
        )
        return result


__all__ = [
    "JointSurfaceCorrespondencePoseLossV3",
    "per_sample_mean_then_batch_mean",
    "covariance_collapse_penalties",
    "top_tail_mean",
]
