"""Active-only scratch loss for clean coordinate registration V3."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from symm_template_reg.models.pose.pose_representation import invert_transform, transform_points
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.pose.weighted_procrustes import WeightedProcrustes
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.registry import LOSSES


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
    per_point = F.smooth_l1_loss(
        predicted_normalized, target_normalized, reduction="none"
    ).mean(-1)
    count = max(1, math.ceil(per_point.shape[-1] * float(tail_fraction)))
    return per_point.mean(-1), per_point.topk(count, dim=-1).values.mean(-1)


def per_sample_mean_then_batch_mean(sample_losses: Sequence[Tensor]) -> Tensor:
    """Give every sample equal weight after its own point-wise reduction."""

    if not sample_losses:
        raise ValueError("at least one sample loss is required")
    return torch.stack(
        [value.mean() if value.ndim else value for value in sample_losses]
    ).mean()


def scratch_warmup_progress(epoch: int, warmup_epochs: int = 250) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(max(float(epoch), 0.0) / float(warmup_epochs), 1.0)


def _normalized(raw: Tensor, scale: float) -> Tensor:
    value = raw / max(float(scale), 1e-12)
    return F.smooth_l1_loss(value, torch.zeros_like(value), reduction="none")


@LOSSES.register_module()
class CleanCoordinatePoseLossV3(nn.Module):
    """Equal-sample raw-q loss with one shared symmetry element per sample."""

    def __init__(
        self,
        coordinate_mean_weight: float = 1.0,
        coordinate_tail_weight: float = 0.5,
        pose_rotation_weight: float = 0.25,
        pose_translation_weight: float = 0.25,
        alignment_weight: float = 0.25,
        rotation_scale_deg: float = 1.0,
        translation_scale_m: float = 0.001,
        alignment_scale_m: float = 0.001,
        warmup_epochs: int = 250,
        current_epoch: int = 0,
        tail_fraction: float = 0.10,
        so2_samples: int = 36,
        loss_reduction: str = "per_sample_mean_then_batch_mean",
        vectorized: bool = False,
    ) -> None:
        super().__init__()
        if loss_reduction != "per_sample_mean_then_batch_mean":
            raise ValueError("clean V3 requires per_sample_mean_then_batch_mean")
        self.base_weights = {
            "fine_coordinate_aux": float(coordinate_mean_weight),
            "fine_coordinate_aux_tail": float(coordinate_tail_weight),
            "raw_aux_rotation": float(pose_rotation_weight),
            "raw_aux_translation": float(pose_translation_weight),
            "raw_aux_alignment": float(alignment_weight),
        }
        self.scales = {
            "rotation_rad": math.radians(float(rotation_scale_deg)),
            "translation_m": float(translation_scale_m),
            "alignment_m": float(alignment_scale_m),
        }
        self.warmup_epochs = int(warmup_epochs)
        self.current_epoch = int(current_epoch)
        self.tail_fraction = float(tail_fraction)
        self.so2_samples = int(so2_samples)
        self.procrustes = WeightedProcrustes()
        self.vectorized = bool(vectorized)

    @property
    def warmup_progress(self) -> float:
        return scratch_warmup_progress(self.current_epoch, self.warmup_epochs)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        predicted_normalized_O: Tensor,
        observed_points_C: Tensor,
        target_points_O: Tensor,
        valid_mask: Tensor,
        gt_pose_T_C_from_O: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
        template_mesh_vertices_O: Sequence[Tensor],
    ) -> dict[str, Any]:
        if self.vectorized:
            return self._forward_vectorized(
                predicted_normalized_O, observed_points_C, target_points_O,
                valid_mask, gt_pose_T_C_from_O, symmetry_metadata,
                effective_symmetry_groups, template_mesh_vertices_O,
            )
        names = tuple(self.base_weights)
        selected_total: list[Tensor] = []
        selected_normalized = {name: [] for name in names}
        selected_raw = {name: [] for name in names}
        selected_indices: list[int] = []
        matched_targets, matched_poses = [], []
        decoded_rows, alignment_rows, rank_rows = [], [], []
        totals_by_element, rotations_by_element, translations_by_element = [], [], []
        correspondence_by_element, selected_distance_rows = [], []
        full_weight_counterfactual_rows: list[Tensor] = []
        predicted_eigenvalues, target_eigenvalues = [], []
        progress = self.warmup_progress
        weights = dict(self.base_weights)
        for name in ("raw_aux_rotation", "raw_aux_translation", "raw_aux_alignment"):
            weights[name] *= progress

        # The solver is intrinsically batched.  Running one 3x3 SVD call per
        # sample produced slightly different CUDA round-off and needless
        # launches; both the reference loop and vectorized symmetry path now
        # share this exact batched pose solve.
        bbox_min_all = torch.stack([
            vertices.to(predicted_normalized_O).amin(0)
            for vertices in template_mesh_vertices_O
        ])
        bbox_max_all = torch.stack([
            vertices.to(predicted_normalized_O).amax(0)
            for vertices in template_mesh_vertices_O
        ])
        extent_all = (bbox_max_all - bbox_min_all).clamp_min(1e-8)
        decoded_all = (
            0.5 * (predicted_normalized_O + 1.0) * extent_all[:, None]
            + bbox_min_all[:, None]
        )
        batched_solution = self.procrustes.solve(
            decoded_all.float(), observed_points_C.float(),
            valid_mask.to(decoded_all.dtype), valid_mask.bool(),
        )

        for index in range(len(predicted_normalized_O)):
            mask = valid_mask[index].bool()
            qn = predicted_normalized_O[index, mask]
            observed = observed_points_C[index, mask]
            metadata = symmetry_metadata[index]
            group = parse_rotation_group(effective_symmetry_groups[index])
            symmetries = symmetry_transforms(
                group, metadata.axis.direction, metadata.axis.origin,
                so2_num_samples=self.so2_samples if group.type == "SO2" else None,
                dtype=qn.dtype, device=qn.device,
            )
            targets = transform_points(
                invert_transform(symmetries), target_points_O[index].unsqueeze(0)
            )
            poses = gt_pose_T_C_from_O[index].unsqueeze(0) @ symmetries
            vertices = template_mesh_vertices_O[index].to(qn)
            bbox_min, bbox_max = vertices.amin(0), vertices.amax(0)
            extent = (bbox_max - bbox_min).clamp_min(1e-8)
            target_normalized = 2.0 * (targets[:, mask] - bbox_min) / extent - 1.0
            coordinate, tail = coordinate_mean_and_tail_loss(
                qn, target_normalized, self.tail_fraction
            )
            decoded = decoded_all[index, mask]
            pose = batched_solution["transform"][index].to(decoded)
            rotation_raw = rotation_geodesic_distance(
                pose[:3, :3][None], poses[:, :3, :3]
            )
            translation_raw = torch.linalg.vector_norm(
                pose[:3, 3][None] - poses[:, :3, 3], dim=-1
            )
            reconstructed = transform_points(pose[None], decoded[None])[0]
            alignment_distance = torch.linalg.vector_norm(
                reconstructed - observed, dim=-1
            )
            alignment_raw = alignment_distance.mean().expand(len(symmetries))
            normalized = {
                "fine_coordinate_aux": coordinate,
                "fine_coordinate_aux_tail": tail,
                "raw_aux_rotation": _normalized(rotation_raw, self.scales["rotation_rad"]),
                "raw_aux_translation": _normalized(translation_raw, self.scales["translation_m"]),
                "raw_aux_alignment": _normalized(alignment_raw, self.scales["alignment_m"]),
            }
            raw = {
                "fine_coordinate_aux": coordinate,
                "fine_coordinate_aux_tail": tail,
                "raw_aux_rotation": rotation_raw,
                "raw_aux_translation": translation_raw,
                "raw_aux_alignment": alignment_raw,
            }
            totals = sum(weights[name] * normalized[name] for name in names)
            selected = int(totals.detach().argmin())
            selected_total.append(totals[selected])
            full_weight_counterfactual_rows.append(
                sum(self.base_weights[name] * normalized[name] for name in names)[selected]
            )
            totals_by_element.append(totals)
            rotations_by_element.append(torch.rad2deg(rotation_raw))
            translations_by_element.append(translation_raw * 1000.0)
            distances = torch.linalg.vector_norm(
                decoded.unsqueeze(0) - targets[:, mask], dim=-1
            )
            correspondence_by_element.append(distances.mean(-1) * 1000.0)
            selected_distance_rows.append(distances[selected])
            selected_indices.append(selected)
            matched_targets.append(targets[selected])
            matched_poses.append(poses[selected])
            decoded_rows.append(decoded)
            alignment_rows.append(alignment_distance)
            rank_rows.append(batched_solution["rank"][index])
            def eigenvalues(points: Tensor) -> Tensor:
                centered = points - points.mean(0)
                covariance = centered.T @ centered / max(len(points) - 1, 1)
                return torch.linalg.eigvalsh(covariance.float()).clamp_min(0).to(points)
            predicted_eigenvalues.append(eigenvalues(decoded))
            target_eigenvalues.append(eigenvalues(targets[selected, mask]))
            for name in names:
                selected_normalized[name].append(normalized[name][selected])
                selected_raw[name].append(raw[name][selected])

        aggregate = {
            name: torch.stack(values).mean()
            for name, values in selected_normalized.items()
        }
        raw_aggregate = {
            name: torch.stack(values).mean()
            for name, values in selected_raw.items()
        }
        total_tensor = torch.stack(selected_total)
        def pad(rows: list[Tensor]) -> Tensor:
            width = max(row.numel() for row in rows)
            value = rows[0].new_full((len(rows), width), float("nan"))
            for row_index, row in enumerate(rows):
                value[row_index, :row.numel()] = row
            return value
        predicted_eigenvalue_tensor = torch.stack(predicted_eigenvalues)
        target_eigenvalue_tensor = torch.stack(target_eigenvalues)
        all_selected_distances = torch.cat(selected_distance_rows)
        result: dict[str, Any] = {
            "loss_total": per_sample_mean_then_batch_mean(selected_total),
            "per_sample_loss_total": total_tensor,
            "selected_shared_symmetry_element": torch.tensor(
                selected_indices, dtype=torch.long, device=total_tensor.device
            ),
            "matched_target_points_O": torch.stack(matched_targets),
            "matched_gt_pose_T_C_from_O": torch.stack(matched_poses),
            "alignment_distance_m": torch.nn.utils.rnn.pad_sequence(
                alignment_rows, batch_first=True
            ),
            "decoded_q_aux_O": torch.nn.utils.rnn.pad_sequence(
                decoded_rows, batch_first=True
            ),
            "correspondence_rank": torch.stack(rank_rows),
            "loss_by_symmetry_element": pad(totals_by_element),
            "rotation_error_by_symmetry_element_deg": pad(rotations_by_element),
            "translation_error_by_symmetry_element_mm": pad(translations_by_element),
            "correspondence_error_by_symmetry_element_mm": pad(correspondence_by_element),
            "predicted_covariance_eigenvalues": predicted_eigenvalue_tensor,
            "gt_covariance_eigenvalues": target_eigenvalue_tensor,
            "selected_eigenvalue_ratios": (
                predicted_eigenvalue_tensor / target_eigenvalue_tensor.clamp_min(1e-12)
            ),
            "rank_margin_m2": predicted_eigenvalue_tensor[:, 0] - 1e-6,
            "aux_coordinate_rmse_mm": all_selected_distances.square().mean().sqrt() * 1000.0,
            "aux_coordinate_p95_mm": torch.quantile(all_selected_distances.float(), 0.95).to(total_tensor) * 1000.0,
            "warmup_progress": total_tensor.new_tensor(progress),
            "actual_optimized_loss": per_sample_mean_then_batch_mean(selected_total),
            "full_weight_counterfactual_loss": torch.stack(
                full_weight_counterfactual_rows
            ).mean(),
            "current_pose_loss_weight": total_tensor.new_tensor(
                self.base_weights["raw_aux_rotation"] * progress
            ),
            "current_alignment_loss_weight": total_tensor.new_tensor(
                self.base_weights["raw_aux_alignment"] * progress
            ),
        }
        for name in names:
            per_normalized = torch.stack(selected_normalized[name])
            per_raw = torch.stack(selected_raw[name])
            result[f"loss_{name}_normalized"] = aggregate[name]
            result[f"raw_{name}"] = raw_aggregate[name]
            result[f"weighted_loss_{name}"] = weights[name] * aggregate[name]
            result[f"per_sample_loss_{name}_normalized"] = per_normalized
            result[f"per_sample_raw_{name}"] = per_raw
            result[f"per_sample_weighted_loss_{name}"] = weights[name] * per_normalized
            result[f"current_weight_{name}"] = total_tensor.new_tensor(weights[name])
            result[f"final_weight_{name}"] = total_tensor.new_tensor(self.base_weights[name])
        result["weighted_loss_rotation"] = result["weighted_loss_raw_aux_rotation"]
        result["weighted_loss_translation"] = result["weighted_loss_raw_aux_translation"]
        result["rotation_error_deg"] = torch.rad2deg(raw_aggregate["raw_aux_rotation"])
        result["translation_total_mm"] = raw_aggregate["raw_aux_translation"] * 1000.0
        return result

    def _forward_vectorized(
        self,
        predicted_normalized_O: Tensor,
        observed_points_C: Tensor,
        target_points_O: Tensor,
        valid_mask: Tensor,
        gt_pose_T_C_from_O: Tensor,
        symmetry_metadata: Sequence[Any],
        effective_symmetry_groups: Sequence[Any],
        template_mesh_vertices_O: Sequence[Tensor],
    ) -> dict[str, Any]:
        """Exact mixed-C2/C4 tensor path; only metadata packing uses Python."""

        names = tuple(self.base_weights)
        batch_size, point_count = valid_mask.shape
        symmetry_rows = []
        for metadata, raw_group in zip(symmetry_metadata, effective_symmetry_groups):
            group = parse_rotation_group(raw_group)
            symmetry_rows.append(symmetry_transforms(
                group, metadata.axis.direction, metadata.axis.origin,
                so2_num_samples=self.so2_samples if group.type == "SO2" else None,
                dtype=predicted_normalized_O.dtype,
                device=predicted_normalized_O.device,
            ))
        symmetry_count = max(len(row) for row in symmetry_rows)
        identity = torch.eye(
            4, dtype=predicted_normalized_O.dtype,
            device=predicted_normalized_O.device,
        )
        symmetries = identity.expand(batch_size, symmetry_count, 4, 4).clone()
        symmetry_valid = torch.zeros(
            (batch_size, symmetry_count), dtype=torch.bool,
            device=predicted_normalized_O.device,
        )
        for row_index, row in enumerate(symmetry_rows):
            symmetries[row_index, : len(row)] = row
            symmetry_valid[row_index, : len(row)] = True

        inverse = invert_transform(symmetries)
        targets = transform_points(inverse, target_points_O[:, None])
        poses = gt_pose_T_C_from_O[:, None] @ symmetries
        bbox_min = torch.stack([
            vertices.to(predicted_normalized_O).amin(0)
            for vertices in template_mesh_vertices_O
        ])
        bbox_max = torch.stack([
            vertices.to(predicted_normalized_O).amax(0)
            for vertices in template_mesh_vertices_O
        ])
        extent = (bbox_max - bbox_min).clamp_min(1e-8)
        target_normalized = (
            2.0 * (targets - bbox_min[:, None, None])
            / extent[:, None, None] - 1.0
        )
        per_coordinate = F.smooth_l1_loss(
            predicted_normalized_O[:, None].expand_as(target_normalized),
            target_normalized, reduction="none",
        )
        per_point_coordinate = per_coordinate.mean(-1)
        point_mask = valid_mask[:, None]
        lengths = valid_mask.sum(-1).clamp_min(1)
        coordinate = (
            per_point_coordinate * point_mask
        ).sum(-1) / lengths[:, None]
        sorted_coordinate = per_point_coordinate.masked_fill(
            ~point_mask, float("-inf")
        ).sort(-1, descending=True).values
        tail_counts = torch.ceil(lengths.float() * self.tail_fraction).long().clamp_min(1)
        tail_prefix = sorted_coordinate.clamp_min(0).cumsum(-1)
        tail = tail_prefix.gather(
            -1,
            (tail_counts[:, None, None] - 1).expand(batch_size, symmetry_count, 1),
        ).squeeze(-1) / tail_counts[:, None]

        decoded = (
            0.5 * (predicted_normalized_O + 1.0) * extent[:, None]
            + bbox_min[:, None]
        )
        uniform = valid_mask.to(decoded.dtype)
        solution = self.procrustes.solve(
            decoded.float(), observed_points_C.float(), uniform.float(), valid_mask
        )
        pose = solution["transform"].to(decoded)
        rotation_raw = rotation_geodesic_distance(
            pose[:, None, :3, :3], poses[:, :, :3, :3]
        )
        translation_raw = torch.linalg.vector_norm(
            pose[:, None, :3, 3] - poses[:, :, :3, 3], dim=-1
        )
        reconstructed = transform_points(pose, decoded)
        alignment_distance = torch.linalg.vector_norm(
            reconstructed - observed_points_C, dim=-1
        )
        alignment_mean = (
            alignment_distance * valid_mask
        ).sum(-1) / lengths
        alignment_raw = alignment_mean[:, None].expand(-1, symmetry_count)

        normalized = {
            "fine_coordinate_aux": coordinate,
            "fine_coordinate_aux_tail": tail,
            "raw_aux_rotation": _normalized(
                rotation_raw, self.scales["rotation_rad"]
            ),
            "raw_aux_translation": _normalized(
                translation_raw, self.scales["translation_m"]
            ),
            "raw_aux_alignment": _normalized(
                alignment_raw, self.scales["alignment_m"]
            ),
        }
        raw = {
            "fine_coordinate_aux": coordinate,
            "fine_coordinate_aux_tail": tail,
            "raw_aux_rotation": rotation_raw,
            "raw_aux_translation": translation_raw,
            "raw_aux_alignment": alignment_raw,
        }
        progress = self.warmup_progress
        weights = dict(self.base_weights)
        for name in ("raw_aux_rotation", "raw_aux_translation", "raw_aux_alignment"):
            weights[name] *= progress
        totals = sum(weights[name] * normalized[name] for name in names)
        totals = totals.masked_fill(~symmetry_valid, float("inf"))
        selected = totals.argmin(-1)
        gather = selected[:, None]
        selected_total = totals.gather(1, gather).squeeze(1)
        full_weight = sum(
            self.base_weights[name] * normalized[name] for name in names
        ).gather(1, gather).squeeze(1)
        batch_indices = torch.arange(batch_size, device=selected.device)
        matched_targets = targets[batch_indices, selected]
        matched_poses = poses[batch_indices, selected]
        distances = torch.linalg.vector_norm(
            decoded[:, None] - targets, dim=-1
        )
        selected_distances = distances[batch_indices, selected]
        valid_selected_distances = selected_distances[valid_mask]

        def masked_eigenvalues(points: Tensor) -> Tensor:
            weights_mask = valid_mask.to(points.dtype)
            mean = (points * weights_mask[..., None]).sum(1) / lengths[:, None]
            centered = (points - mean[:, None]) * weights_mask[..., None]
            covariance = centered.transpose(1, 2) @ centered
            covariance = covariance / (lengths - 1).clamp_min(1)[:, None, None]
            return torch.linalg.eigvalsh(covariance.float()).clamp_min(0).to(points)

        predicted_eigenvalues = masked_eigenvalues(decoded)
        target_eigenvalues = masked_eigenvalues(matched_targets)
        nan = totals.new_tensor(float("nan"))
        totals_output = torch.where(symmetry_valid, totals, nan)
        rotation_output = torch.where(
            symmetry_valid, torch.rad2deg(rotation_raw), nan
        )
        translation_output = torch.where(
            symmetry_valid, translation_raw * 1000.0, nan
        )
        correspondence_output = torch.where(
            symmetry_valid,
            (distances * valid_mask[:, None]).sum(-1) / lengths[:, None] * 1000.0,
            nan,
        )
        result: dict[str, Any] = {
            "loss_total": selected_total.mean(),
            "per_sample_loss_total": selected_total,
            "selected_shared_symmetry_element": selected,
            "matched_target_points_O": matched_targets,
            "matched_gt_pose_T_C_from_O": matched_poses,
            "alignment_distance_m": alignment_distance * valid_mask,
            "decoded_q_aux_O": decoded * valid_mask.unsqueeze(-1),
            "correspondence_rank": solution["rank"],
            "loss_by_symmetry_element": totals_output,
            "rotation_error_by_symmetry_element_deg": rotation_output,
            "translation_error_by_symmetry_element_mm": translation_output,
            "correspondence_error_by_symmetry_element_mm": correspondence_output,
            "predicted_covariance_eigenvalues": predicted_eigenvalues,
            "gt_covariance_eigenvalues": target_eigenvalues,
            "selected_eigenvalue_ratios": (
                predicted_eigenvalues / target_eigenvalues.clamp_min(1e-12)
            ),
            "rank_margin_m2": predicted_eigenvalues[:, 0] - 1e-6,
            "aux_coordinate_rmse_mm": (
                valid_selected_distances.square().mean().sqrt() * 1000.0
            ),
            "aux_coordinate_p95_mm": (
                torch.quantile(valid_selected_distances.float(), 0.95).to(totals)
                * 1000.0
            ),
            "warmup_progress": totals.new_tensor(progress),
            "actual_optimized_loss": selected_total.mean(),
            "full_weight_counterfactual_loss": full_weight.mean(),
            "current_pose_loss_weight": totals.new_tensor(
                self.base_weights["raw_aux_rotation"] * progress
            ),
            "current_alignment_loss_weight": totals.new_tensor(
                self.base_weights["raw_aux_alignment"] * progress
            ),
        }
        for name in names:
            selected_normalized = normalized[name].gather(1, gather).squeeze(1)
            selected_raw = raw[name].gather(1, gather).squeeze(1)
            result[f"loss_{name}_normalized"] = selected_normalized.mean()
            result[f"raw_{name}"] = selected_raw.mean()
            result[f"weighted_loss_{name}"] = weights[name] * selected_normalized.mean()
            result[f"per_sample_loss_{name}_normalized"] = selected_normalized
            result[f"per_sample_raw_{name}"] = selected_raw
            result[f"per_sample_weighted_loss_{name}"] = weights[name] * selected_normalized
            result[f"current_weight_{name}"] = totals.new_tensor(weights[name])
            result[f"final_weight_{name}"] = totals.new_tensor(self.base_weights[name])
        result["weighted_loss_rotation"] = result["weighted_loss_raw_aux_rotation"]
        result["weighted_loss_translation"] = result["weighted_loss_raw_aux_translation"]
        result["rotation_error_deg"] = torch.rad2deg(result["raw_raw_aux_rotation"])
        result["translation_total_mm"] = result["raw_raw_aux_translation"] * 1000.0
        return result


__all__ = ["CleanCoordinatePoseLossV3", "scratch_warmup_progress"]
