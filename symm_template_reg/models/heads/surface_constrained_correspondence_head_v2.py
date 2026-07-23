"""Coarse-to-local correspondence head with exact triangle-surface outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from symm_template_reg.geometry import closest_points_on_triangle_mesh
from symm_template_reg.models.geometry.point_ops import (
    batched_gather,
    farthest_point_indices,
)
from symm_template_reg.models.geometry.patch_targets import valid_patch_mask
from symm_template_reg.models.geometry.patch_targets import multi_positive_softmax_loss
from symm_template_reg.models.geometry.triangle_targets import (
    deduplicate_candidate_ids,
    inject_valid_triangle_ids,
    local_valid_triangle_mask,
    triangle_target_sets,
)
from symm_template_reg.models.pose.pose_representation import (
    invert_transform,
    transform_points,
)
from symm_template_reg.models.symmetry.groups import parse_rotation_group
from symm_template_reg.models.symmetry.hypothesis_expander import symmetry_transforms
from symm_template_reg.registry import GEOMETRY_MODULES, HEADS, build_from_cfg


def selected_patch_candidate_scores(
    queries: Tensor, patch_candidate_features: Tensor, selected_patches: Tensor
) -> Tensor:
    """Score only selected patch candidates without expanding ``[N,K,C,D]``.

    Grouping point/slot pairs by patch turns the operation into at most P
    ordinary matrix multiplications.  This preserves exact logits and
    gradients while avoiding the multi-gigabyte advanced-indexing tensor.
    """
    if queries.ndim != 2 or patch_candidate_features.ndim != 3:
        raise ValueError("expected queries [N,D] and patch features [P,C,D]")
    if selected_patches.ndim != 2 or selected_patches.shape[0] != len(queries):
        raise ValueError("selected patches must have shape [N,K]")
    point_ids = torch.arange(len(queries), device=queries.device)[:, None].expand_as(
        selected_patches
    ).reshape(-1)
    flat_patch_ids = selected_patches.reshape(-1)
    positions_by_patch = []
    scores_by_patch = []
    for patch_id in range(len(patch_candidate_features)):
        positions = torch.nonzero(
            flat_patch_ids.eq(patch_id), as_tuple=False
        ).flatten()
        if positions.numel() == 0:
            continue
        positions_by_patch.append(positions)
        scores_by_patch.append(
            queries[point_ids[positions]] @ patch_candidate_features[patch_id].transpose(0, 1)
        )
    if not scores_by_patch:
        raise ValueError("selected patches contain no valid patch ids")
    positions = torch.cat(positions_by_patch)
    scores = torch.cat(scores_by_patch)
    restore_order = positions.argsort()
    return scores[restore_order].reshape(
        len(queries), -1
    )


def point_conditioned_candidate_scores(
    queries: Tensor, candidate_features: Tensor
) -> Tensor:
    """Score paired ``[N,L,D]`` candidates with per-point ``[N,D]`` queries."""

    if queries.ndim != 2 or candidate_features.ndim != 3:
        raise ValueError("expected queries [N,D] and candidates [N,L,D]")
    if candidate_features.shape[0] != len(queries) or candidate_features.shape[-1] != queries.shape[-1]:
        raise ValueError("point/candidate feature dimensions disagree")
    return torch.einsum("nd,nld->nl", queries, candidate_features)


@HEADS.register_module()
class SurfaceConstrainedCorrespondenceHeadV2(nn.Module):
    """Classify top-k patches, then a local triangle and barycentric point."""

    is_surface_constrained_v2 = True
    requires_template_mesh = True
    deprecated_for_new_configs = True

    def __init__(
        self,
        embed_dim: int = 256,
        num_patches: int = 64,
        top_k_patches: int = 4,
        local_candidates: int = 32,
        fine_mode: str = "triangle_barycentric",
        temperature: float = 1.0,
        initial_temperature: float | None = None,
        final_temperature: float | None = None,
        anneal_epochs: int = 0,
        teacher_forcing_initial_probability: float = 0.0,
        teacher_forcing_final_probability: float = 0.0,
        teacher_forcing_start_epoch: int = 0,
        teacher_forcing_anneal_epochs: int = 0,
        teacher_forcing_decay_min_top4_recall: float = 0.0,
        teacher_forcing_during_evaluation: bool = False,
        deduplicate_local_candidates: bool = False,
        inject_all_valid_triangles: bool = False,
        teacher_force_exact_triangle: bool = False,
        teacher_forcing_select_shared_symmetry_element: bool = False,
        triangle_target_tolerance_m: float = 0.00015,
        candidate_geometry_weight: float = 0.0,
        max_local_candidate_total: int | None = None,
        sort_owned_faces_by_distance: bool = False,
        fine_feature_adapter: Mapping[str, object] | None = None,
        fine_candidate_triangle_head: Mapping[str, object] | None = None,
        fine_coordinate_auxiliary_head: Mapping[str, object] | None = None,
        coordinate_guided_triangle_head: Mapping[str, object] | None = None,
        analytic_barycentric_projection: bool = False,
        learned_barycentric_status: str = "legacy_available",
    ) -> None:
        super().__init__()
        if fine_mode not in {"triangle_barycentric", "local_fine_point_distribution"}:
            raise ValueError(fine_mode)
        if temperature <= 0 or top_k_patches < 1:
            raise ValueError("temperature and top_k_patches must be positive")
        for value in (teacher_forcing_initial_probability, teacher_forcing_final_probability):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError("teacher-forcing probability must be in [0,1]")
        self.num_patches = int(num_patches)
        self.top_k_patches = int(top_k_patches)
        self.local_candidates = int(local_candidates)
        self.fine_mode = fine_mode
        self.temperature = float(temperature)
        self.initial_temperature = float(initial_temperature if initial_temperature is not None else temperature)
        self.final_temperature = float(final_temperature if final_temperature is not None else temperature)
        self.anneal_epochs = int(anneal_epochs)
        self.teacher_forcing_initial_probability = float(teacher_forcing_initial_probability)
        self.teacher_forcing_final_probability = float(teacher_forcing_final_probability)
        self.teacher_forcing_start_epoch = int(teacher_forcing_start_epoch)
        self.teacher_forcing_anneal_epochs = int(teacher_forcing_anneal_epochs)
        self.teacher_forcing_decay_min_top4_recall = float(
            teacher_forcing_decay_min_top4_recall
        )
        self.teacher_forcing_during_evaluation = bool(
            teacher_forcing_during_evaluation
        )
        self.deduplicate_local_candidates = bool(deduplicate_local_candidates)
        self.inject_all_valid_triangles = bool(inject_all_valid_triangles)
        self.teacher_force_exact_triangle = bool(teacher_force_exact_triangle)
        self.teacher_forcing_select_shared_symmetry_element = bool(
            teacher_forcing_select_shared_symmetry_element
        )
        self.triangle_target_tolerance_m = float(triangle_target_tolerance_m)
        self.candidate_geometry_weight = float(candidate_geometry_weight)
        self.max_local_candidate_total = (
            None if max_local_candidate_total is None else int(max_local_candidate_total)
        )
        self.sort_owned_faces_by_distance = bool(sort_owned_faces_by_distance)
        if fine_feature_adapter is None and (
            fine_candidate_triangle_head is not None
            or coordinate_guided_triangle_head is not None
        ):
            raise ValueError(
                "candidate heads require a fine adapter; coarse-only/broadcast "
                "fine inputs are forbidden"
            )
        if fine_feature_adapter is not None and fine_candidate_triangle_head is None and coordinate_guided_triangle_head is None:
            raise ValueError("fine adapter requires a point-conditioned candidate head")
        self.fine_feature_adapter = (
            build_from_cfg(fine_feature_adapter, GEOMETRY_MODULES)
            if fine_feature_adapter is not None else None
        )
        self.fine_candidate_triangle_head = (
            build_from_cfg(fine_candidate_triangle_head, HEADS)
            if fine_candidate_triangle_head is not None else None
        )
        self.fine_coordinate_auxiliary_head = (
            build_from_cfg(fine_coordinate_auxiliary_head, HEADS)
            if fine_coordinate_auxiliary_head is not None else None
        )
        self.coordinate_guided_triangle_head = (
            build_from_cfg(coordinate_guided_triangle_head, HEADS)
            if coordinate_guided_triangle_head is not None else None
        )
        self.analytic_barycentric_projection = bool(analytic_barycentric_projection)
        if self.coordinate_guided_triangle_head is not None and not self.analytic_barycentric_projection:
            raise ValueError("coordinate-guided triangle fallback requires analytic projection")
        self.learned_barycentric_status = str(learned_barycentric_status)
        if self.fine_coordinate_auxiliary_head is not None and self.fine_feature_adapter is None:
            raise ValueError("fine coordinate auxiliary head requires dense fine features")
        self.uses_separate_fine_features = self.fine_feature_adapter is not None
        self.latest_top4_recall = 0.0
        self._teacher_forcing_decay_epoch0: int | None = None
        self.teacher_forcing_probability = self.teacher_forcing_initial_probability
        self.observed_query = nn.Linear(embed_dim, embed_dim)
        self.template_key = nn.Linear(embed_dim, embed_dim)
        self.fine_query = nn.Linear(embed_dim, embed_dim)
        self.barycentric_head = nn.Linear(embed_dim, 3)

    def set_epoch(self, epoch: int) -> float:
        if self.anneal_epochs > 0:
            ratio = min(max(float(epoch), 0.0) / self.anneal_epochs, 1.0)
            self.temperature = self.initial_temperature + ratio * (
                self.final_temperature - self.initial_temperature
            )
        else:
            self.temperature = self.initial_temperature
        recall_gate_passed = (
            self.latest_top4_recall >= self.teacher_forcing_decay_min_top4_recall
        )
        if (
            recall_gate_passed
            and epoch >= self.teacher_forcing_start_epoch
            and self._teacher_forcing_decay_epoch0 is None
        ):
            self._teacher_forcing_decay_epoch0 = int(epoch)
        if epoch < self.teacher_forcing_start_epoch or not recall_gate_passed:
            ratio = 0.0
        elif self.teacher_forcing_anneal_epochs > 0:
            ratio = min(
                (epoch - max(
                    self.teacher_forcing_start_epoch,
                    int(self._teacher_forcing_decay_epoch0 or epoch),
                ))
                / self.teacher_forcing_anneal_epochs,
                1.0,
            )
        else:
            ratio = 1.0
        self.teacher_forcing_probability = self.teacher_forcing_initial_probability + ratio * (
            self.teacher_forcing_final_probability
            - self.teacher_forcing_initial_probability
        )
        return self.temperature

    def set_patch_recall(self, top4_recall: float) -> None:
        self.latest_top4_recall = float(top4_recall)

    @staticmethod
    def inject_gt_patch(
        predicted_topk: Tensor, gt_patch: Tensor, probability: float
    ) -> tuple[Tensor, Tensor]:
        """Replace the last candidate when sampled teacher forcing is active."""

        included = predicted_topk.eq(gt_patch[..., None]).any(-1)
        if probability <= 0:
            return predicted_topk, torch.zeros_like(included)
        force = torch.rand_like(gt_patch.float()).lt(probability) & ~included
        result = predicted_topk.clone()
        result[..., -1] = torch.where(force, gt_patch, result[..., -1])
        return result, force

    @staticmethod
    def inject_valid_patch_set(
        predicted_topk: Tensor,
        valid_targets: Tensor,
        preferred_patch: Tensor,
        probability: float,
    ) -> tuple[Tensor, Tensor]:
        """Inject one valid patch only when no valid patch is already selected."""

        included = valid_targets.gather(-1, predicted_topk.long()).any(-1)
        if probability <= 0:
            return predicted_topk, torch.zeros_like(included)
        force = torch.rand_like(preferred_patch.float()).lt(probability) & ~included
        result = predicted_topk.clone()
        result[..., -1] = torch.where(force, preferred_patch, result[..., -1])
        return result, force

    def forward(
        self,
        observed_features: Tensor,
        template_features: Tensor,
        template_points: Tensor,
        observed_mask: Tensor,
        template_mask: Tensor,
        *,
        template_mesh_vertices_O: Sequence[Tensor],
        template_mesh_faces: Sequence[Tensor],
        teacher_forcing_target_points_O: Tensor | None = None,
        teacher_forcing_symmetry_metadata: Sequence[object] | None = None,
        teacher_forcing_effective_symmetry_groups: Sequence[object] | None = None,
        original_dense_observed_features: Tensor | None = None,
        interpolated_observed_interaction_features: Tensor | None = None,
        observed_points_C: Tensor | None = None,
        observed_normals_C: Tensor | None = None,
        fine_template_interaction_features: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        patch_indices, patch_mask = farthest_point_indices(
            template_points, template_mask, self.num_patches
        )
        patch_points = batched_gather(template_points, patch_indices)
        patch_features = batched_gather(template_features, patch_indices)
        coarse_logits = (
            self.observed_query(observed_features)
            @ self.template_key(patch_features).transpose(-2, -1)
            / (math.sqrt(observed_features.shape[-1]) * self.temperature)
        )
        coarse_logits = coarse_logits.masked_fill(~patch_mask[:, None], float("-inf"))
        fine_adapter_output = None
        fine_auxiliary_coordinates = None
        if self.fine_feature_adapter is not None:
            if any(
                value is None
                for value in (
                    original_dense_observed_features,
                    interpolated_observed_interaction_features,
                    observed_points_C,
                )
            ):
                raise ValueError("separate fine feature sources are required by the fine adapter")
            fine_adapter_output = self.fine_feature_adapter(
                original_dense_observed_features,
                interpolated_observed_interaction_features,
                observed_points_C,
                observed_mask,
                observed_normals_C,
            )
            if self.fine_coordinate_auxiliary_head is not None:
                fine_auxiliary_coordinates = self.fine_coordinate_auxiliary_head(
                    fine_adapter_output["fine_point_features"]
                )
        predicted_topk_patch = coarse_logits.topk(
            min(self.top_k_patches, coarse_logits.shape[-1]), dim=-1
        ).indices
        outputs, fine_rows, triangle_rows, bary_rows, bary_logit_rows = [], [], [], [], []
        candidate_rows, candidate_mask_rows, candidate_patch_rows = [], [], []
        raw_candidate_rows, duplicate_count_rows = [], []
        all_candidate_rows, face_owner_rows = [], []
        global_feature_rows, centroid_rows, normal_rows, triangle_vertex_rows = [], [], [], []
        fine_query_rows = []
        fine_geometry_rows, triangle_geometry_rows = [], []
        selected_topk_rows, gt_patch_rows, gt_triangle_rows = [], [], []
        valid_patch_rows, valid_triangle_local_rows, triangle_injected_rows = [], [], []
        valid_triangle_global_rows = []
        symmetry_element_rows, injected_rows = [], []
        guided_candidate_rows = []
        for batch_index in range(len(observed_features)):
            vertices = template_mesh_vertices_O[batch_index].to(template_points)
            faces = template_mesh_faces[batch_index].to(
                device=template_points.device, dtype=torch.long
            )
            triangles = vertices[faces]
            centroids = triangles.mean(1)
            patch_to_face = torch.cdist(
                patch_points[batch_index].float(), centroids.float()
            )
            face_owner = patch_to_face.argmin(0)
            face_owner_rows.append(face_owner)
            maximum_owned = max(
                int(face_owner.eq(patch_id).sum())
                for patch_id in range(len(patch_points[batch_index]))
            )
            candidate_count = min(
                len(faces), max(self.local_candidates, maximum_owned)
            )
            candidate_face_rows = []
            for patch_id in range(len(patch_points[batch_index])):
                owned = torch.nonzero(
                    face_owner.eq(patch_id), as_tuple=False
                ).flatten()
                if self.sort_owned_faces_by_distance and owned.numel() > 1:
                    owned = owned[patch_to_face[patch_id, owned].argsort()]
                nearest = patch_to_face[patch_id].argsort()
                remaining = nearest[~torch.isin(nearest, owned)]
                candidate_face_rows.append(
                    torch.cat((owned, remaining))[:candidate_count]
                )
            all_candidate_face = torch.stack(candidate_face_rows)
            all_candidate_rows.append(all_candidate_face)
            # The wide list is needed to preserve the Stage-A overlapping
            # patch-target contract.  Stage B ultimately scores at most 32
            # deduplicated faces and injects every valid target explicitly, so
            # feeding hundreds of owned faces per selected patch into the
            # local deduplicator only wastes memory and synchronization time.
            candidate_face = (
                all_candidate_face[
                    :, : min(self.local_candidates, all_candidate_face.shape[-1])
                ]
                if self.max_local_candidate_total is not None
                and self.inject_all_valid_triangles
                else all_candidate_face
            )
            nearest_anchor = torch.cdist(
                centroids.reshape(1, -1, 3).float(),
                template_points[batch_index : batch_index + 1].float(),
            ).argmin(-1).reshape(-1)
            global_face_features = template_features[batch_index][nearest_anchor]
            face_normals = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            face_normals = face_normals / torch.linalg.vector_norm(
                face_normals, dim=-1, keepdim=True
            ).clamp_min(1e-12)
            centered_vertices = (triangles - centroids[:, None]).reshape(len(faces), -1)
            edge_lengths = torch.stack(
                (
                    torch.linalg.vector_norm(triangles[:, 1] - triangles[:, 0], dim=-1),
                    torch.linalg.vector_norm(triangles[:, 2] - triangles[:, 1], dim=-1),
                    torch.linalg.vector_norm(triangles[:, 0] - triangles[:, 2], dim=-1),
                ),
                dim=-1,
            )
            raw_geometry = torch.cat(
                (centroids, face_normals, centered_vertices, edge_lengths), dim=-1
            )
            geometry = (raw_geometry - raw_geometry.mean(0)) / raw_geometry.std(
                0, unbiased=False
            ).clamp_min(1e-6)
            if self.candidate_geometry_weight != 0.0:
                geometry_for_features = geometry
                repeats = math.ceil(
                    global_face_features.shape[-1] / geometry_for_features.shape[-1]
                )
                geometry_for_features = geometry_for_features.repeat(1, repeats)[
                    :, : global_face_features.shape[-1]
                ]
                global_face_features = global_face_features + (
                    self.candidate_geometry_weight
                    * geometry_for_features.to(global_face_features)
                )
            global_triangle_local_geometry = geometry.to(global_face_features)
            if fine_adapter_output is not None:
                if fine_template_interaction_features is None:
                    raise ValueError("fine template interaction features are required")
                if teacher_forcing_symmetry_metadata is None:
                    raise ValueError("fine triangle metadata requires the template symmetry axis")
                metadata = teacher_forcing_symmetry_metadata[batch_index]
                fine_triangle = self.fine_feature_adapter.template_triangle_features(
                    vertices,
                    faces,
                    patch_features[batch_index],
                    fine_template_interaction_features[batch_index],
                    face_owner,
                    nearest_anchor,
                    axis_direction_O=metadata.axis.direction,
                    axis_origin_O=metadata.axis.origin,
                )
                global_face_features = fine_triangle["fine_triangle_features"]
                global_triangle_local_geometry = fine_triangle["triangle_local_geometry"]
            topk = predicted_topk_patch[batch_index]
            if teacher_forcing_target_points_O is not None:
                raw_target = teacher_forcing_target_points_O[batch_index]
                target_options = raw_target.unsqueeze(0)
                if (
                    self.teacher_forcing_select_shared_symmetry_element
                    and
                    teacher_forcing_symmetry_metadata is not None
                    and teacher_forcing_effective_symmetry_groups is not None
                ):
                    metadata = teacher_forcing_symmetry_metadata[batch_index]
                    group = parse_rotation_group(
                        teacher_forcing_effective_symmetry_groups[batch_index]
                    )
                    symmetries = symmetry_transforms(
                        group,
                        metadata.axis.direction,
                        metadata.axis.origin,
                        so2_num_samples=36 if group.type == "SO2" else None,
                        dtype=raw_target.dtype,
                        device=raw_target.device,
                    )
                    target_options = transform_points(
                        invert_transform(symmetries), raw_target.unsqueeze(0)
                    )
                option_nearest = [
                    closest_points_on_triangle_mesh(
                        target,
                        vertices,
                        faces,
                        point_chunk_size=256,
                    )
                    for target in target_options
                ]
                option_patch_masks = [
                    valid_patch_mask(option["face_ids"], all_candidate_face)
                    for option in option_nearest
                ]
                option_losses = torch.stack(
                    [
                        multi_positive_softmax_loss(
                            coarse_logits[batch_index][observed_mask[batch_index]],
                            patch_mask_option[observed_mask[batch_index]],
                        )
                        for patch_mask_option in option_patch_masks
                    ]
                )
                selected_symmetry = int(option_losses.argmin().detach())
                nearest = triangle_target_sets(
                    target_options[selected_symmetry],
                    vertices,
                    faces,
                    tolerance_m=self.triangle_target_tolerance_m,
                    point_chunk_size=256,
                )
                gt_triangle = nearest["face_ids"]
                gt_patch = face_owner[gt_triangle]
                valid_targets = option_patch_masks[selected_symmetry]
                global_valid_triangles = nearest["valid_triangle_mask"]
                topk, injected = self.inject_valid_patch_set(
                    topk,
                    valid_targets,
                    gt_patch,
                    self.teacher_forcing_probability,
                )
            else:
                gt_patch = torch.zeros_like(topk[:, 0])
                gt_triangle = torch.zeros_like(gt_patch)
                valid_targets = torch.zeros(
                    (*gt_patch.shape, candidate_face.shape[0]),
                    dtype=torch.bool,
                    device=gt_patch.device,
                )
                global_valid_triangles = torch.zeros(
                    (*gt_patch.shape, len(faces)),
                    dtype=torch.bool,
                    device=gt_patch.device,
                )
                selected_symmetry = -1
                injected = torch.zeros_like(gt_patch, dtype=torch.bool)
            selected_topk_rows.append(topk)
            gt_patch_rows.append(gt_patch)
            gt_triangle_rows.append(gt_triangle)
            valid_patch_rows.append(valid_targets)
            symmetry_element_rows.append(
                torch.as_tensor(selected_symmetry, device=topk.device, dtype=torch.long)
            )
            injected_rows.append(injected)
            selected_candidates_raw = candidate_face[topk].flatten(1, 2)
            candidates_per_patch = candidate_face.shape[-1]
            selected_candidate_patches = topk[..., None].expand(
                -1, -1, candidates_per_patch
            ).flatten(1, 2)
            if self.deduplicate_local_candidates:
                selected_candidates, selected_candidate_mask, selected_candidate_patches = (
                    deduplicate_candidate_ids(
                        selected_candidates_raw, selected_candidate_patches
                    )
                )
                unique_count = selected_candidate_mask.sum(-1)
            else:
                selected_candidates = selected_candidates_raw
                selected_candidate_mask = torch.ones_like(
                    selected_candidates, dtype=torch.bool
                )
                # Keep duplicate diagnostics without one Python/CUDA sync per
                # point.  The vectorized stable compaction is diagnostic-only
                # in this legacy branch.
                _, unique_mask, _ = deduplicate_candidate_ids(
                    selected_candidates_raw
                )
                unique_count = unique_mask.sum(-1)
            duplicate_count_rows.append(
                selected_candidates_raw.ge(0).sum(-1) - unique_count
            )
            if self.max_local_candidate_total is not None:
                width = min(self.max_local_candidate_total, selected_candidates.shape[-1])
                selected_candidates = selected_candidates[:, :width]
                selected_candidate_mask = selected_candidate_mask[:, :width]
                assert selected_candidate_patches is not None
                selected_candidate_patches = selected_candidate_patches[:, :width]
            triangle_injected = torch.zeros_like(selected_candidate_mask)
            if teacher_forcing_target_points_O is not None and self.inject_all_valid_triangles:
                selected_candidates, selected_candidate_mask, triangle_injected = (
                    inject_valid_triangle_ids(
                        selected_candidates,
                        selected_candidate_mask,
                        global_valid_triangles,
                    )
                )
                assert selected_candidate_patches is not None
                selected_candidate_patches = selected_candidate_patches.clone()
                selected_candidate_patches[triangle_injected] = face_owner[
                    selected_candidates[triangle_injected]
                ]
            guided = None
            if fine_adapter_output is None:
                query_features = self.fine_query(observed_features[batch_index])
                global_scores = (
                    query_features @ global_face_features.transpose(0, 1)
                ) / (math.sqrt(observed_features.shape[-1]) * self.temperature)
                fine_logits = global_scores.gather(
                    1, selected_candidates.clamp_min(0)
                )
                assert selected_candidate_patches is not None
                selected_coarse_scores = coarse_logits[batch_index].gather(
                    1, selected_candidate_patches.clamp_min(0)
                )
                fine_logits = fine_logits + selected_coarse_scores
                fine_logits = fine_logits.masked_fill(
                    ~selected_candidate_mask, float("-inf")
                )
                observed_local_geometry = observed_features.new_zeros(
                    (len(query_features), 0)
                )
            else:
                query_features = fine_adapter_output["fine_point_features"][batch_index]
                observed_local_geometry = fine_adapter_output[
                    "observed_local_geometry"
                ][batch_index]
                candidate_features = global_face_features[
                    selected_candidates.clamp_min(0)
                ]
                candidate_geometry = global_triangle_local_geometry[
                    selected_candidates.clamp_min(0)
                ]
                if self.coordinate_guided_triangle_head is not None:
                    if fine_auxiliary_coordinates is None:
                        raise ValueError("coordinate-guided triangle head requires q_aux")
                    bbox_min, bbox_max = vertices.amin(0), vertices.amax(0)
                    q_aux_O = .5 * (
                        fine_auxiliary_coordinates[batch_index] + 1.0
                    ) * (bbox_max - bbox_min).clamp_min(1e-8) + bbox_min
                    candidate_triangles = triangles[selected_candidates.clamp_min(0)]
                    candidate_normals = face_normals[selected_candidates.clamp_min(0)]
                    candidate_edges = torch.stack((
                        torch.linalg.vector_norm(candidate_triangles[:, :, 1] - candidate_triangles[:, :, 0], dim=-1),
                        torch.linalg.vector_norm(candidate_triangles[:, :, 2] - candidate_triangles[:, :, 1], dim=-1),
                        torch.linalg.vector_norm(candidate_triangles[:, :, 0] - candidate_triangles[:, :, 2], dim=-1),
                    ), -1)
                    coarse_candidate_features = patch_features[batch_index][
                        selected_candidate_patches.clamp_min(0)
                    ]
                    guided = self.coordinate_guided_triangle_head(
                        query_features, candidate_features, q_aux_O,
                        candidate_triangles, candidate_normals, candidate_edges,
                        coarse_candidate_features, selected_candidate_mask,
                    )
                    fine_logits = guided["logits"] / self.temperature
                else:
                    assert self.fine_candidate_triangle_head is not None
                    guided = None
                    fine_logits = self.fine_candidate_triangle_head(
                        query_features,
                        candidate_features,
                        observed_local_geometry,
                        candidate_geometry,
                        selected_candidate_mask,
                    ) / self.temperature
            # The hard top-k membership is discrete, but the local objective still
            # reaches the selected coarse scores.  Patch CE supplies gradients for
            # patches outside the current set.
            selected_local = fine_logits.argmax(-1)
            valid_triangle_local = local_valid_triangle_mask(
                selected_candidates, global_valid_triangles
            )
            if teacher_forcing_target_points_O is not None and (
                self.inject_all_valid_triangles or self.teacher_force_exact_triangle
            ):
                if not bool(valid_triangle_local[observed_mask[batch_index]].any(-1).all()):
                    raise AssertionError(
                        "teacher forcing failed to include a valid GT triangle"
                    )
                if self.teacher_force_exact_triangle:
                    exact_local = selected_candidates.eq(gt_triangle[:, None])
                    if not bool(exact_local[observed_mask[batch_index]].any(-1).all()):
                        raise AssertionError(
                            "exact GT triangle is absent from teacher-forced candidates"
                        )
                    selected_local = exact_local.to(torch.int64).argmax(-1)
            rows = torch.arange(len(selected_local), device=selected_local.device)
            chosen_face = selected_candidates[rows, selected_local]
            chosen_triangle = triangles[chosen_face]
            if self.analytic_barycentric_projection and guided is not None:
                barycentric = guided["candidate_analytic_barycentric"][rows, selected_local]
                barycentric_logits = torch.zeros_like(barycentric)
                q = guided["candidate_closest_points_O"][rows, selected_local]
            else:
                barycentric_logits = self.barycentric_head(
                    query_features if fine_adapter_output is not None else observed_features[batch_index]
                ) / self.temperature
                barycentric = torch.softmax(barycentric_logits, -1)
            if self.analytic_barycentric_projection and guided is not None:
                pass
            elif self.fine_mode == "triangle_barycentric":
                q = (barycentric[..., None] * chosen_triangle).sum(1)
            else:
                probability = torch.softmax(fine_logits, -1)
                candidate_centroids = centroids[selected_candidates]
                q = (probability[..., None] * candidate_centroids).sum(1)
            outputs.append(q * observed_mask[batch_index, :, None])
            fine_rows.append(fine_logits)
            triangle_rows.append(chosen_face)
            bary_rows.append(barycentric)
            bary_logit_rows.append(barycentric_logits)
            candidate_rows.append(selected_candidates)
            candidate_mask_rows.append(selected_candidate_mask)
            candidate_patch_rows.append(selected_candidate_patches)
            raw_candidate_rows.append(selected_candidates_raw)
            valid_triangle_local_rows.append(valid_triangle_local)
            valid_triangle_global_rows.append(global_valid_triangles)
            triangle_injected_rows.append(triangle_injected)
            global_feature_rows.append(global_face_features)
            centroid_rows.append(centroids)
            normal_rows.append(face_normals)
            triangle_vertex_rows.append(triangles)
            fine_query_rows.append(query_features)
            fine_geometry_rows.append(observed_local_geometry)
            triangle_geometry_rows.append(global_triangle_local_geometry)
            guided_candidate_rows.append(guided)
        points = torch.stack(outputs)
        fine_logits = torch.stack(fine_rows)
        selected_topk_patch = torch.stack(selected_topk_rows)
        selected_patch = selected_topk_patch[..., 0]
        auxiliary = {
            "coarse_patch_logits": coarse_logits,
            "fine_local_logits": fine_logits,
            "selected_patch_ids": selected_patch,
            "selected_topk_patch_ids": selected_topk_patch,
            "selected_triangle_ids": torch.stack(triangle_rows),
            "predicted_barycentric": torch.stack(bary_rows),
            "barycentric_logits": torch.stack(bary_logit_rows),
            "candidate_triangle_ids": torch.stack(candidate_rows),
            "candidate_triangle_mask": torch.stack(candidate_mask_rows),
            "candidate_patch_ids": torch.stack(candidate_patch_rows),
            "raw_candidate_triangle_ids": torch.stack(raw_candidate_rows),
            "duplicate_candidate_count": torch.stack(duplicate_count_rows),
            "valid_triangle_local_mask": torch.stack(valid_triangle_local_rows),
            "teacher_forcing_valid_triangle_global_mask": torch.stack(
                valid_triangle_global_rows
            ),
            "valid_triangle_injected_mask": torch.stack(triangle_injected_rows),
            "all_candidate_triangle_ids": torch.stack(all_candidate_rows),
            "face_owner_patch_ids": torch.stack(face_owner_rows),
            "global_triangle_features": torch.stack(global_feature_rows),
            "global_triangle_centroids_O": torch.stack(centroid_rows),
            "global_triangle_normals_O": torch.stack(normal_rows),
            "global_triangle_vertices_O": torch.stack(triangle_vertex_rows),
            "fine_observed_query_features": torch.stack(fine_query_rows),
            "fine_point_features": torch.stack(fine_query_rows),
            "fine_observed_local_geometry": torch.stack(fine_geometry_rows),
            "fine_triangle_local_geometry": torch.stack(triangle_geometry_rows),
            "coarse_patch_features": patch_features,
            "fine_triangle_features": torch.stack(global_feature_rows),
            "patch_points_O": patch_points,
            "temperature": points.new_tensor(self.temperature),
            "teacher_forcing_probability": points.new_tensor(
                self.teacher_forcing_probability
            ),
            "gt_patch_injected_mask": torch.stack(injected_rows),
            "analytic_barycentric_projection": points.new_tensor(
                self.analytic_barycentric_projection, dtype=torch.bool
            ),
        }
        if teacher_forcing_target_points_O is not None:
            auxiliary["teacher_forcing_gt_patch_ids"] = torch.stack(gt_patch_rows)
            auxiliary["teacher_forcing_gt_triangle_ids"] = torch.stack(gt_triangle_rows)
            auxiliary["teacher_forcing_valid_patch_mask"] = torch.stack(valid_patch_rows)
            auxiliary["teacher_forcing_selected_symmetry_element"] = torch.stack(
                symmetry_element_rows
            )
        if fine_adapter_output is not None:
            auxiliary["fine_feature_variance"] = fine_adapter_output[
                "fine_feature_variance"
            ].reshape(1)
            auxiliary["fine_feature_effective_rank"] = fine_adapter_output[
                "fine_feature_effective_rank"
            ].reshape(1)
            auxiliary["fine_feature_pairwise_distance"] = fine_adapter_output[
                "fine_feature_pairwise_distance"
            ].reshape(1)
            auxiliary["fine_feature_collision_fraction"] = fine_adapter_output[
                "fine_feature_collision_fraction"
            ].reshape(1)
            auxiliary["fine_candidate_logit_variance"] = torch.stack(
                [row[mask].var(unbiased=False) for row, mask in zip(fine_rows, observed_mask)]
            )
        if fine_auxiliary_coordinates is not None:
            auxiliary["fine_aux_coordinate_normalized"] = fine_auxiliary_coordinates
        confidence = torch.softmax(fine_logits, -1).amax(-1) * observed_mask
        return {
            "points_O": points,
            "confidence": confidence,
            "logits": coarse_logits,
            "auxiliary": auxiliary,
        }


__all__ = [
    "SurfaceConstrainedCorrespondenceHeadV2",
    "selected_patch_candidate_scores",
    "point_conditioned_candidate_scores",
]
