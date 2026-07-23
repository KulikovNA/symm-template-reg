"""Exact and tolerance-aware set-valued triangle targets."""

from __future__ import annotations

import torch
from torch import Tensor

from symm_template_reg.geometry import closest_points_on_triangle_mesh


def closest_barycentric_on_triangles(points: Tensor, triangles: Tensor) -> dict[str, Tensor]:
    """Closest point/barycentric coordinates for one triangle per point."""

    query = torch.as_tensor(points).float()
    tri = torch.as_tensor(triangles, dtype=query.dtype, device=query.device)
    if tri.shape != (len(query), 3, 3):
        raise ValueError("triangles must have shape [N,3,3]")
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, query - a
    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)
    d20 = (ap * ab).sum(-1)
    d21 = (ap * ac).sum(-1)
    denom = (d00 * d11 - d01.square()).clamp_min(1e-20)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    plane_bary = torch.stack((u, v, w), -1)
    plane = (plane_bary[..., None] * tri).sum(1)
    inside = (plane_bary >= 0).all(-1)
    candidate_points = [plane]
    candidate_bary = [plane_bary]
    for x, y, left, right in ((a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)):
        edge = y - x
        t = ((query - x) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
        t = t.clamp(0, 1)
        candidate_points.append(x + t[:, None] * edge)
        bary = torch.zeros((len(query), 3), dtype=query.dtype, device=query.device)
        bary[:, left] = 1.0 - t
        bary[:, right] = t
        candidate_bary.append(bary)
    points_by_case = torch.stack(candidate_points, 1)
    bary_by_case = torch.stack(candidate_bary, 1)
    distance2 = (points_by_case - query[:, None]).square().sum(-1)
    distance2[:, 0] = distance2[:, 0].masked_fill(~inside, float("inf"))
    selected = distance2.argmin(-1)
    row = torch.arange(len(query), device=query.device)
    return {
        "points": points_by_case[row, selected],
        "barycentric": bary_by_case[row, selected],
        "distances": distance2[row, selected].clamp_min(0).sqrt(),
    }


def point_triangle_distance_matrix(
    points: Tensor,
    vertices: Tensor,
    faces: Tensor,
    *,
    point_chunk_size: int = 256,
) -> Tensor:
    """Return exact point-to-triangle distances with shape ``[N,F]``."""

    query = torch.as_tensor(points).float()
    mesh_vertices = torch.as_tensor(vertices, dtype=query.dtype, device=query.device)
    mesh_faces = torch.as_tensor(faces, dtype=torch.long, device=query.device)
    triangles = mesh_vertices[mesh_faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    d00 = (ab * ab).sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = (ac * ac).sum(-1)
    denom = (d00 * d11 - d01.square()).clamp_min(1e-20)
    rows: list[Tensor] = []
    for start in range(0, len(query), int(point_chunk_size)):
        p = query[start : start + int(point_chunk_size)][:, None, :]
        ap = p - a[None]
        d20 = (ap * ab[None]).sum(-1)
        d21 = (ap * ac[None]).sum(-1)
        v = (d11[None] * d20 - d01[None] * d21) / denom[None]
        w = (d00[None] * d21 - d01[None] * d20) / denom[None]
        u = 1.0 - v - w
        plane = u[..., None] * a + v[..., None] * b + w[..., None] * c
        inside = (u >= 0) & (v >= 0) & (w >= 0)
        distance2 = (plane - p).square().sum(-1).masked_fill(~inside, float("inf"))
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x
            t = ((p - x) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
            projection = x + t.clamp(0, 1)[..., None] * edge
            distance2 = torch.minimum(distance2, (projection - p).square().sum(-1))
        rows.append(distance2.clamp_min(0).sqrt())
    return torch.cat(rows)


def triangle_target_sets(
    points: Tensor,
    vertices: Tensor,
    faces: Tensor,
    *,
    tolerance_m: float = 0.00015,
    point_chunk_size: int = 256,
) -> dict[str, Tensor]:
    """Build exact-owner and tolerance-aware valid triangle targets.

    All faces whose exact point-to-triangle distance is no more than
    ``best_distance + tolerance_m`` are valid.  ``adjacent_valid_mask`` marks
    the subset sharing at least one mesh vertex with the exact nearest face;
    this explicitly identifies shared-edge/shared-vertex ambiguity.
    """

    if tolerance_m < 0:
        raise ValueError("triangle target tolerance must be non-negative")
    nearest = closest_points_on_triangle_mesh(
        points, vertices, faces, point_chunk_size=point_chunk_size
    )
    distances = point_triangle_distance_matrix(
        points, vertices, faces, point_chunk_size=point_chunk_size
    )
    best = nearest["distances"]
    valid = distances <= best[:, None] + float(tolerance_m) + 1e-12
    valid.scatter_(1, nearest["face_ids"][:, None], True)
    mesh_faces = torch.as_tensor(faces, dtype=torch.long, device=valid.device)
    exact_vertices = mesh_faces[nearest["face_ids"]]
    adjacent_rows: list[Tensor] = []
    for start in range(0, len(points), int(point_chunk_size)):
        exact = exact_vertices[start : start + int(point_chunk_size)]
        shares_vertex = mesh_faces[None, :, :, None].eq(
            exact[:, None, None, :]
        ).any(-1).any(-1)
        adjacent_rows.append(
            shares_vertex & valid[start : start + int(point_chunk_size)]
        )
    return {
        **nearest,
        "all_triangle_distances": distances,
        "valid_triangle_mask": valid,
        "adjacent_valid_mask": torch.cat(adjacent_rows),
    }


def local_valid_triangle_mask(
    candidate_global_ids: Tensor, global_valid_mask: Tensor
) -> Tensor:
    """Map a global valid set to padded local candidate positions."""

    if candidate_global_ids.shape[:-1] != global_valid_mask.shape[:-1]:
        raise ValueError("candidate and global valid point dimensions disagree")
    valid_id = candidate_global_ids.ge(0)
    gathered = global_valid_mask.gather(-1, candidate_global_ids.clamp_min(0).long())
    return gathered & valid_id


def deduplicate_candidate_ids(
    candidate_ids: Tensor,
    candidate_patch_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Stable per-row deduplication with ``-1`` padding and a validity mask."""

    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must have shape [N,L]")
    if candidate_patch_ids is not None and candidate_patch_ids.shape != candidate_ids.shape:
        raise ValueError("candidate patch ids must match candidate ids")
    width = candidate_ids.shape[1]
    # Sort by id only to locate the first original occurrence of every value,
    # then sort those occurrence positions.  This is stable-equivalent to the
    # former Python ``seen`` loop but performs two batched tensor sorts instead
    # of millions of CUDA scalar synchronizations.
    sorted_ids, sorted_positions = candidate_ids.sort(dim=1, stable=True)
    is_first = sorted_ids.ge(0)
    if width > 1:
        is_first[:, 1:] &= sorted_ids[:, 1:].ne(sorted_ids[:, :-1])
    sentinel = torch.full_like(sorted_positions, width)
    first_positions = torch.where(is_first, sorted_positions, sentinel)
    stable_positions = first_positions.sort(dim=1).values
    mask = stable_positions.lt(width)
    gather_positions = stable_positions.clamp_max(max(width - 1, 0))
    output = candidate_ids.gather(1, gather_positions).masked_fill(~mask, -1)
    patch_output = None
    if candidate_patch_ids is not None:
        patch_output = candidate_patch_ids.gather(1, gather_positions).masked_fill(
            ~mask, -1
        )
    return output, mask, patch_output


def inject_valid_triangle_ids(
    candidate_ids: Tensor,
    candidate_mask: Tensor,
    global_valid_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministically include every valid triangle that fits in local width."""

    if candidate_ids.ndim != 2 or candidate_mask.shape != candidate_ids.shape:
        raise ValueError("candidate ids and mask must have shape [N,L]")
    if global_valid_mask.ndim != 2 or len(global_valid_mask) != len(candidate_ids):
        raise ValueError("global valid mask must have shape [N,F]")
    output = candidate_ids.clone()
    mask = candidate_mask.clone()
    injected = torch.zeros_like(candidate_mask)
    rows, width = output.shape
    face_count = global_valid_mask.shape[1]
    if width == 0:
        if bool(global_valid_mask.any()):
            raise ValueError("valid triangle set exceeds local candidate capacity")
        return output, mask, injected
    valid_count = global_valid_mask.sum(-1)
    if bool(valid_count.gt(width).any()):
        raise ValueError("valid triangle set exceeds local candidate capacity")

    face_ids = torch.arange(face_count, device=output.device).expand(rows, -1)
    required_width = min(width, face_count)
    required = face_ids.masked_fill(~global_valid_mask, face_count).topk(
        required_width, dim=1, largest=False, sorted=True
    ).values
    if required_width < width:
        required = torch.nn.functional.pad(
            required, (0, width - required_width), value=face_count
        )
    required_mask = required.lt(face_count)
    already_present = (
        required[:, :, None].eq(output[:, None, :])
        & mask[:, None, :]
        & required_mask[:, :, None]
    ).any(-1)
    missing_mask = required_mask & ~already_present
    missing_rank = missing_mask.cumsum(-1) - 1
    packed_missing = output.new_full((rows, width), -1)
    row_ids = torch.arange(rows, device=output.device)[:, None].expand(-1, width)
    packed_missing[
        row_ids[missing_mask], missing_rank[missing_mask]
    ] = required[missing_mask]
    missing_count = missing_mask.sum(-1)

    current_valid = global_valid_mask.gather(
        1, output.clamp_min(0).long()
    ) & mask
    columns = torch.arange(width, device=output.device)[None].expand(rows, -1)
    # Free slots first from left to right, then occupied invalid slots from the
    # tail.  Existing valid targets receive the largest priority and therefore
    # can never be evicted.
    priority = torch.where(
        ~mask,
        columns,
        torch.where(
            ~current_valid,
            width + (width - 1 - columns),
            torch.full_like(columns, 3 * width),
        ),
    )
    destination = priority.argsort(-1)
    assign = columns < missing_count[:, None]
    if bool(assign.any()):
        destination_columns = destination[assign]
        destination_rows = row_ids[assign]
        output[destination_rows, destination_columns] = packed_missing[assign]
        mask[destination_rows, destination_columns] = True
        injected[destination_rows, destination_columns] = True
    return output, mask, injected


__all__ = [
    "closest_barycentric_on_triangles",
    "deduplicate_candidate_ids",
    "inject_valid_triangle_ids",
    "local_valid_triangle_mask",
    "point_triangle_distance_matrix",
    "triangle_target_sets",
]
