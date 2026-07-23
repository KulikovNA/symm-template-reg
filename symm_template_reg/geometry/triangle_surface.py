"""Dependency-free closest-point and barycentric operations on triangle meshes."""

from __future__ import annotations
import torch
from torch import Tensor


def barycentric_points(triangle_vertices: Tensor, barycentric: Tensor) -> Tensor:
    """Return convex triangle points; input weights are normalized safely."""
    weights = torch.as_tensor(barycentric).clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    triangles = torch.as_tensor(
        triangle_vertices, dtype=weights.dtype, device=weights.device
    )
    return torch.sum(weights.unsqueeze(-1) * triangles, dim=-2)


def closest_points_on_triangle_mesh(
    points: Tensor,
    vertices: Tensor,
    faces: Tensor,
    *,
    point_chunk_size: int = 32,
) -> dict[str, Tensor]:
    """Exact closest point among face interiors, edges and vertices.

    The implementation chunks query points and vectorizes over faces.  It is
    intended for deterministic audits/cache construction, not model forward.
    """
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
    all_points, all_faces, all_bary, all_distances = [], [], [], []
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

        candidates = [plane]
        bary_candidates = [torch.stack((u, v, w), -1)]
        for x, y, bary_x, bary_y in (
            (a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)
        ):
            edge = y - x
            t = ((p - x) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-20)
            t = t.clamp(0, 1)
            candidates.append(x + t[..., None] * edge)
            bary = torch.zeros((*t.shape, 3), dtype=query.dtype, device=query.device)
            bary[..., bary_x] = 1.0 - t
            bary[..., bary_y] = t
            bary_candidates.append(bary)
        candidate_points = torch.stack(candidates, dim=2)
        candidate_bary = torch.stack(bary_candidates, dim=2)
        distance2 = (candidate_points - p[:, :, None]).square().sum(-1)
        distance2[:, :, 0] = distance2[:, :, 0].masked_fill(~inside, float("inf"))
        local = distance2.argmin(dim=2)
        best_per_face = distance2.gather(2, local.unsqueeze(-1)).squeeze(-1)
        face_id = best_per_face.argmin(dim=1)
        row = torch.arange(len(p), device=query.device)
        local_id = local[row, face_id]
        all_points.append(candidate_points[row, face_id, local_id])
        all_bary.append(candidate_bary[row, face_id, local_id])
        all_faces.append(face_id)
        all_distances.append(best_per_face[row, face_id].clamp_min(0).sqrt())
    return {
        "points": torch.cat(all_points),
        "face_ids": torch.cat(all_faces),
        "barycentric": torch.cat(all_bary),
        "distances": torch.cat(all_distances),
    }


def nearest_triangles_on_mesh(
    points: Tensor,
    vertices: Tensor,
    faces: Tensor,
    k: int,
    *,
    point_chunk_size: int = 32,
) -> dict[str, Tensor]:
    """Return the exact K nearest faces for every query point.

    Distance is measured to the complete triangle (interior, edges and
    vertices), not to its centroid.  Candidate selection is discrete, so this
    helper is intended to be called on detached coarse predictions.
    """
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
    count = min(max(int(k), 1), len(mesh_faces))
    all_faces, all_distances = [], []
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
            closest = x + t.clamp(0, 1)[..., None] * edge
            distance2 = torch.minimum(distance2, (closest - p).square().sum(-1))
        values, indices = distance2.topk(count, dim=1, largest=False, sorted=True)
        all_faces.append(indices)
        all_distances.append(values.clamp_min(0).sqrt())
    return {
        "face_ids": torch.cat(all_faces),
        "distances": torch.cat(all_distances),
    }


__all__ = [
    "barycentric_points",
    "closest_points_on_triangle_mesh",
    "nearest_triangles_on_mesh",
]
