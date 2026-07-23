"""Frame-4 correctness/reference config; use the refined F1 checkpoint."""

_base_ = ["../fine_correspondence_v1/01_fine_adapter_coordinate_control_frame04_refine.py"]
experiment = dict(name="coordinate_guided_surface_v2_frame04")
coordinate_guided_surface_v2 = dict(
    type="CoordinateGuidedSurfaceCorrespondenceV2",
    projection_mode="exact_global",
    candidate_mode="aux_guided_global_topk",
    candidate_k=16,
    projection_chunk_size=256,
    fallback_to_global_exact=True,
    learned_triangle_head=False,
    learned_barycentric_head=False,
)
