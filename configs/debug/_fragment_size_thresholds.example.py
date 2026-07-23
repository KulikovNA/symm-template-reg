"""Copy a user-selected audit threshold into each debug config before training."""

fragment_mesh_filter = dict(
    enabled=True,
    # Replace only after reviewing fragment_size_audit/.../fragment_size_report.md.
    min_num_faces=None,
    max_num_faces=None,
    min_num_vertices=None,
    min_surface_area_m2=None,
    min_bbox_diagonal_m=None,
    exclude_entire_fragment=True,
    missing_mesh_policy="error",
    manifest_mismatch_policy="error",
    cache_metadata=True,
    train_policy="exclude",
    debug_eval_policy="exclude",
    validation_policy="report_only",
)
