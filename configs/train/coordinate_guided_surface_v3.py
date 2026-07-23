"""Portable fp32 production training for one known template."""

runtime = "production"
config_role = "train"
initialization_mode = "scratch"
seed = 0

experiment = dict(name="coordinate_guided_surface_v3")

data = dict(
    # Set with --cfg-options data.dataset_root=... or FRAG_DATASET_ROOT.
    dataset_root=None,
    train=dict(
        type="SplitDirectoryFragmentDataset",
        min_num_faces=840,
        min_observed_shell_points=128,
        max_observed_shell_points=4096,
        point_sampling="random_up_to_max",
        boundary_augmentation=dict(
            enabled=True,
            apply_probability=0.6,
            mode="random",
            mode_probabilities=dict(
                none=0.10, erode=0.35, dilate=0.35, mixed=0.20,
            ),
            radius_px=dict(min=1, max=2),
            max_removed_fraction=0.08,
            max_added_fraction=0.05,
            min_points_after_augmentation=128,
            partial_boundary_probability=0.7,
            boundary_arc_fraction_range=(0.15, 0.50),
            max_pseudo_target_distance_m=0.002,
            max_local_depth_difference_m=0.010,
            local_depth_window_px=3,
            include_fracture_candidates=True,
            include_depth_ring_candidates=True,
        ),
    ),
    validation=dict(
        type="SplitDirectoryFragmentDataset",
        min_num_faces=840,
        min_observed_shell_points=128,
        max_observed_shell_points=4096,
        point_sampling="farthest_point_up_to_max",
        boundary_augmentation=dict(enabled=False),
    ),
    train_batch_size=4,
    validation_batch_size=2,
    effective_batch_size=16,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=False,
    bucket_size_multiplier=20,
)

collate = dict(type="FragmentTemplateCollator", mode="padded")

model = dict(
    type="CoordinateGuidedSurfaceRegistrationV3",
    embed_dim=256,
    max_observed_tokens=256,
    max_template_tokens=512,
    final_coordinate_initialization_std=1e-3,
    shared_template_encoding=True,
    static_geometry_cache=False,
    observed_encoder=dict(
        type="SimplePointEncoder", embed_dim=256, hidden_dim=128,
        num_neighbors=12, dropout=0.0,
    ),
    template_encoder=dict(
        type="SimplePointEncoder", embed_dim=256, hidden_dim=128,
        num_neighbors=12, dropout=0.0,
    ),
    interaction_transformer=dict(
        type="RegTRInteractionTransformer", embed_dim=256, num_heads=8,
        num_layers=4, feedforward_dim=512, dropout=0.0,
    ),
    dual_stream_geometry_encoder=dict(
        type="DualStreamGeometryEncoder", embed_dim=256, matching_only=True,
        matching_geometric_embedding=dict(
            type="GeometricStructureEmbedding", embed_dim=256,
            num_neighbors=8, distance_scale_m=0.01,
        ),
        matching_ppf_embedding=dict(
            type="LocalPointPairFeatureEmbedding", embed_dim=256,
            num_neighbors=8, distance_scale_m=0.01,
        ),
    ),
    fine_feature_adapter=dict(
        type="FineLocalCorrespondenceFeatureAdapter", embed_dim=256,
        knn_scales=(8, 16, 32), observed_only=True,
    ),
    canonical_coordinate_head=dict(
        type="FineCanonicalCoordinateAuxiliaryHead", embed_dim=256,
        hidden_dim=256,
    ),
    weighted_procrustes=dict(
        type="WeightedProcrustes", minimum_effective_points=3.0,
        rank_tolerance=1e-7, fail_on_degenerate=False,
    ),
)

loss = dict(
    type="CleanCoordinatePoseLossV3",
    enabled=True,
    coordinate_mean_weight=1.0,
    coordinate_tail_weight=0.5,
    pose_rotation_weight=0.25,
    pose_translation_weight=0.25,
    alignment_weight=0.25,
    rotation_scale_deg=1.0,
    translation_scale_m=0.001,
    alignment_scale_m=0.001,
    warmup_epochs=1000,
    tail_fraction=0.10,
    so2_samples=36,
    loss_reduction="per_sample_mean_then_batch_mean",
    vectorized=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
    prefix_learning_rates=dict(
        observed_encoder=1e-4,
        template_encoder=1e-4,
        interaction_transformer=1e-4,
        dual_stream_geometry_encoder=1e-4,
        dense_observed_fine_projection=3e-4,
        fine_template_projection=3e-4,
        template_context_projection=3e-4,
        fine_feature_adapter=3e-4,
        canonical_coordinate_head=3e-4,
    ),
)

scheduler = dict(
    type="cosine",
    warmup_optimizer_steps=1000,
    min_lr=1e-6,
)

train = dict(
    max_epochs=150,
    max_optimizer_steps=100000,
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0,
    amp=False,
    eval_interval_optimizer_steps=1000,
    latest_checkpoint_interval_optimizer_steps=500,
)

validation = dict(
    max_batches=None,
    evaluation_role="validation",
)
evaluation = dict(candidate_k=16, projection_chunk_size=64)
