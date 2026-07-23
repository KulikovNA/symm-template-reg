"""Geometry-first correspondence -> Weighted Procrustes research base."""

_base_ = ["../conditioned_pose_v2/_base.py"]

experiment = dict(name="correspondence_pose_v1_DO_NOT_RUN")

data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/view_ladder/frames04_05_02_08.json",
    validation_manifest="same_as_train",
    expected_selected_samples=4,
)

model = dict(
    base_pose_source="weighted_procrustes",
    correspondence_only=False,
    dual_stream_geometry_encoder=dict(
        matching_geometric_only=True,
        matching_ppf_embedding=dict(
            type="LocalPointPairFeatureEmbedding",
            embed_dim=256,
            num_neighbors=8,
            distance_scale_m=0.01,
        ),
    ),
    weighted_procrustes=dict(
        type="WeightedProcrustes",
        minimum_effective_points=3.0,
        rank_tolerance=1e-7,
        fail_on_degenerate=False,
    ),
    residual_pose_head=dict(num_hypotheses=1),
)

loss = dict(
    pose_query_ranking=dict(weight=0.0),
    observed_region_weight=0.0,
    active_region_weight=0.0,
    cross_view_world_consistency=dict(enabled=False),
    pairwise_pose_response=dict(enabled=False),
    correspondence_loss=dict(
        enabled=True,
        weight=1.0,
        robust_type="smooth_l1",
        beta=0.01,
        use_shared_symmetry_element=True,
        confidence_regularization=dict(
            enabled=True,
            weight=0.01,
            minimum_effective_point_count=16.0,
            minimum_weight_sum=1e-3,
        ),
    ),
    hybrid_direct_residual=dict(enabled=False),
)

train_budget = dict(mode="sample_exposures", target_exposures_per_sample=1500)

train = dict(
    max_epochs=15000,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    min_sample_exposures_before_early_stop=750,
)

augmentation = dict(enabled=False)
multi_view_batch = dict(enabled=False)
target_leakage_policy = dict(forbid_detected=True, audit_path=None)

diagnostic_gates = dict(
    enabled=True,
    action="stop_with_diagnosis",
    min_sample_exposures=300,
    correspondence_constant_output=dict(enabled=True, min_pairwise_distance_m=1e-5),
    confidence_collapse=dict(enabled=True, minimum_effective_point_count=16.0),
    procrustes_rank_failure=dict(enabled=True),
    correspondence_target_leakage=dict(enabled=True),
    residual_static_codebook=dict(enabled=False),
    residual_bound_saturation=dict(enabled=False),
)

stage = dict(
    readiness_thresholds=dict(
        correspondence_point_p95_mm=2.0,
        correspondence_point_rmse_mm=1.0,
        minimum_effective_correspondence_count=16.0,
        require_no_target_leakage=True,
        require_all_views_finite=True,
    )
)
