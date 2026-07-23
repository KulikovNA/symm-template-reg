_base_ = ["_base_/runtime.py"]

dataset_root = (
    "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/"
    "fragment_template_registration/differBig/2026-07-08/test"
)

debug_training_on_test_split = True
results_are_not_final_evaluation = True

fragment_mesh_filter = dict(
    enabled=False,
    min_num_faces=840,
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

observed_filter = dict(
    min_observed_points=128,
    max_observed_points=4096,
    point_policy="deterministic_all_or_geometric_cap",
)

data = dict(
    fragment_mesh_filter=fragment_mesh_filter,
    observed_filter=observed_filter,
)

dataset = dict(
    type="FragmentTemplateRegistrationDataset",
    dataset_root=dataset_root,
    fragment_mesh_filter=fragment_mesh_filter,
    observed_filter=observed_filter,
    min_observed_points=128,
    max_observed_points=4096,
    voxel_size_m=0.002,
    template_fine_points=2048,
    template_coarse_points=512,
)

model = dict(
    type="SymmTemplateReg",
    embed_dim=256,
    max_observed_tokens=128,
    max_template_tokens=128,
    observed_encoder=dict(
        type="SimplePointEncoder",
        embed_dim=256,
        hidden_dim=128,
        num_neighbors=12,
    ),
    template_encoder=dict(
        type="SimplePointEncoder",
        embed_dim=256,
        hidden_dim=128,
        num_neighbors=12,
    ),
    geometric_embedding=dict(
        type="GeometricStructureEmbedding",
        embed_dim=256,
        num_neighbors=8,
        distance_scale_m=0.01,
    ),
    coarse_matcher=dict(type="CoarseMatching", temperature=0.1, mutual=True),
    interaction_transformer=dict(
        type="RegTRInteractionTransformer",
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=512,
        dropout=0.0,
    ),
    overlap_head=dict(type="OverlapHead", embed_dim=256),
    template_visibility_head=dict(type="OverlapHead", embed_dim=256),
    correspondence_head=dict(type="CorrespondenceHead", embed_dim=256),
    point_weight_head=dict(type="PointWeightHead", embed_dim=256),
    symmetry_head=dict(type="SymmetryRegionHead", embed_dim=256, max_regions=16),
    symmetry_expander=dict(type="SymmetryHypothesisExpander", so2_num_samples=36),
    pose_head=dict(
        type="PoseQueryHead",
        embed_dim=256,
        num_heads=8,
        num_queries=8,
        num_decoder_layers=3,
        feedforward_dim=512,
        uncertainty_dim=6,
        dropout=0.0,
        pose_representation=dict(type="PoseRepresentation"),
        pose_codec=dict(type="PoseCodec", min_scale_m=1e-6),
    ),
)

loss = dict(
    type="PoseSetLoss",
    translation_weight=10.0,
    rotation_weight=1.0,
    classification_weight=1.0,
    auxiliary_weight=0.5,
)
