"""Deterministic conditioned-pose debug base; ranking and regions are disabled."""

_base_ = ["../view_ladder/_base.py"]

experiment = dict(name="conditioned_pose_base_DO_NOT_RUN")

model = dict(
    _delete_=True,
    type="ConditionedSymmTemplateReg",
    embed_dim=256,
    max_observed_tokens=128,
    max_template_tokens=128,
    observed_encoder=dict(
        type="SimplePointEncoder",
        embed_dim=256,
        hidden_dim=128,
        num_neighbors=12,
        dropout=0.0,
    ),
    template_encoder=dict(
        type="SimplePointEncoder",
        embed_dim=256,
        hidden_dim=128,
        num_neighbors=12,
        dropout=0.0,
    ),
    interaction_transformer=dict(
        type="RegTRInteractionTransformer",
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        feedforward_dim=512,
        dropout=0.0,
    ),
    dual_stream_geometry_encoder=dict(
        type="DualStreamGeometryEncoder",
        embed_dim=256,
        matching_geometric_embedding=dict(
            type="GeometricStructureEmbedding",
            embed_dim=256,
            num_neighbors=8,
            distance_scale_m=0.01,
        ),
    ),
    sample_context_aggregator=dict(
        type="SampleConditionedContextAggregator",
        embed_dim=256,
        hidden_dim=512,
        aggregation="masked_attention_pooling",
    ),
    base_pose_head=dict(
        type="ConditionedBasePoseHead",
        embed_dim=256,
        hidden_dim=512,
        uncertainty_dim=6,
        pose_codec=dict(type="PoseCodec", min_scale_m=1e-6),
    ),
    residual_pose_head=dict(
        type="ResidualPoseHypothesisHead",
        embed_dim=256,
        num_heads=8,
        num_hypotheses=1,
        num_decoder_layers=3,
        feedforward_dim=512,
        uncertainty_dim=6,
        dropout=0.0,
        query_conditioning=dict(
            type="film",
            apply_each_decoder_layer=True,
            allow_unconditioned_bypass=False,
        ),
    ),
    overlap_head=dict(type="OverlapHead", embed_dim=256),
    template_visibility_head=dict(type="OverlapHead", embed_dim=256),
    correspondence_head=dict(type="CorrespondenceHead", embed_dim=256),
    point_weight_head=dict(type="PointWeightHead", embed_dim=256),
    symmetry_head=None,
    weighted_procrustes=None,
)

loss = dict(
    symmetry_pose_weight=0.0,
    pose_query_classification_weight=0.0,
    pose_query_ranking=dict(weight=0.0),
    observed_region_weight=0.0,
    active_region_weight=0.0,
    region_consistency_weight=0.0,
    correspondence_weight=0.0,
    auxiliary_registration_losses=False,
    pose_decoder_auxiliary_weight=0.0,
    translation_cost_weight=10.0,
    rotation_cost_weight=1.0,
    conditioned_pose_loss=dict(
        base_pose_weight=1.0,
        best_residual_pose_weight=0.0,
        residual_regularization_weight=0.01,
    ),
    correspondence_loss=dict(
        enabled=False,
        weight=1.0,
        robust_type="smooth_l1",
        use_shared_symmetry_element=True,
    ),
    correspondence_pose_loss_weight=0.0,
    direct_vs_correspondence_pose_consistency_weight=0.0,
)

train = dict(
    max_optimizer_steps=1500,
    max_epochs=1500,
    optimizer=dict(type="AdamW", lr=3e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    gradient_accumulation_steps=1,
    amp=False,
    early_stopping_patience_evals=0,
    eval_interval_optimizer_steps=50,
    debug_visualization_interval_optimizer_steps=250,
)

stage = dict(
    name="conditioned_pose_base",
    trainable_module_prefixes=None,
    checkpoint_filename="best_conditioned_pose.pth",
    readiness_thresholds=dict(),
)
