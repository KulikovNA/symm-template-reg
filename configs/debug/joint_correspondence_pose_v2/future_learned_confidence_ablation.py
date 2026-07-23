"""Future-only ablation. Do not run before the 8-view uniform gate passes."""

_base_ = ["03_uniform_joint_8views.py"]
experiment = dict(name="future_learned_confidence_ablation_DO_NOT_RUN")
model = dict(
    weighting_mode="learned_confidence_ablation",
    point_weight_head=dict(type="PointWeightHead", embed_dim=256),
)
stage = dict(name="future_learned_confidence_ablation_DO_NOT_RUN")
future_ablation_policy = dict(
    enabled_for_current_runbook=False,
    requires_successful_uniform_8view_stage_gate=True,
    anti_collapse_constraints=dict(
        minimum_effective_correspondence_fraction=0.5,
        maximum_correspondence_weight=0.05,
        entropy_regularization_required=True,
    ),
)
