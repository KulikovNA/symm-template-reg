_base_ = ["four_fragments_four_views_scratch.py"]
experiment = dict(name="audit_static_geometry_fp32")
model = dict(static_geometry_cache=dict(
    observed_encoder_neighbors=12,
    template_encoder_neighbors=12,
    observed_tokens=256,
    template_tokens=512,
    geometry_neighbors=8,
    fine_neighbors=32,
))
static_geometry_cache = dict(enabled=True, cache_learned_features=False)

