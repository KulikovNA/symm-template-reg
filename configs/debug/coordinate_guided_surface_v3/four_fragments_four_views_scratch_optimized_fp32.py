"""Exact-fp32 performance path for the fixed four-fragment/four-view run."""

_base_ = ["four_fragments_four_views_scratch.py"]

experiment = dict(
    name="coordinate_guided_surface_v3_four_fragments_four_views_optimized_fp32",
)

optimization_mode = "exact_fp32"
semantic_model_changes = False
loss_changes = False
point_count_changes = False

model = dict(
    shared_template_encoding=True,
    static_geometry_cache=dict(
        observed_encoder_neighbors=12,
        template_encoder_neighbors=12,
        observed_tokens=256,
        template_tokens=512,
        geometry_neighbors=8,
        fine_neighbors=32,
    ),
)

loss = dict(
    joint_surface_correspondence_pose_v3=dict(vectorized=True),
)

static_geometry_cache = dict(
    enabled=True,
    schema_version="static-geometry-v1",
    cache_learned_features=False,
)
shared_template_encoding = True
vectorized_symmetry_loss = True

data = dict(
    train_batch_size=16,
    effective_views_per_optimizer_step=16,
)

train = dict(
    gradient_accumulation_steps=1,
    amp=False,
    torch_compile=False,
)

performance_logging = dict(
    progress_update_interval_steps=10,
    scalar_log_interval_steps=10,
    per_module_gradient_norm_interval_steps=100,
    train_physical_metric_interval_steps=50,
    gpu_memory_metric_interval_steps=100,
)

frozen_feature_cache = dict(enabled=False, require_passing_audit=False)
augmentations = dict(enabled=False)
