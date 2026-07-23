"""Prepared two-view coordinate-control overfit; no later heads are trained."""

_base_ = ["frame08.py"]
experiment = dict(name="coordinate_guided_surface_v2_views02")
data = dict(
    train_manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_views02_shell_only.json",
    validation_manifest="same_as_train", expected_selected_samples=2,
    train_batch_size=2, validation_batch_size=2,
)
dataset = dict(random_seed=0)
augmentations = dict(enabled=False)
train_budget = dict(mode="epochs", epochs=1500)
stage = dict(
    name="TWO_VIEW_coordinate_guided_surface_v2_fine_only",
    initialization="model_only_via_cli_from_selected_two_view_initialization",
    strict_initialization=True,
    trainable_module_prefixes=(
        "correspondence_head.fine_feature_adapter",
        "correspondence_head.fine_coordinate_auxiliary_head",
    ),
    prefix_learning_rates={
        "correspondence_head.fine_feature_adapter": 1e-4,
        "correspondence_head.fine_coordinate_auxiliary_head": 1e-4,
    },
)
train = dict(
    max_epochs=1500,
    gradient_accumulation_steps=1,
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
    scheduler=dict(type="constant"),
    amp=False,
    eval_interval_epochs=25,
    debug_visualization_interval_epochs=250,
    evaluate_before_training=True,
    visualize_before_training=True,
    best_metric="eval/worst_sample_projection_score",
    best_metric_mode="min",
    best_metric_tie_breaker="eval/all_samples_projection_gate_passed",
    best_metric_tie_breaker_mode="max",
)
loss = dict(joint_surface_correspondence_pose_v3=dict(
    fine_coordinate_aux_weight=1.0,
    fine_coordinate_tail_weight=0.5,
    raw_pose_rotation_weight=0.25,
    raw_pose_translation_weight=0.25,
    raw_alignment_weight=0.25,
    raw_pose_rotation_scale_deg=1.0,
    raw_pose_translation_scale_m=0.001,
    raw_alignment_scale_m=0.001,
))
coordinate_guided_surface_v2 = dict(
    projection_mode="exact_global", candidate_mode="aux_guided_global_topk",
    candidate_k=16, projection_chunk_size=256, fallback_to_global_exact=True,
    evaluate_modes=("exact_global", "aux_guided_global_topk"),
)
model = dict(correspondence_head=dict(
    # Legacy local candidates are not a supervised/output path here.  Avoid
    # GT-triangle injection capacity constraints in the mixed-view batch.
    inject_all_valid_triangles=False,
    teacher_force_exact_triangle=False,
))
disabled_paths = dict(
    ranking=True, regions=True, K8=True, learned_confidence=True,
    direct_pose_head=True, learned_triangle=True, learned_barycentric=True,
)
debug_visualization = dict(
    num_samples=2, single_fragment_layout=True, joint_correspondence_pose=True,
    coordinate_guided_primary=True,
    required_epochs=(0,250,500,750,1000,1250,1500),
)
