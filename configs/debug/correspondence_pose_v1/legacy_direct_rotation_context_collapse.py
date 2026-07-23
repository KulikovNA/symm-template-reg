"""Reproducibility wrapper; the inherited conditioned_pose_v2 file is untouched."""

_base_ = ["../conditioned_pose_v2/01_k1_direct_equal_exposure.py"]

experiment = dict(name="legacy_direct_failed_rotation_context_collapse")
research_status = dict(
    status="failed_rotation_context_collapse",
    recommended_for_new_runs=False,
    continuing_same_training_recommended=False,
)
plateau_detection = dict(
    enabled=True,
    min_sample_exposures=300,
    patience_eval_records=10,
    metric_min_delta=1e-4,
    detect_static_rotation=True,
    max_rotation_response_ratio=0.01,
    max_predicted_pairwise_rotation_deg=1.0,
    min_gt_pairwise_rotation_deg=10.0,
    min_static_fraction=0.9,
    action="stop_with_diagnosis",
)
