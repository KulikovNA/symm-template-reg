"""F0 metadata for the standalone capacity audits; no model training."""

experiment = dict(name="fine_capacity_v2_frame04_AUDIT_ONLY")
audit = dict(
    checkpoint="/home/nikita/disser/fragment-template-registration-lab/work_dirs/v4_patch_classifier_frame04_20260720_113703/checkpoints/best.pth",
    manifest="/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/joint_correspondence_pose/fragment0002_frame04_only.json",
    max_steps=1500,
    eval_every=50,
    learning_rates=(1e-3, 3e-4, 1e-4),
    max_points=1024,
)

