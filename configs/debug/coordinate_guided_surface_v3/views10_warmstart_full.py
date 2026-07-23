"""Prepared active-module warm-start control; never launch before scratch analysis."""

_base_ = ["views10_scratch_full.py"]
initialization_mode = "warmstart_active_modules_only"
pretrained_checkpoint = None
experiment = dict(
    name="coordinate_guided_surface_v3_views10_warmstart_full",
    initialization_mode="warmstart_active_modules_only",
    pretrained_checkpoint=None,
)
stage = dict(
    initialization="explicit_active_key_mapping_from_eight_view_only",
    warmstart_requires_external_approval=True,
    warmstart_source_expected="eight_view_best_checkpoint",
    warmstart_legacy_heads_forbidden=True,
)
