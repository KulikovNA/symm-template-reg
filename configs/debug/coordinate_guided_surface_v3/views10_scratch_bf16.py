"""Prepared bf16 override; requires a separate numerical-equivalence smoke."""

_base_ = ["views10_scratch_full.py"]
experiment = dict(name="coordinate_guided_surface_v3_views10_scratch_bf16")
train = dict(amp=True, amp_dtype="bf16")
