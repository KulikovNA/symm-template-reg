"""Optional AMP ablation; never selected by the fp32 performance workflow."""

_base_ = ["four_fragments_four_views_scratch_optimized_fp32.py"]

experiment = dict(
    name="coordinate_guided_surface_v3_four_fragments_four_views_optimized_amp",
)
optimization_mode = "optional_amp_ablation"

train = dict(
    amp=True,
    amp_dtype="bfloat16",
    torch_compile=False,
)
