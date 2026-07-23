"""Optional PTv3 registration point.

The audited upstream implementation relies on optional sparse/CUDA packages.
It is intentionally not imported by the guaranteed baseline.
"""

from torch import nn

from symm_template_reg.registry import BACKBONES


@BACKBONES.register_module(name="PTV3Encoder")
class PTV3Encoder(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()
        raise RuntimeError(
            "PTV3Encoder is optional and unavailable in the dependency-free baseline; "
            "see docs/ENVIRONMENT_COMPATIBILITY.md"
        )

