"""Independent distance-preservation consistency formula, not in the baseline.

Reference inspected: PointDSC (https://github.com/XuyangBai/PointDSC), commit
b009d536ac10b570853833f2178397c154745da9, paths ``models/PointDSC.py`` and
``models/common.py``. License: NOASSERTION (no repository-level grant found).
Direct source port was rejected and no source text was copied. This file contains
only an independently written distance-discrepancy Gaussian interface.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from symm_template_reg.registry import MATCHERS


@MATCHERS.register_module()
class SpatialConsistency(nn.Module):
    def __init__(self, sigma_m: float = 0.01) -> None:
        super().__init__()
        self.sigma_m = sigma_m

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        source_dist = torch.cdist(source, source)
        target_dist = torch.cdist(target, target)
        discrepancy = (source_dist - target_dist).abs()
        return torch.exp(-(discrepancy / max(self.sigma_m, 1e-8)) ** 2)
