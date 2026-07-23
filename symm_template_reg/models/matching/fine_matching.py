from __future__ import annotations

from torch import Tensor, nn

from symm_template_reg.registry import MATCHERS


@MATCHERS.register_module()
class FineMatching(nn.Module):
    def __init__(self, topk: int = 3) -> None:
        super().__init__()
        self.topk = topk

    def forward(self, scores: Tensor, template_points: Tensor) -> tuple[Tensor, Tensor]:
        values, indices = scores.topk(min(self.topk, scores.shape[-1]), dim=-1)
        weights = values.softmax(-1)
        batch = torch.arange(scores.shape[0], device=scores.device)[:, None, None]
        selected = template_points[batch, indices]
        return (selected * weights.unsqueeze(-1)).sum(-2), weights.max(-1).values

