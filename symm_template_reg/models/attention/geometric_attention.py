"""Scaled dot-product attention with an optional geometric pair bias."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from symm_template_reg.registry import ATTENTION


@ATTENTION.register_module()
class GeometricAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_valid_mask: Tensor,
        pair_bias: Tensor | None = None,
    ) -> Tensor:
        batch, num_query, _ = query.shape
        num_key = key_value.shape[1]
        q = self.query(query).view(batch, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key_value).view(batch, num_key, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(key_value).view(batch, num_key, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pair_bias is not None:
            logits = logits + (pair_bias.unsqueeze(1) if pair_bias.ndim == 3 else pair_bias)
        logits = logits.masked_fill(~key_valid_mask[:, None, None, :], float("-inf"))
        weights = logits.softmax(-1)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch, num_query, -1)
        return self.output(output)

