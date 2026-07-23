"""Evaluation-only diagnostics."""

from .context_conditioning import (
    context_conditioning_metrics,
    input_permutation_equivariance_error,
)

__all__ = [
    "context_conditioning_metrics",
    "input_permutation_equivariance_error",
]
