"""Resolve fair training budgets in optimizer steps and sample exposures."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResolvedTrainingBudget:
    mode: str
    selected_samples: int
    batch_size: int
    gradient_accumulation_steps: int
    drop_last: bool
    batches_per_epoch: int
    optimizer_steps_per_epoch: int
    target_sample_exposures: int | None
    computed_max_optimizer_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_training_budget(
    train_budget: Mapping[str, Any] | None,
    *,
    selected_samples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    drop_last: bool,
    configured_max_optimizer_steps: int | None,
    configured_max_epochs: int,
) -> ResolvedTrainingBudget:
    """Convert a policy into an optimizer-step ceiling.

    Runtime additionally tracks exact counts per sample, so uneven final
    batches remain visible in the run artifacts.
    """
    count = int(selected_samples)
    batch = int(batch_size)
    accumulation = int(gradient_accumulation_steps)
    if count < 1 or batch < 1 or accumulation < 1:
        raise ValueError("selected_samples, batch_size and accumulation must be positive")
    batches = count // batch if drop_last else math.ceil(count / batch)
    if batches < 1:
        raise ValueError("drop_last removes every selected sample")
    steps_per_epoch = math.ceil(batches / accumulation)
    policy = dict(train_budget or {})
    mode = str(policy.get("mode", "optimizer_steps"))
    if mode not in {"optimizer_steps", "epochs", "sample_exposures"}:
        raise ValueError(f"unsupported train_budget.mode: {mode}")
    target: int | None = None
    if mode == "optimizer_steps":
        if configured_max_optimizer_steps is None:
            raise ValueError("optimizer_steps budget requires train.max_optimizer_steps")
        computed = int(configured_max_optimizer_steps)
    elif mode == "epochs":
        epochs = int(policy.get("epochs", configured_max_epochs))
        if epochs < 1:
            raise ValueError("epochs budget must be positive")
        # A complete epoch presents every selected sample exactly once.  Keep
        # this explicit in artifacts so different nested-view stages are
        # directly comparable by per-sample exposure.
        target = epochs
        computed = epochs * steps_per_epoch
    else:
        target = int(policy.get("target_exposures_per_sample", 0))
        if target < 1:
            raise ValueError("sample_exposures budget requires a positive target")
        if drop_last and count % batch:
            raise ValueError(
                "sample_exposures with an incomplete drop_last batch cannot "
                "guarantee a minimum exposure for every selected sample"
            )
        # Complete epochs make the per-sample guarantee exact even when the
        # last batch is smaller (for example 10 views with batch size 4).
        computed = target * steps_per_epoch
    if computed < 1:
        raise ValueError("computed optimizer-step budget must be positive")
    return ResolvedTrainingBudget(
        mode=mode,
        selected_samples=count,
        batch_size=batch,
        gradient_accumulation_steps=accumulation,
        drop_last=bool(drop_last),
        batches_per_epoch=batches,
        optimizer_steps_per_epoch=steps_per_epoch,
        target_sample_exposures=target,
        computed_max_optimizer_steps=computed,
    )


def sample_exposure_statistics(
    exposures: Mapping[str, int], *, target: int | None
) -> dict[str, int | float | None]:
    values = [int(value) for value in exposures.values()]
    if not values:
        raise ValueError("sample exposure map must not be empty")
    minimum = min(values)
    mean = sum(values) / len(values)
    maximum = max(values)
    return {
        "selected_samples": len(values),
        "samples_seen": sum(values),
        "min_sample_exposures": minimum,
        "mean_sample_exposures": mean,
        "max_sample_exposures": maximum,
        "sample_exposures_min": minimum,
        "sample_exposures_mean": mean,
        "sample_exposures_max": maximum,
        "target_sample_exposures": target,
    }


def early_stopping_is_eligible(
    exposures: Mapping[str, int], minimum_exposures: int
) -> bool:
    return bool(exposures) and min(exposures.values()) >= int(minimum_exposures)


__all__ = [
    "ResolvedTrainingBudget",
    "early_stopping_is_eligible",
    "resolve_training_budget",
    "sample_exposure_statistics",
]
