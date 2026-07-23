"""Grouped batches for multiple camera views of one physical fragment."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence

from torch.utils.data import Sampler


class MultiViewBatchSampler(Sampler[list[int]]):
    """Yield local dataset indices grouped by scene and fragment identity."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, object]],
        *,
        views_per_group: int,
        group_by: Sequence[str] = ("scene_id", "fragment_id"),
        require_same_fragment_mesh: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if views_per_group < 1:
            raise ValueError("views_per_group must be positive")
        self.views_per_group = int(views_per_group)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        grouped: dict[tuple[object, ...], list[int]] = defaultdict(list)
        mesh_by_group: dict[tuple[object, ...], object] = {}
        for index, sample in enumerate(samples):
            key = tuple(sample.get(name) for name in group_by)
            if any(value is None for value in key):
                raise ValueError(f"sample is missing group key {tuple(group_by)}")
            mesh = sample.get("fragment_mesh_sha256")
            if require_same_fragment_mesh:
                previous = mesh_by_group.setdefault(key, mesh)
                if mesh is None or mesh != previous:
                    raise ValueError(f"group {key} contains different fragment meshes")
            grouped[key].append(index)
        self.groups = [
            indices for _, indices in sorted(grouped.items(), key=lambda item: str(item[0]))
        ]
        if not self.groups:
            raise ValueError("samples must not be empty")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for original in self.groups:
            indices = list(original)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.views_per_group):
                batch = indices[start : start + self.views_per_group]
                if len(batch) == self.views_per_group or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(group) // self.views_per_group for group in self.groups)
        return sum(
            (len(group) + self.views_per_group - 1) // self.views_per_group
            for group in self.groups
        )


__all__ = ["MultiViewBatchSampler"]
