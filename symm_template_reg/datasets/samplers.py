"""Optional length-aware batching for variable-size point clouds."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Sort local pools by point count to reduce padded-batch waste."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths = [int(value) for value in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        size = len(self.lengths) // self.batch_size
        if not self.drop_last and len(self.lengths) % self.batch_size:
            size += 1
        return size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


__all__ = ["LengthBucketBatchSampler"]
