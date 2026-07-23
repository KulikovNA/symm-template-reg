"""Dataset public API.

Imports are deliberately tolerant during package bootstrap: registry-backed
registration is performed when the root registry is available, while the
classes remain directly importable in standalone dataset tools.
"""

from .collate import (
    FragmentTemplateCollator,
    build_collate_fn,
    fragment_template_collate,
    packed_collate,
    padded_collate,
)
from .boundary_augmentation import (
    BoundaryMaskAugmentation,
    DEFAULT_BOUNDARY_AUGMENTATION,
)
from .fragment_template_dataset import (
    FragmentTemplateRegistrationDataset,
    resolve_split_root,
)
from .split_directory_fragment_dataset import SplitDirectoryFragmentDataset
from .fragment_mesh_filter import (
    FragmentFilterDecision,
    FragmentMeshFilter,
    FragmentMeshMetadata,
    FragmentMeshMetadataCache,
    scan_fragment_mesh_metadata,
)
from .samplers import LengthBucketBatchSampler
from .multi_view_batch_sampler import MultiViewBatchSampler
from .structures import DatasetSampleRecord, PackedPointBatch
from .template_repository import TemplateRepository, load_ply
from .transforms import ObservedPointSelector

__all__ = [
    "DatasetSampleRecord",
    "BoundaryMaskAugmentation",
    "DEFAULT_BOUNDARY_AUGMENTATION",
    "FragmentTemplateRegistrationDataset",
    "SplitDirectoryFragmentDataset",
    "FragmentFilterDecision",
    "FragmentMeshFilter",
    "FragmentMeshMetadata",
    "FragmentMeshMetadataCache",
    "FragmentTemplateCollator",
    "LengthBucketBatchSampler",
    "MultiViewBatchSampler",
    "ObservedPointSelector",
    "PackedPointBatch",
    "TemplateRepository",
    "build_collate_fn",
    "fragment_template_collate",
    "load_ply",
    "packed_collate",
    "padded_collate",
    "resolve_split_root",
    "scan_fragment_mesh_metadata",
]
