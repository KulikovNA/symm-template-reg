# Debug training contract

> `debug_training_on_test_split = true`  
> `results_are_not_final_evaluation = true`

The available `2026-07-08/test` split is used only for controlled overfit and
pipeline debugging. Metrics from these runs are not a final evaluation or a
generalization claim.

## Coordinate and pose contract

All geometry uses metres and the BOP/OpenCV camera convention: X right, Y down,
Z forward. Homogeneous transforms act on column vectors. Template points are in
the object frame `O`; observed points are in camera frame `C`; `T_C_from_O`
maps object coordinates to camera coordinates.

The observed cloud is not permanently recentered before the point encoder. The
shared `PoseCodec` computes a per-sample context from valid observed points:

```text
observed_centroid_C = mean(valid observed points_C)
observed_scale      = max distance from centroid to a valid observed point
t_normalized        = (t_C_from_O - observed_centroid_C) / observed_scale
t_C_from_O          = observed_centroid_C + observed_scale * t_normalized
```

`observed_scale` is clamped to at least `1e-6 m`. Rotation is encoded by the
first two columns of its matrix (6D representation) and decoded by the shared
orthonormalizing `rotation_6d_to_matrix` implementation. Dataset targets and
the pose head use the same codec. Losses, evaluation, and visualization consume
the decoded `T_C_from_O`; they do not implement a second translation formula.

## Symmetry and pose queries

Production symmetry metadata and group operations are the only source of
symmetry semantics. A base prediction is compared as

```text
min over S in effective_group: distance(prediction, T_C_from_O @ S)
```

For `C1`, `C2`, `C4`, and `C10`, the corresponding finite group has 1, 2, 4,
or 10 equivalent poses. `SO2` is handled analytically and twist around the
configured symmetry axis is not penalized.

The eight DETR-style pose queries are eight base-pose alternatives. They are
not eight symmetry elements. Assignment selects exactly one base query using
the symmetry-aware pose cost; its query logit receives the positive
classification target. Symmetry expansion is reserved for group-aware metrics
and explicit hypothesis galleries.

## Two independent data filters

The physical fragment mesh filter runs first, while building the Dataset index.
It reads each unique `scene_XXXXXX/fragments/fragment_XXXX.ply` once through the
metadata cache and applies the configured polygon-face threshold. If a physical
fragment is rejected, every observation of it is excluded before `__len__` is
fixed. `__getitem__` and collate never skip samples dynamically.

The observed-view filter runs second and evaluates each camera observation.
Currently it enforces `min_observed_points` and deterministically keeps at most
`max_observed_points`. Physical face count and observed point count are separate
fields with separate rejection reasons.

`min_num_faces` has no hidden Dataset default. Enabling the physical filter
without setting it is an error. The threshold must be chosen manually from the
size audit, and the same filtered set is used by tiny and scene overfit runs.
Training validates the manifest sidecar hash, threshold, accepted membership,
mesh hash, and mesh face count before constructing the first batch.

## Inference policy

The physical fragment mesh is ground-truth training metadata and is unknown in
production inference. `num_faces` is never a model input and the physical mesh
filter is not an inference rejection rule. It only excludes fragments judged
too small for pose supervision.

Future inference rejection may use observed shell point count, mask area, 3D
bounds, estimated observed surface area, pose-query uncertainty, or an
insufficient-information score. In this iteration rejected physical fragments
are not negative examples for the insufficient-information head; that head is
not trained by the debug trainer.

## Reproducibility boundary

Every debug config, sample manifest, run summary, evaluation report, and
checkpoint manifest carries the two test-split warning flags. Runs save the
resolved config, manifest SHA256, physical mesh metadata/filter CSVs, template
and symmetry-sidecar hashes, environment information, and model/optimizer
state. A manifest made with another face threshold is rejected rather than
silently rebuilt.
