# Debug training V2: audit and corrected semantics

## Scope

The audit followed the actual Dataset → collate → model → loss/evaluator →
visualization calls used by `run_overfit_training`; field names alone were not
treated as proof of behavior. The archived pre-V2 configuration is
`configs/debug/archive/test_overfit_faces840_gpu_pose_only_legacy_20260716.py`.

## Behavior before V2

### Pose loss and query assignment

`engine.trainer.compute_training_losses()` passed
`batch["gt"]["effective_symmetry_group"]` into `PoseSetLoss`. In
`PoseSetLoss._single_symmetry_aware()` every query was compared with the GT pose
under that GT group, using
`translation_weight * translation_m + rotation_weight * rotation_rad`; the
minimum-cost query alone received pose regression.

The old score loss was an independent `BCEWithLogits` over K queries with one
positive (the matched query) and K-1 negatives. Thus score differences did not
encode how much better or worse the other queries were.

The overfit trainer reconstructed `loss_symmetry_pose` from the translation and
rotation errors of the matched/oracle query. Consequently
`eval/symmetry_pose_loss` described top-K coverage, not the pose selected at
runtime by `argmax(pose_logits)`. Nevertheless, the old config selected the best
checkpoint by that oracle/matched value.

### Region supervision

The Dataset correctly built GT region targets from the row-aligned visible
`points_O` stored in each frame NPZ, not from the complete physical fragment
mesh. It produced point region indices, active regions, and a GT effective
group. The collator, however, discarded the point region indices.

`SymmetryRegionHead` produced both `observed_region_logits [B,N,16]` and
`active_region_logits [B,16]`. `compute_training_losses()` had only an active
region BCE branch, guarded by `auxiliary_registration_losses`. The faces840
overfit trainer forced that flag off and rejected non-zero region weights.
Therefore both region heads ran forward but received no region-supervision
gradient. No per-point region CE existed. Although the active BCE sliced to the
target width, no equivalent point-head masking existed because that loss did
not exist.

### Prediction visualization

`visualization.prediction_debug.export_prediction_visualizations()` thresholded
the untrained `active_region_logits`, forced the most probable region active
when the set was empty, and used that learned group for the main gallery.

The geometry called “predicted fragment” was loaded from the complete
`fragment_XXXX.ply`, placed in camera coordinates with the dataset-only
`gt.T_C_from_F`, and projected onto the template with the predicted top-1 pose.
It was not limited to observed visible points. The same full mesh drove the old
gallery footprint, while the group came from a different source (the learned
active head). This allowed a PLY footprint spanning C10 and C4 to coexist with
a C10 gallery. Unchanged GT footprint PLY files were regenerated in every
epoch directory.

### Data policies

The config carried `data.observed_filter.point_policy` and an independent
`dataset.observed_policy`. The Dataset silently canonicalized the former legacy
name, so a disagreement could be hidden. Validation policy was `report_only`.

## V2 changes

- `config.validate_data_policy()` and the Dataset constructor now reject
  semantically conflicting legacy/authoritative point policies. V2 keeps only
  `data.observed_filter.point_policy="farthest_point_up_to_max"` and uses
  `validation_policy="exclude"`.
- `FragmentTemplateRegistrationDataset._symmetry_targets()` consumes the shared
  explicit activity policy (`min_points=1`, `min_fraction=0`, boundary tolerance
  `1e-6 m`), preserving the actual old point-target defaults.
- `datasets.collate._stack_point_region_indices()` retains padded per-point
  targets with `-1` as ignore/OOB.
- `PoseConditionedSymmetryResolver.resolve()` transforms only visible
  `points_C` by each inverse predicted base pose and reuses production region
  assignment, group intersection, and hypothesis expansion. Empty evidence is
  explicitly unresolved and expands to one base pose.
- `engine.trainer.compute_training_losses()` adds masked point CE, valid-slot
  active BCE/focal BCE, differentiable point/global consistency, raw and
  weighted loss fields, and soft-quality query ranking. Pose regression still
  receives only the GT effective group; predicted or learned groups are never
  passed to it.
- `PoseQueryRankingLoss` builds detached per-sample min-max-normalized quality
  distributions from the same GT symmetry-aware pose cost used for assignment.
  The legacy binary and matched categorical modes remain explicit alternatives.
- `engine.metrics.batch_pose_metric_rows()` now distinguishes top-1 selected
  cost from oracle top-K cost and reports regret, oracle selection accuracy,
  score/cost Spearman correlation, point-region metrics, active-set metrics,
  learned effective-group accuracy, and hypothesis-count accuracy.
- `overfit_trainer.is_checkpoint_improvement()` uses
  `eval/top1_scored_pose_cost` first and the configured top-1 success tie breaker
  second. `best_metrics.json` records both top-1 and oracle diagnostics.
- `prediction_debug.export_prediction_visualizations()` uses the resolver for
  the main gallery, uses only observed visible points in that gallery, writes a
  separate learned-head gallery, and labels the complete physical mesh output
  as oracle-only in PLY comments and JSON. Static GT artifacts live once below
  `debug_visualizations/reference/`; epoch folders contain predictions and a
  JSON reference.

The existing faces840 manifest contains the same 36 physical fragments and 360
observations. Manifest validation canonicalizes only the old point-policy alias
and permits the stricter runtime validation phase policy; all selection fields,
sample IDs, mesh hashes, template hash, and sidecar hash remain strictly
checked.
