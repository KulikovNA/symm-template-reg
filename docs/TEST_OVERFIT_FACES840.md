# faces840 controlled GPU overfit

> `debug_training_on_test_split = true`  
> `train_and_validation_use_same_samples = true`  
> `results_are_not_final_evaluation = true`

This is a controlled optimization/debug experiment, not a generalization
evaluation. Train and validation both use the same content-addressed manifest:

`/home/nikita/disser/fragment-template-registration-lab/work_dirs/manifests/test_faces840_all_9e91dfb58d07.json`

It contains 36 accepted physical fragments and all 360 observations of those
fragments. Four fragments with fewer than 840 polygon faces and their 40
observations are excluded before Dataset indexing. Train shuffles; validation
does not. Neither path uses augmentation.

The editable config is `configs/debug/test_overfit_faces840_gpu.py`. Evaluation
runs before training and every two epochs. A deterministic eight-sample debug
set is fixed once per run, and predicted PLY files are written before training
and every five epochs. Predicted effective groups come only from
`active_region_logits`; predicted hypothesis expansion never consumes the GT
effective group.

Each selected sample directory contains these geometry artifacts:

- `fragment_XXXX.ply`: byte-for-byte copy of the physical source fragment;
- `pred_top1_camera_frame.ply`: observed points and the top-1 template pose in
  the camera frame;
- `predicted_fragment_regions_on_template.ply`: the top-1 predicted fragment
  footprint colored directly on adaptively split template faces; this is a
  strict sub-millimeter surface-overlap diagnostic and can collapse to a thin
  intersection when the predicted pose is poor;
- `gt_fragment_regions_on_template.ply`: filled GT-aligned reference produced
  by the same triangle projection and boundary-splitting code;
- `predicted_registered_fragment_on_template.ply`: complete physical fragment
  transformed into the object frame implied by the top-1 model pose, overlaid
  with the template so a bad registration remains visible instead of vanishing;
- `predicted_hypotheses_gallery.ply`: the same face-colored footprint for every
  symmetry-expanded top-1 hypothesis;
- `optional_pred_topK_base_gallery.ply`: independent base pose queries;
- `optional_pred_vs_gt_camera_frame.ply`: predicted versus GT template pose.

The footprint projection uses only physical-fragment shell faces. Dataset
`T_C_from_F` places that source mesh in the camera frame for visualization; the
template registration itself uses the model-predicted `T_C_from_O`. The local
template splitting parameters live under `debug_visualization` in the config.

At startup the trainer prints the model, total/trainable/frozen parameter
counts, per-module parameter counts, device/AMP state and Dataset sizes. Every
training epoch has a progress bar with the running total, symmetry, rotation,
translation and query-classification losses. Validation has a separate bar and
prints the complete `eval/*` metric table after it finishes. These controls live
under `terminal_output` in the config.

Only these checkpoint files are permitted:

```text
checkpoints/best.pth
checkpoints/best_metrics.json
checkpoints/best_manifest.json
```

`best.pth` is atomically replaced only when `eval/symmetry_pose_loss` improves
by at least `best_metric_min_delta`. No periodic, latest, or final checkpoint is
created.

Full launch:

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg
python tools/train.py \
  --config configs/debug/test_overfit_faces840_gpu.py \
  --device cuda
```

This starts 100 epochs, so it must be run explicitly by the user. The Codex
implementation run stopped after the required two- and five-epoch smokes.
