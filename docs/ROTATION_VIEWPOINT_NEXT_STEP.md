# Rotation/viewpoint architecture decision note

Current evidence is diagnostic, not a final architecture benchmark:

- direct optimization of the current PoseCodec, 6D rotation and C2-aware pose
  cost passes 16/16 starts on frames 4, 8 and 6;
- the 10-view K1 run reaches 0/10 at 5 degrees / 5 mm;
- the K8 oracle reaches 3/10 and uses stable view-specific query assignments;
- swapping observed clouds changes decoded query rotations by less than
  0.00002 degree and normalized translations by less than 0.000002.

The current diagnosis is therefore
`pose_head_or_context_conditioning_problem_with_multi_query_view_specialization`.
The learned queries behave as a nearly static pose codebook. One-frame and
view-ladder experiments must still establish exactly where scaling first fails.

## Option 1: correspondence-guided pose

Use the existing observed/template token streams and correspondence head:

1. supervise observed-to-template correspondences with the existing
   symmetry-aware correspondence targets;
2. predict correspondence confidence/overlap;
3. optionally feed weighted correspondences to a differentiable weighted rigid
   alignment step;
4. retain the direct pose head as an auxiliary proposal or residual path.

This gives the pose estimator an explicit rotation-bearing relationship between
camera-frame observed coordinates and object-frame template coordinates.

## Option 2: geometric matching before direct pose

Reuse the existing PPF/geometric modules and GeoTransformer-style
distance/angle embeddings for matching. Rotation-invariant descriptors are
appropriate for finding correspondences, but they cannot by themselves encode
absolute camera-frame orientation. Keep an equivariant coordinate stream, or
recover the transform from matched camera/object coordinates.

## Decision gate

- If K1 fails on one frame, inspect pose-decoder attention, pose-projection
  gradients and output saturation before adding either branch.
- If K1 passes one frame but fails as views are added, prioritize explicit
  correspondence supervision and rotation-aware/equivariant geometry.
- If K1 multi-view passes but K8 fails, fix multi-query assignment/diversity.
- Start ranking only after K8 oracle success passes the pose gate.

No branch in this document is enabled automatically and neither replaces the
current direct pose head without a separate ablation.
