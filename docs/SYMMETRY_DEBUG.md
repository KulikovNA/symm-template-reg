# Symmetry annotation debug

`tools/debug_symmetry_visualization.py` checks template symmetry bands and the
annotated fragment meshes from `scene_000000`, `scene_000001`, and
`scene_000002`. Its analysis key is `scene_id + fragment_id`: it does not build
the frame-based Dataset index and does not read RGB, depth, masks, or
`visible_points/frame_*.npz`.

## Shared production semantics

- `fragment_annotations.json` declares fragment mesh vertices in local frame
  `F`; every mesh is transformed with its explicit `T_O_from_F` before use;
- half-open region membership comes from
  `models.symmetry.region_assignment.assign_symmetry_regions`;
- mesh-area activation, group intersection, group elements, and training poses
  come from `models.symmetry.targets.build_fragment_symmetry_targets`;
- Dataset, the symmetry-aware loss target-building path, model helper, and the
  debug tool all call that production builder;
- hypothesis placement uses
  `models.symmetry.hypothesis_expander.place_fragment_for_hypothesis`, namely
  `inverse(hypothesis_pose) @ base_pose @ fragment_points_O`;
- visualization reads the annotated face labels and projects only `shell`
  triangle interiors onto the template; `fracture` and `unknown` faces are
  never painted onto the template;
- template faces crossing the shell boundary are locally subdivided and cut;
  every generated boundary vertex remains on the original template triangle;
- template-only region PLYs are also cut exactly by every configured axial
  `y_min_m/y_max_m` plane, so a symmetry-band color never consumes an entire
  triangle that crosses into the neighboring band;
- Cn groups are exact and exhaustive; SO2 training is analytic. The configured
  SO2 gallery size is only a finite visualization of the continuous group.

Mesh regions are active when any enabled robust area criterion passes. The
defaults require at least 1% surface area or 16 of 2048 deterministic
area-weighted diagnostic samples. Vertex counts are reported but cannot by
themselves activate an extra mesh band.

## Commands

Full annotated-fragment run:

```bash
python tools/debug_symmetry_visualization.py \
  --dataset-root /home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test \
  --object-model-id object_000004 \
  --scene-ids scene_000000 scene_000001 scene_000002 \
  --all-annotated-fragments \
  --so2-visualization-samples 12 \
  --gallery-columns 4 \
  --template-projection-distance-m 0.0005 \
  --template-boundary-resolution-m 0.0001 \
  --template-boundary-max-depth 2 \
  --output-root /home/nikita/disser/fragment-template-registration-lab/output_debug
```

Template only:

```bash
python tools/debug_symmetry_visualization.py \
  --dataset-root /home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test \
  --object-model-id object_000004 \
  --mode template \
  --output-root /home/nikita/disser/fragment-template-registration-lab/output_debug
```

Every invocation creates `symmetry_debug_YYYYMMDD_HHMMSS`; timestamp collisions
receive `_001`, `_002`, and so on. Writers refuse to overwrite existing files.

## Output to inspect

Open these first:

1. `template/template_symmetry_regions_with_boundaries.ply`;
2. `scenes/scene_000000/fragments/fragment_0000/fragment_regions_on_template.ply`;
3. `scenes/scene_000000/fragments/fragment_0000/hypothesis_gallery.ply`.

`annotation_audit.json` and `.md` record the actual annotation schema and every
F-to-O mesh join. `fragments_index.csv` is the compact per-fragment index.
`fragment_regions_on_template.ply` contains one template whose covered faces
are colored by symmetry band. `hypothesis_gallery.ply` contains one template
copy per pose; the corresponding shell footprint is colored directly on each
template. No fragment mesh is added on top of the template. The accompanying
`template_projected_vertex_mask.npy` and `template_projected_face_mask.npy`
contain the identity-pose selection, while `hypothesis_index.json` records the
selected template vertex/face counts and boundary-split counts for every pose.
The original dataset PLY is also copied byte-for-byte into the fragment output
directory, for example as `fragment_0000/fragment_0000.ply`; it is a separate
inspection artifact and is not merged into either visualization PLY.

`template/template_symmetry_regions.ply` and
`template/template_symmetry_regions_with_boundaries.ply` use a visualization
mesh split at the exact symmetry-band planes. The original centroid-based face
assignments remain available in `template_face_regions.npy`; the split visual
assignments and their source-face mapping are stored in
`template_visual_face_regions.npy` and
`template_visual_source_face_indices.npy`.
