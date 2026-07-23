# Synthetic fragment-template registration dataset

This document describes the files inspected at the concrete split

`/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test`

on 2026-07-15. It is an observed schema, not a generator-side assumption. Run
`tools/inspect_dataset.py` again for another generated split.

## Inventory

The inspected split contains 10 scenes, 100 frames, one object template and
400 visible `(scene, frame, fragment)` samples. Each of the 100 frame NPZ files
contains four visible fragment IDs. Per-fragment point counts vary from 7 to
8816 (mean 2034.7425): four samples have fewer than the default minimum of 128,
and 35 have more than the default cap of 4096.

The layout is:

```text
test/
├── models/
│   ├── object_000004__scale_0p05.ply
│   └── object_000004__scale_0p05.meta.json
└── scene_XXXXXX/
    ├── camera_info.json
    ├── gt_annotations.json
    ├── scene_meta.json
    ├── coco_annotations.json
    ├── fragments/
    │   ├── fragment_annotations.json
    │   ├── fragment_XXXX.ply
    │   ├── labels/fragment_XXXX_face_labels.npy
    │   └── samples/fragment_XXXX_samples.npz
    ├── images/frame_XXXXXX.png
    ├── depth/frame_XXXXXX.png
    ├── instance_masks/frame_XXXXXX.png
    ├── surface_masks/frame_XXXXXX.png
    ├── visible_points/frame_XXXXXX.npz
    └── scene_gt/
        ├── static_scene_mesh_W.ply
        ├── support_plane_gt.json
        └── frame_XXXXXX/
```

There are 100 RGB PNGs, depth PNGs, instance masks, surface masks and
visible-point NPZs. There are 40 annotated fragment meshes, 40 face-label NPYs
and 40 fragment-sample NPZs. Some scenes contain an ignored small fracture in
their fragment metadata; ignored fragments are deliberately absent from frame
GT, masks and visible points.

The audit opened all 400 primary raster files, all 100 visible NPZs, all 100
per-frame scene-GT depths and all 110 scene-GT meshes. There are no orphan or
uninspected referenced visible NPZs, duplicate frame/fragment IDs, broken path
joins or error-severity findings in this split.

## IDs and joins

`scene_id` is the directory name and the value stored in `gt_annotations.json`.
`frame_id` selects an entry under `gt_annotations.json["frames"]` and the
zero-padded `frame_XXXXXX` files. A frame NPZ holds all visible fragments;
rows are grouped by its integer `fragment_id`. The same ID selects the frame's
GT fragment entry and the scene-level entry in
`fragments/fragment_annotations.json`.

`scene_meta.json["object_model"]` is `../models/object_000004__scale_0p05.ply`.
The loader uses the mesh stem `object_000004__scale_0p05` as the unique
`object_model_id`; the corresponding source object ID in the model meta file is
`object_000004`.

One loader sample ID is therefore:

```text
scene_000000/frame_000000/fragment_0000
```

`gt_annotations.json` has `scene_id` and a `frames` list. Every frame entry
contains `frame_id`, relative paths for image/depth/instance mask/surface mask/
visible NPZ, `T_C_from_W [4,4]`, and a `fragments` list. A GT fragment entry
contains `fragment_id`, `instance_mask_value`, `T_C_from_F`, `T_C_from_O`,
`bbox_visib`, `visible_pixels`, `visible_shell_pixels`, and
`visible_fracture_pixels`.

## Camera, images, depth and masks

Every `camera_info.json` declares width 640, height 480, a 3×3 intrinsic matrix
`K`, scalar `fx/fy/cx/cy`, depth format/scale and coordinate conventions. For
`scene_000000`, for example, `fx=614.8546`, `fy=615.0924`, `cx=327.2523`, and
`cy=238.2554`; consumers should read each scene file rather than hard-code this
example.

The full raster audit observed:

| directory | PIL mode / NumPy dtype | shape | observed values |
| --- | --- | --- | --- |
| `images` | RGB / uint8 | `[480,640,3]` | 0..255 |
| `depth` | I;16 / uint16 | `[480,640]` | 217..4616 mm |
| `instance_masks` | L / uint8 | `[480,640]` | 0 background, 1..4 compact instance IDs |
| `surface_masks` | L / uint8 | `[480,640]` | 0 background/unlabeled, 1 shell, 2 fracture |
| `scene_gt/frame_*/static_scene_depth_C.png` | I;16 / uint16 | `[480,640]` | 217..4616 mm |

For every sample, the number of pixels equal to its `instance_mask_value`
matches `visible_pixels`; shell/fracture mask counts match the corresponding GT
counts. Every NPZ `(u,v)` row points to the expected instance and surface label.
However, `visible_pixels` is 5..125 larger than the NPZ row count in every
sample: those extra instance pixels have surface-mask value 0 and are excluded
from the NPZ. Thus `N = visible_shell_pixels + visible_fracture_pixels`, not
`visible_pixels`.

Depth at NPZ `(u,v)` is a related render product, not an exact replacement for
`points_C[:,2]`: the measured absolute difference has mean 0.707 mm, p95
1.969 mm and maximum 79.021 mm. Code should use the explicitly corresponding
`points_C`/`points_O` rows for registration supervision.

## Coordinate frames, transforms and units

The declared convention is BOP/OpenCV: camera X right, Y down and Z forward.
Transforms use column-vector homogeneous matrices. `T_C_from_O` is a 4×4
object-to-camera transform:

```text
p_C = R_C_from_O @ p_O + t_C_from_O
```

Across every inspected sample, applying the annotated transform to every
corresponding `points_O` row reproduces `points_C` with maximum absolute error
`6.11e-8 m`. Applying `T_C_from_F` to `points_F` has maximum error `6.01e-8 m`.
Rotation determinants range
from `0.99999955` to `1.00000054`, and every homogeneous last row is exactly
`[0, 0, 0, 1]` within the inspected precision.

Point and mesh coordinates are scene units, which are meters in this dataset.
Depth images are uint16 millimeters with `depth_scale_m = 0.001`.

The fragment metadata additionally provides `T_O_from_F`, `T_W_from_F` and
per-frame `T_C_from_F`. Fragment mesh vertices and `points_F` are in the local
fragment frame F. Template vertices and `points_O` are in object frame O.

## `visible_points/frame_XXXXXX.npz`

All 100 files have the same keys and dtypes. Their total frame row count varies
from 2051 to 15476.

| key | dtype | shape | meaning |
| --- | --- | --- | --- |
| `u`, `v` | int32 | `[P]` | source image pixel coordinates |
| `fragment_id` | int32 | `[P]` | scene fragment ID, observed range 0..3 |
| `surface_label` | uint8 | `[P]` | 0 shell, 1 fracture; 255 is reserved but absent in this split |
| `points_C` | float32 | `[P,3]` | visible point in camera frame C |
| `points_F` | float32 | `[P,3]` | same point in fragment frame F |
| `points_O` | float32 | `[P,3]` | same point in intact-object frame O |
| `face_id` | int32 | `[P]` | fragment PLY face element index |
| `barycentric` | float32 | `[P,3]` | position on `face_id` |
| `shell_indices` | int32 | `[Ps]` | indices into the P rows where label is shell |
| `fracture_indices` | int32 | `[Pf]` | indices into the P rows where label is fracture |

Rows of the first nine arrays correspond exactly: index `i` always describes
the same rendered visible point. `shell_indices` and `fracture_indices` are
index lists, not row-aligned arrays. The audit validates equal row lengths,
exact equality of those lists to `flatnonzero(surface_label == 0/1)`, index and
pixel bounds, both F/O transform round-trips and mask-pixel joins for every
file/sample; all checks pass in this split.

The visible NPZs do **not** contain camera-frame normals or a profile. The loader
therefore returns `normals_C=None`; it does not synthesize misleading normals.

## Fragment annotations and auxiliary samples

`fragments/fragment_annotations.json` records object/model IDs, scale history,
fracture generation settings, label encoding, ignored-fragment policy and one
entry per fracture. Each annotated fragment entry includes its mesh, checksums,
`T_O_from_F`, `T_W_from_F`, vertex/face counts, face-label NPY and sample NPZ.

Fragment sample NPZ files contain:

- `points_F`, `normals_F`, `points_O`, and `barycentric`: float32 `[Q,3]`;
- `face_id`: int32 `[Q]`;
- `surface_label`: uint8 `[Q]`.

They also do not contain a profile. The registration dataset currently uses
the rendered visible-point NPZ as the observed input and does not replace it
with these denser fragment-surface samples.

## Static-scene GT

Each `scene_gt/support_plane_gt.json` contains the scene ID, a world-frame
static mesh reference, five `planes_W`, and ten frame entries. A frame entry has
`frame_id`, `T_C_from_W`, `static_scene_mesh_C`, `static_scene_depth_C`, mesh
counts and the same five planes transformed to camera frame C. The audit found
10 world meshes, 100 camera-frame meshes, 100 uint16 camera-frame depth maps,
50 world-plane entries total, and exact GT/support-file frame and transform
joins. The static meshes contain 20 vertices and 10 faces in this split.

## Template mesh and repository

The single template is ASCII PLY with 4953 vertices, 9950 triangular faces and
stored vertex normals. Its O-frame bounds in meters are approximately
`[-0.03925, -0.04945, -0.03925]` to
`[0.03925, 0.04945, 0.03925]`.

`TemplateRepository` supports ASCII and binary PLY directly, so `trimesh`,
`plyfile` and Open3D are not mandatory. If a mesh has faces but lacks normals,
area-weighted vertex normals are computed. Fine/coarse vertex subsets are
configurable and selected geometrically. Loaded tensor dictionaries are cached
by canonical model ID; repeatedly reading samples does not reread the PLY, and
base/scaled aliases share feature-cache entries. Dataset samples clone cached
tensors before optional transforms, so in-place augmentation cannot corrupt the
repository or later samples. A separate feature-cache API is reserved for
future template encoders.

## Loader sample contract

`FragmentTemplateRegistrationDataset` returns:

```text
sample_id, scene_id, frame_id, fragment_id, object_model_id
observed:
  points_C               float32 [N,3]
  normals_C              optional float32 [N,3] (None in this split)
  surface_labels         int64 [N]
  valid_mask             bool [N]
template:
  points_O               float32 [M,3]
  normals_O              optional float32 [M,3]
  faces                  optional int64 [F,3]
  fine_points_O          float32 [Mf,3]
  coarse_points_O        float32 [Mc,3]
gt:
  T_C_from_O             float32 [4,4]
  points_O_corresponding float32 [N,3]
  overlap_labels         bool [N] (shell label == 0)
  active_symmetry_regions / effective_symmetry_group / equivalent_T_C_from_O
                          optional, None without a sidecar
meta:
  coord_unit="m", symmetry_available, policy and source diagnostics
```

The default `min_observed_points=128` filters the four smaller source samples;
it never duplicates points. The default `max_observed_points=4096` is a cap,
not a fixed shape.

Supported observed input policies are:

- `all_points`: preserve every source row, explicitly bypassing the cap;
- `voxel_downsample`: one point per metric voxel, then a geometric cap;
- `random_up_to_max`: seeded random selection without replacement;
- `farthest_point_up_to_max`: deterministic farthest-point sampling;
- `precomputed_dataset_points`: preserve source rows below the cap and use
  farthest-point sampling above it. This is the default and never uses
  `first_1024` or raster-order `linspace` selection.

Every selected row index is applied identically to camera points, object-frame
correspondences and labels.

## Batch contracts

Packed is the primary mode. `observed` and `template` are
`PackedPointBatch` values with `points`, `batch_indices`, prefix-sum `offsets`,
`lengths` and point-aligned `features`. An explicit sample `valid_mask` is
combined with the structural padding mask and therefore reaches encoders.
Observed GT correspondences are packed with the same lengths; transforms are
`[B,4,4]`.

The packed template points are the configurable fine subset. Full-mesh faces
are exposed only together with their full vertices in `template_meshes`.
`template_faces[i]` is `None` when sample `i` was subsampled, preventing full
face indices from being accidentally applied to a shorter point tensor.

Padded mode returns dictionaries with `points_C`/`points_O` shaped
`[B,Nmax,3]`, a boolean `valid_mask [B,Nmax]` and `lengths [B]`. Labels are
padded with 255, while masks distinguish padding from real rows.

Optional mixed-batch supervision is never dropped globally. Correspondences,
overlap labels and region targets have explicit per-point/per-region validity
masks (`points_O_corresponding_valid_mask`, `overlap_labels_valid_mask`, and
`active_symmetry_regions_valid_mask`), plus a per-sample
`symmetry_supervision_mask`. Missing regions are not treated as negative labels.

`PackedPointBatch.to(device)`, `to_padded()`, `from_padded()`, `split()` and
`validate()` preserve arbitrary variable lengths.

The callable collator is registry/config buildable, for example
`dict(type="FragmentTemplateCollator", mode="packed")`; direct
`packed_collate` and `padded_collate` functions remain available for small
scripts and tests.

## Optional symmetry sidecar

No `*.symmetry.json` exists in the inspected `models/` directory. This is not
interpreted as known `C1`: `symmetry_available=false`, all symmetry GT fields
remain `None`, and symmetry supervision must be disabled. When a conventional
`<model-stem>.symmetry.json` or base-object sidecar is added later, the template
repository loads it once and the dataset derives active regions, the effective
group and equivalent GT poses from the selected `points_O`.

## Reproduce this audit

```bash
python tools/inspect_dataset.py \
  --dataset-root "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test" \
  --out-dir work_dirs/dataset_inspection/2026-07-08-test
```

The command writes `dataset_inventory.json`, `dataset_inventory.md`,
`npz_schema.json`, `sample_index.csv`, `template_inventory.json`, and
`warnings.json` without modifying source data.
