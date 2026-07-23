# Iteration-one implementation report

Date: 2026-07-15  
Repository: `/home/nikita/disser/fragment-template-registration-lab/symm-template-reg`

## Outcome

The first modular baseline is complete. The editable package installs without
runtime dependencies on adjacent repositories, reads the supplied split, builds
packed variable-size batches, reuses a cached template, builds every baseline
component from config/registry, returns eight direct pose hypotheses and dense
diagnostics, and completes a finite real-batch CPU optimizer step.

No long training was started. No PyTorch, CUDA, torchvision, or NumPy version was
changed. The only installed package was this editable project with `--no-deps`.

## Repository tree

```text
.
├── pyproject.toml
├── README.md
├── configs/
│   ├── _base_/runtime.py
│   └── symm_template_reg_baseline.py
├── symm_template_reg/
│   ├── registry.py
│   ├── config.py
│   ├── datasets/
│   │   ├── fragment_template_dataset.py
│   │   ├── template_repository.py
│   │   ├── collate.py
│   │   ├── structures.py
│   │   ├── transforms.py
│   │   └── samplers.py
│   └── models/
│       ├── builder.py
│       ├── backbones/{simple_point_encoder.py,ptv3/}
│       ├── attention/{regtr/,geometric_attention.py,rotation_invariant_attention.py,focus_attention.py}
│       ├── geometry/{point_ops.py,ppf.py,geometric_embedding.py}
│       ├── matching/{coarse_matching.py,fine_matching.py,optimal_transport.py,spatial_consistency.py,hungarian_assigner.py}
│       ├── heads/{overlap,correspondence,point_weight,symmetry_region,pose_query,uncertainty}_head.py
│       ├── losses/{correspondence,overlap,region,pose_set,symmetry_pose,consistency}_loss.py
│       ├── pose/{rotation.py,pose_representation.py,pose_hypotheses.py,metrics.py}
│       ├── symmetry/{metadata.py,groups.py,region_assignment.py,hypothesis_expander.py}
│       ├── structures/{model_outputs.py,point_batch.py}
│       └── detectors/symm_template_reg.py
├── tools/
│   ├── inspect_environment.py
│   ├── inspect_dataset.py
│   ├── inspect_dataset_sample.py
│   ├── show_sample.py
│   ├── smoke_dataset.py
│   ├── smoke_model.py
│   ├── smoke_train_step.py
│   ├── profile_model_memory.py
│   └── check_forbidden_imports.py
├── tests/                         # 70 unittest cases
├── docs/
│   ├── DATASET_FORMAT.md
│   ├── THIRD_PARTY_AUDIT.md
│   ├── ENVIRONMENT_COMPATIBILITY.md
│   ├── MODULE_PORTING_PLAN.md
│   └── ITERATION1_REPORT.md
├── schemas/template_symmetry.schema.json
├── examples/object_000004__scale_0p05.symmetry.example.json
├── THIRD_PARTY_NOTICES.md
├── third_party_revisions.json
├── third_party_modules.json
├── third_party_licenses/          # eight exact permissive license texts
├── environment_fracs.txt
└── work_dirs/
    ├── dataset_inspection/2026-07-08-test/
    └── dataset_sample/
```

Generated `__pycache__`, editable-install metadata, and work products are omitted
from the source tree above.

## Third-party decisions

No upstream production package is imported and no complete source directory was
copied. Influenced modules are independently written clean-room implementations;
their headers and `third_party_modules.json` record exact repository, 40-character
commit, original path, license, and changes.

| Reference | Project-local result | Baseline state |
|---|---|---|
| RegTR | bidirectional self/cross interaction; overlap and correspondence heads | enabled, tested |
| GeoTransformer | bounded geometric embedding; coarse matching; device-safe log Sinkhorn | embedding/matcher enabled; Sinkhorn tested but optional |
| RoITr | PPF and PPF-biased attention contract | tested, optional |
| DETR | learnable pose queries, no-pose logits, auxiliary outputs, polynomial rectangular set assignment | enabled, tested |
| TAX-Pose | per-point confidence/importance head | enabled, tested |
| PointTransformerV3 / Pointcept | explicit unavailable PTv3 registry placeholder and compatibility audit | not enabled; compiled deps absent |
| DFAT | saliency/focus interface only | second queue, not enabled |
| PointDSC | independent distance-preservation formula only | source port rejected (`NOASSERTION`), not enabled |

The guaranteed `SimplePointEncoder` is an independent pure-PyTorch temporary
baseline: point MLP, local kNN aggregation, global token, packed/padded input, no
fixed point count, and no native extension.

## Observed dataset schema

The concrete split has 10 scenes, 100 frames, 400 raw fragment samples, one
template, and per-fragment `N=7..8816`. Four samples are below the default minimum
128; the loader exposes 396 usable samples. Thirty-five raw samples exceed 4096
and are geometrically capped by the baseline policy.

Every frame NPZ contains row-aligned `u`, `v`, `fragment_id`, `surface_label`,
`points_C`, `points_F`, `points_O`, `face_id`, and `barycentric`; shell/fracture
index arrays are separate index lists. The full audit inspected 100/100 NPZ files,
all referenced RGB/depth/masks, fragment assets, and `scene_gt` joins. It found
zero row/index/mask/join failures and zero error-severity findings.

Transform evidence:

- `points_O -> T_C_from_O -> points_C`: maximum absolute error `6.110e-08 m`;
- `points_F -> T_C_from_F -> points_C`: maximum absolute error `6.008e-08 m`;
- units are meters; depth PNG units are millimetres (`depth_scale_m=0.001`).

The complete factual schema is in `docs/DATASET_FORMAT.md`; machine-readable
artifacts are under `work_dirs/dataset_inspection/2026-07-08-test/`.

## Environment compatibility

| Item | Verified value |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.9.1+cu130 |
| torchvision | 0.24.1+cu130 |
| NumPy | 1.24.4 |
| CUDA build in torch | 13.0 |
| CUDA runtime visibility | unavailable (`torch.cuda.is_available() == False`) |
| nvcc / ninja | unavailable / unavailable |
| SciPy, Open3D, trimesh, plyfile | absent and not required |

Full upstream PTv3/Pointcept/RoITr/GeoTransformer/DFAT/TAX-Pose stacks are blocked
by missing or incompatible compiled dependencies. None was installed or built.

## Runtime output shapes

The real CPU batch contained observed lengths 1423 and 2488; the cached fine
template length was 2048. With `B=2`, `K=8`, `U=6`, `R=16`:

| Output | Verified shape |
|---|---|
| `pose_hypotheses` | `[2,8,4,4]` |
| `pose_logits` | `[2,8]` |
| `pose_uncertainty` | `[2,8,6]` |
| `observed_overlap_logits` | `[2,2488]` |
| `template_visibility_logits` | `[2,2048]` |
| `correspondence_points_O` | `[2,2488,3]` |
| `correspondence_confidence` | `[2,2488]` |
| `observed_region_logits` | `[2,2488,16]` |
| `active_region_logits` | `[2,16]` |
| `insufficient_information_logit` | `[2,1]` |
| observed/template masks | `[2,2488]` / `[2,2048]` |
| each of two auxiliary decoder outputs | poses `[2,8,4,4]`, logits `[2,8]`, uncertainty `[2,8,6]` |

All floating outputs were finite. Dense outputs are padded and always accompanied
by masks.

## Verification record

| Check | Result |
|---|---|
| `python -m pip install -e . --no-deps` | success |
| `python -m compileall symm_template_reg tools tests configs` | success |
| environment audit | success; all nine adjacent tracked worktrees clean |
| dataset audit | 400 samples; zero invariant/error findings |
| dataset smoke | 8 lengths `[1423,2488,2008,1426,2072,1620,3508,4096]`; cache load count 1 |
| unittest | 70/70 passed, including numerical/batching and symmetry-visualization regressions |
| forbidden production import scan | zero violations |
| CPU model forward | passed, finite |
| CPU backward + AdamW step | passed; total loss `18.185543`; all gradients finite |
| CPU parameter/memory profile | 10,233,911 trainable parameters |
| CUDA forward/backward | skipped because CUDA is not visible in this process |

The regression suite explicitly covers `K=8, G=36` assignment, `N=1`, padding
invariance when valid counts are below kNN `k`, finite gradients at zero
axis-angle and parallel/antiparallel PPF endpoints, boolean metadata/device
movement, optional region heads, and symmetry metrics/losses on native
`[B,K,4,4]` predictions.

## Symmetry state

The original iteration-one smoke was completed before a real sidecar was
available. The sidecar is now present and has been validated against the full
template mesh by the follow-up symmetry debug workflow documented in
`docs/SYMMETRY_DEBUG.md`. The parser/schema/tests cover arbitrary `Cn`, `SO2`,
exact Cn expansion, `Cn intersect Cm = Cgcd(n,m)`, `SO2 intersect Cn = Cn`,
non-zero axis origins, and an analytic SO2 loss/metric that ignores twist.

## Unresolved blockers and temporary choices

1. The sidecar and numerical/visual artifacts are now available; a human visual
   review of the three primary PLY files remains required before long training.
2. CUDA execution is unverified because the current process cannot access a GPU;
   this is an environment/driver visibility blocker, not a CPU-path failure.
3. PTv3 remains an optional unavailable registry entry; adopting it requires an
   explicit torch 2.9/CUDA 13 compatibility task, not dependency churn here.
4. DFAT focus and PointDSC consistency are intentionally excluded from baseline.
5. The model is untrained; smoke losses prove differentiability, not registration
   quality or convergence.
6. `SimplePointEncoder`, bounded FPS interaction tokens, and max region capacity
   are iteration-one engineering baselines to be replaced/profiled experimentally.

## Exact next step

Open the generated real-data symmetry artifacts before any training:

```bash
python tools/debug_symmetry_visualization.py \
  --dataset-root "/home/nikita/data_generator/generation_dataset/generation_synthetic/output/fragment_template_registration/differBig/2026-07-08/test" \
  --object-model-id object_000004 \
  --mode all \
  --scene-id scene_000000 \
  --frame-id 0 \
  --all-fragments \
  --include-fragment-mesh \
  --output-root output_debug
```

Review `template_symmetry_regions_with_boundaries.ply`, then each fragment's
`active_regions_on_template.ply` and `gt_hypotheses_gallery.ply` before any
longer training run is authorized.
