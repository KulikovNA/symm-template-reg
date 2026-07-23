# Third-party source audit

Audit date: 2026-07-15. All facts below were collected from the nine local
checkouts under `/home/nikita/disser/fragment-template-registration-lab`.
No checkout was modified, no package was installed, and no native extension was
built. The exact machine-readable snapshot is in `third_party_revisions.json`;
the per-module decisions are in `third_party_modules.json`.

## Scope and method

For each checkout the audit read Git metadata, the root license, dependency
manifests, native source/binary footprints, and the concrete model files named in
the project requirements. Compatibility means compatibility with the active
environment, not with a hypothetical fresh upstream environment. It was assessed
using static import/dependency inspection and small read-only source smokes where
the candidate file could be isolated safely.

The following were deliberately not done:

- no direct production imports from adjacent repositories;
- no `pip`/`conda` installation or version change;
- no invocation of any upstream `setup.py`;
- no CUDA/C++ compilation;
- no wholesale directory copy;
- no source copy from PointDSC, whose local checkout has no repository-level
  license grant.

## Revision and license inventory

| Repository | Local HEAD | Origin | Local license finding | Tracked tree |
|---|---|---|---|---|
| RegTR | `0edee25cda6b1ac1c2b0ac686dcdf2593abf25ba` | `https://github.com/yewzijian/RegTR.git` | MIT, `LICENSE` | clean |
| PointTransformerV3 | `3229e9b7de1770c8ad17c316f8e349982de509f8` | `https://github.com/Pointcept/PointTransformerV3.git` | MIT, `LICENSE` | clean |
| Pointcept | `2b97e6e77a5ad731778f268ccea28ed684367b3f` | `https://github.com/Pointcept/Pointcept` | MIT, `LICENSE` | clean |
| RoITr | `393539d6709c55b2465231cccb7b951f736a5c72` | `https://github.com/haoyu94/RoITr.git` | MIT, `LICENSE` | clean |
| GeoTransformer | `e7a135af4c318ff3b8d7f6c963df094d7e4ea540` | `https://github.com/qinzheng93/GeoTransformer` | MIT, `LICENSE` | clean |
| DFAT | `884149656199c734e2fceff1eda7d7d3b8ebf8c6` | `https://github.com/fukexue/DFAT.git` | MIT, `LICENSE` | clean |
| taxpose | `0c4298fa0486fd09e63bf24d618a579b66ba0f18` | `https://github.com/r-pad/taxpose.git` | MIT, `LICENSE`; `pyproject.toml` incorrectly names `LICENSE.txt` | clean |
| PointDSC | `b009d536ac10b570853833f2178397c154745da9` | `https://github.com/XuyangBai/PointDSC.git` | **NOASSERTION**: no root license and no README grant | clean |
| detr | `29901c51d7fe8712168b8d0d64351170bc0f83e0` | `https://github.com/facebookresearch/detr.git` | Apache-2.0, `LICENSE`; no upstream `NOTICE` file | clean |

Exact upstream license texts for the eight permissively licensed checkouts are
preserved in `third_party_licenses/`. PointDSC has no copied license because none
was present. This is a local-checkout finding; it is intentionally not inferred
from a paper, repository name, or unrelated files with their own notices.

## Dependency and extension findings

| Repository | Declared/observed core dependencies | Custom native code | torch 2.9.1+cu130 decision |
|---|---|---|---|
| RegTR | torch, torchvision, NumPy, SciPy, PyTorch3D, MinkowskiEngine, Open3D | 5 C++ files for KPConv grid/neighbor wrappers; external MinkowskiEngine/PyTorch3D ops | Full runtime blocked. Isolated attention is usable after project-local adaptation. |
| PointTransformerV3 | torch, addict, spconv, torch-scatter, timm, optional flash-attn | none in this detached checkout; binary dependencies are mandatory | Optional encoder blocked; serialization can be reconsidered separately. |
| Pointcept | environment targets torch 2.5.0/cu124; PyG family, spconv, flash-attn, Open3D | 25 C++, 21 CUDA, 22 header files in five `libs/` extension families | Full stack blocked and must not be compiled silently. |
| RoITr | pinned torch 1.8, SciPy, einops, Open3D | pointops: 8 C++, 6 CUDA, legacy CPython 3.7/3.8 `.so` files | Full runtime blocked; pure PPF/embedding path is adaptable. |
| GeoTransformer | torch, NumPy, SciPy, einops, Open3D 0.11.2, scikit-learn | grid/radius C++ sources declared through `CUDAExtension` | Full runtime blocked; pure matching/embedding/Sinkhorn requires device fixes. |
| DFAT | GeoTransformer stack plus focus-attention code | Geo extension, CUDA chamfer, legacy CPython 3.8 `.so` files | Deferred; interface only for iteration one. |
| taxpose | declares Python `<3.10`; torch 2.0.1/cu118, PyTorch3D, PyG family, DGL, SciPy | none in repo, several compiled dependencies | Full runtime incompatible; small point-weight idea can be cleanly reimplemented. |
| PointDSC | Python 3.7, torch 1.6/cu101, SciPy, Open3D 0.9 | prebuilt CPU `utils/libpmc.so`, no matching source in checkout | Reference only because of license; do not copy or load the binary. |
| detr | torch, torchvision, SciPy, COCO APIs, ONNX | none in repo | Decoder is compatible; upstream matcher import is blocked by missing SciPy. |

Native counts include checked-in build artifacts where present because those
artifacts are themselves compatibility evidence. `tools/inspect_environment.py`
emits the full path list and suffix counts.

## Repository-by-repository source decisions

### RegTR

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/RegTR`.
- Candidate sources: `src/models/transformer/transformers.py` and the
  correspondence/overlap portions of `src/models/regtr.py`.
- Intended purpose: bidirectional self/cross interaction, overlap logits, and
  template-coordinate correspondence prediction.
- Excluded sources: `src/models/backbone_kpconv/` and its C++ wrappers. The
  upstream package auto-discovery imports `MinkowskiEngine` before a model can be
  used.
- Evidence: package import failed on missing MinkowskiEngine, while isolated
  `transformers.py` produced finite torch 2.9 CPU outputs of shapes
  `(2,5,2,16)` and `(2,7,2,16)`.

### PointTransformerV3

- Absolute path:
  `/home/nikita/disser/fragment-template-registration-lab/PointTransformerV3`.
- Candidate sources: `model.py` and `serialization/{default,z_order,hilbert}.py`.
- Intended purpose: optional serialized packed-point encoder.
- Current decision: reject the encoder port for iteration one and register an
  explicit optional/unavailable implementation. `model.py` imports `addict`,
  `spconv`, `torch_scatter`, and `timm` unconditionally. The optional Pointcept
  submodule is not initialized in this checkout.
- Evidence: direct model import stopped at missing `addict`; no dependency was
  installed to advance the import chain.

### Pointcept

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/Pointcept`.
- Candidate references: `pointcept/models/utils/structure.py`,
  `pointcept/models/utils/serialization/`, and
  `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`.
- Intended purpose: cross-check packed offsets, serialization, pooling, and PTv3
  interfaces.
- Current decision: reference only. The current project's `PackedPointBatch` is
  independent rather than a Pointcept source port.
- Blocker: the recorded environment targets torch 2.5/cu124 and the relevant
  imports require `torch_scatter`, `spconv`, and project CUDA libraries.

### RoITr

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/RoITr`.
- Candidate sources: `lib/utils.py` and `dataset/common.py` for the PPF formula;
  `model/transformer/positional_encoding.py`,
  `model/transformer/ppftransformer.py`, and
  `model/transformer/attention.py` for optional PPF-conditioned attention.
- Intended purpose: pure-torch point-pair features and a rotation-invariant
  attention interface.
- Excluded source: `cpp_wrappers/pointops/`. Its checked-in binaries target
  CPython 3.7/3.8, not the active CPython 3.10.
- Evidence: upstream PPF and geometric embedding classes ran on torch 2.9 CPU
  with finite shapes `(2,8,5,16)` and `(2,8,8,16)`.

### GeoTransformer

- Absolute path:
  `/home/nikita/disser/fragment-template-registration-lab/GeoTransformer`.
- Candidate sources:
  `geotransformer/modules/geotransformer/geotransformer.py`,
  `superpoint_matching.py`, `modules/ops/pairwise_distance.py`, and
  `modules/sinkhorn/learnable_sinkhorn.py`.
- Intended purpose: geometric structure embedding, coarse dual-normalized
  matching, and log-domain optimal transport.
- Required adaptation: remove hard-coded `.cuda()` allocations, add explicit
  masks, avoid extension-package imports, and keep allocations on the input
  device/dtype.
- Evidence: upstream Sinkhorn failed on CPU with `No CUDA GPUs are available`;
  this is a source portability defect, not a reason to alter the environment.

### DFAT

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/DFAT`.
- Candidate sources:
  `geotransformer/modules/transformer/spotguided_transformer.py`,
  `modules/lineartransformer/`, and `experiments/3DMatch/model.py`.
- Intended purpose: second-queue focus-attention experiments only.
- Current decision: keep a clean-room interface outside the baseline. Do not
  copy fine-scale integration or native/chamfer extensions.
- Evidence: focus-attention import currently stops at missing `einops`; the
  repository also contains a CPython 3.8 extension binary.

### taxpose

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/taxpose`.
- Candidate sources: the weight projections and soft-correspondence sections of
  `taxpose/nets/transformer_flow_pm.py`, `transformer_flow.py`, and
  `taxpose/models/taxpose.py`.
- Intended purpose: a small per-point confidence head, not TAX-Pose's full
  dual-flow solver.
- Current decision: clean-room MLP interface only. The package itself declares
  Python `<3.10` and pins GPU dependencies around torch 2.0/CUDA 11.8.

### PointDSC

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/PointDSC`.
- Examined sources: `models/PointDSC.py`, `models/common.py`,
  `utils/max_clique.py`, and `utils/libpmc.so`.
- Architectural purpose: second-queue pairwise spatial-consistency scoring.
- Legal decision: **reject every direct source or binary port**. No root license
  file and no grant in README were found. A small project-local expression of
  Euclidean distance preservation may exist only as a clean-room mathematical
  interface, not as copied PointDSC source.
- Evidence: `NonLocalBlock` itself runs on current torch, but technical
  executability does not cure the missing license grant.

### DETR

- Absolute path: `/home/nikita/disser/fragment-template-registration-lab/detr`.
- Candidate sources: `models/detr.py`, `models/transformer.py`, and
  `models/matcher.py`.
- Intended purpose: learnable pose queries, intermediate decoder outputs,
  no-pose classification, auxiliary losses, and small-K set assignment.
- Required adaptation: replace boxes/classes with direct 6D-rotation and
  translation outputs; replace SciPy assignment with a project-local exact
  small-K implementation or an explicitly optional SciPy path.
- Evidence: isolated upstream decoder produced finite output `(2,3,2,16)` on
  torch 2.9; package import failed because SciPy is absent.

## Verification and provenance contract

Re-run the read-only inventory with:

```bash
python tools/inspect_environment.py \
  --third-party-root "/home/nikita/disser/fragment-template-registration-lab"
```

Any derivative file must retain the applicable copyright/license and identify:
source repository, full source commit, original relative path, license, and a
brief change summary. Clean-room files must say that no upstream source was
copied. The canonical notice text and source-to-target mapping live in
`THIRD_PARTY_NOTICES.md` and `third_party_modules.json`.
