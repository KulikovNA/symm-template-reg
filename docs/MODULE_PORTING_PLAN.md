# Module porting plan

This plan converts the audit into bounded, reviewable modules. It does not
authorize importing adjacent repositories at runtime or copying complete
directories. `third_party_modules.json` is the canonical machine-readable
mapping; this document explains sequencing and acceptance gates.

## Porting modes

Every candidate uses exactly one mode:

- **adapted derivative**: a small permissively licensed source unit is modified;
  its copyright/license and a complete provenance header are retained;
- **clean-room implementation**: only the public architecture/contract is used,
  with no copied source expression; the file says this explicitly and the
  notice records the inspiration;
- **reference-only**: no source is copied or adapted;
- **rejected**: incompatible, unnecessary, or not licensed for copying.

Current iteration-one model files use clean-room project interfaces. The status
`ported` in the table means that such a local interface exists. It does not mean
that an upstream directory was vendored.

## Current mapping

| Priority | Capability | Source file(s) | Project target | Mode | Status |
|---|---|---|---|---|---|
| 1 | bidirectional self/cross interaction | RegTR `src/models/transformer/transformers.py`, `src/models/regtr.py` | `models/attention/regtr/interaction.py` | clean-room | ported |
| 1 | overlap/correspondence heads | RegTR `src/models/regtr.py` | `models/heads/{overlap,correspondence}_head.py` | clean-room | ported |
| 1 | geometric structure embedding | GeoTransformer `modules/geotransformer/geotransformer.py`, pairwise/position helpers | `models/geometry/geometric_embedding.py` | clean-room reduced form | ported |
| 1 | coarse matching | GeoTransformer `superpoint_matching.py` | `models/matching/coarse_matching.py` | clean-room | ported |
| 1 | Sinkhorn/dustbin transport | GeoTransformer `modules/sinkhorn/learnable_sinkhorn.py` | `models/matching/optimal_transport.py` | device-aware adaptation | ported |
| 1 | point-pair features | RoITr `lib/utils.py`, `dataset/common.py`, positional embedding | `models/geometry/ppf.py` | clean-room | ported |
| 1 optional | PPF-conditioned attention | RoITr transformer/model files | `models/attention/rotation_invariant_attention.py` | clean-room interface | ported |
| 1 | pose query decoder and auxiliary outputs | DETR `models/detr.py`, `models/transformer.py` | `models/heads/pose_query_head.py` | clean-room domain adaptation | ported |
| 1 | small-K set assignment | DETR `models/matcher.py`, `models/detr.py` | `models/matching/hungarian_assigner.py` | clean-room, SciPy-free | ported |
| 1 | point importance logits | TAX-Pose flow/head files | `models/heads/point_weight_head.py` | clean-room | ported |
| 1 optional | PTv3 encoder | PointTransformerV3 `model.py`, `serialization/` | `models/backbones/ptv3/__init__.py` | unavailable placeholder only | rejected |
| 1 optional | Pointcept packed/serialized encoder | Pointcept structure/serialization/PTv3 files | none | reference-only | rejected |
| 2 | focus-attention interface | DFAT spot-guided/linear transformer files | `models/attention/focus_attention.py` | clean-room interface | ported |
| 2 | spatial consistency | PointDSC `models/PointDSC.py`, `models/common.py` | `models/matching/spatial_consistency.py` | formula-level clean-room only | source port rejected |
| never in iteration 1 | KPConv C++/Minkowski path | RegTR backbone/wrappers | none | rejected | rejected |
| never in iteration 1 | Geo grid/radius extension | GeoTransformer extensions/setup | none | rejected | rejected |
| never in iteration 1 | pointops CUDA | RoITr `cpp_wrappers/pointops` | none | rejected | rejected |
| never in iteration 1 | DFAT extensions/chamfer | DFAT extensions/setup | none | rejected | rejected |
| never | prebuilt PMC clique binary | PointDSC `utils/libpmc.so` | none | rejected | rejected |

Paths in the project-target column are relative to `symm_template_reg/`. Full
commits, licenses, purpose, and changes for each row are recorded in
`third_party_modules.json`.

## Phase 1: guaranteed baseline

The baseline must run with the existing torch/NumPy environment and no optional
third-party packages.

### Interaction and heads

Use RegTR only for the architectural organization: self-attention inside each
cloud, bidirectional cross-attention, overlap supervision, and coordinate
correspondence diagnostics. The local contract differs intentionally:

- batch-first padded tensors plus explicit valid masks;
- no KPConv or upstream preprocessor;
- no upstream package registry/import discovery;
- no mandatory weighted-SVD pose solver;
- heads registered independently through the project registry.

Acceptance: two samples with different point counts must pass interaction,
masks must zero padded outputs, and gradients must remain finite.

### Geometric embedding and matching

Use a bounded local-distance geometric embedding for the smoke baseline. The
full GeoTransformer triplet tensor may be reconsidered after profiling, but must
not make the initial forward quadratic-to-cubic in an uncontrolled `N`.

Coarse matching uses masked normalized feature similarity. Optimal transport
uses log-domain normalization and an optional learned dustbin. Every allocation
must derive its device/dtype from inputs; upstream `.cuda()` calls are not
carried over.

Acceptance: invalid rows/columns are masked, CPU forward/backward is finite, and
the module imports without GeoTransformer or its extension.

### Rotation-invariant geometry

Implement the PPF tuple directly in torch:

`[||d||, angle(n_i,d), angle(n_j,d), angle(n_i,n_j)]`.

Normals and neighbor associations are inputs to the module; Open3D normal
estimation and RoITr pointops are out of scope. PPF-conditioned attention stays
optional because many dataset samples may not provide normals.

Acceptance: finite output for parallel/antiparallel vectors, correct final
dimension four, and no Open3D/einops/pointops import.

### Pose queries and set assignment

Retain DETR's useful contracts rather than its detection domain:

- K learned query slots;
- decoder cross-attention to point-token memory;
- a pose/no-pose logit for every query;
- intermediate decoder outputs for auxiliary losses;
- one-to-one assignment to the valid GT symmetry set.

The output is direct 6D-rotation plus translation, not a box. Because `K=8` is
small and SciPy is absent, use the exact project-local small-K assignment rather
than adding SciPy as a mandatory dependency.

Acceptance: output `[B,K,4,4]`, finite gradients, `det(R)≈1`, extra queries map
to no-pose targets, and auxiliary output count matches decoder depth minus one.

### Point importance

Use the TAX-Pose idea of a learned per-point importance value, but not its full
flow network, PyTorch3D transform stack, or dual-flow pose solver. The local head
returns masked logits and is supervised/aggregated by project code.

Acceptance: shape follows the observed point batch and padded values cannot
contribute to a loss.

## Phase 2: optional research modules

### PTv3

Do not enable PTv3 merely because a registry name exists. To move the status
from rejected to planned, all of the following must be demonstrated without
changing the core environment:

1. a pure-torch or explicitly compatible replacement for mandatory `spconv`
   and `torch_scatter` paths;
2. a torch 2.9/CUDA 13 support statement or a CPU-only implementation;
3. successful packed variable-N forward/backward in a disposable test;
4. no build/install side effect during ordinary project installation.

Serialization functions may be adapted independently if they remain pure torch
and have dedicated round-trip tests. Pointcept is a cross-reference, not a
runtime dependency.

### DFAT focus attention

The current interface may be exercised only in a dedicated config after the
baseline is stable. A later implementation should preserve masks and variable
lengths and must not assume GeoTransformer neighbor metadata or compile the
DFAT extension.

### Spatial consistency

PointDSC source cannot be ported under the locally observed license state. A
future module may implement the published distance-preservation formula and an
independently designed interface. It must not reproduce source structure,
comments, naming, or the PMC binary path. Legal status must remain
`NOASSERTION` unless an explicit license grant is added and audited.

## Provenance header contract

An **adapted derivative** must begin with a header equivalent to:

```python
"""Adapted third-party module.

Source repository: https://github.com/OWNER/REPOSITORY
Source commit: FULL_40_CHARACTER_COMMIT
Original path: relative/path/in/upstream.py
License: SPDX-ID (see third_party_licenses/FILENAME)
Changes: concise list of project-specific changes.
"""
```

A **clean-room implementation** must instead state:

```python
"""Clean-room implementation inspired by PUBLIC_ARCHITECTURE.

Reference repository: URL at FULL_40_CHARACTER_COMMIT
Reference path(s): ...
No upstream source code is imported or copied.
"""
```

If multiple repositories informed a file, list each one. Do not claim an MIT
license for PointDSC; use `NOASSERTION` and state that only a mathematical idea
was referenced.

## Review gates for changing status

`planned -> ported` requires:

- exact upstream path/commit/license in `third_party_modules.json`;
- a source or clean-room header in every affected project file;
- a small diff, not a copied directory;
- no adjacent-repository production import;
- no unapproved dependency or extension.

`ported -> tested` additionally requires:

- focused unit coverage for shapes, masks, device behavior and gradients;
- CPU smoke in `fracs`;
- CUDA smoke only when CUDA is actually visible;
- a provenance/forbidden-import scan;
- update of `THIRD_PARTY_NOTICES.md` if distributed content changed.

Rejected entries remain rejected until the blocker itself changes. A missing
dependency alone is not authorization to install it, and technical
compatibility alone is not a license grant.

## Required final checks

Before release or a status promotion:

```bash
python -m compileall symm_template_reg tools tests configs
python -m unittest discover -s tests -v
python tools/inspect_environment.py \
  --third-party-root "/home/nikita/disser/fragment-template-registration-lab"
```

Then scan production source for forbidden imports from `RegTR`,
`PointTransformerV3`, `Pointcept`, `RoITr`, `GeoTransformer`, `DFAT`, `taxpose`,
`PointDSC`, `detr`, `frag_template_reg`, and `frag_geometry_engine`. Comments,
documentation, provenance JSON, and notices may name them; executable imports
may not.
