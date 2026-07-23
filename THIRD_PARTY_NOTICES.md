# Third-party notices

This repository contains project-local, clean-room implementations and small
adaptations informed by the reference projects listed below. Adjacent checkouts
are not runtime dependencies and are not redistributed wholesale.

The source revisions are fixed in `third_party_revisions.json`; detailed
source-to-target decisions are fixed in `third_party_modules.json`. Full license
texts are in `third_party_licenses/`.

## RegTR

- Source: `https://github.com/yewzijian/RegTR.git`
- Audited revision: `0edee25cda6b1ac1c2b0ac686dcdf2593abf25ba`
- License: MIT
- Copyright: Copyright (c) 2022 Zi Jian Yew
- License text: `third_party_licenses/RegTR-MIT.txt`
- Project use: clean-room bidirectional self/cross-attention organization and
  overlap/correspondence output contracts.
- Changes: removed KPConv, MinkowskiEngine, PyTorch3D, upstream registry/package
  imports, sequence-first layout, and mandatory numerical pose solving; added
  project registry, batch-first masks, and separate heads.

## PointTransformerV3

- Source: `https://github.com/Pointcept/PointTransformerV3.git`
- Audited revision: `3229e9b7de1770c8ad17c316f8e349982de509f8`
- License: MIT
- Copyright: Copyright (c) 2023 Pointcept
- License text: `third_party_licenses/PointTransformerV3-MIT.txt`
- Project use: compatibility reference for a future serialized packed-point
  encoder. No upstream encoder source is included in iteration one.
- Changes: the project registers an explicit optional/unavailable placeholder;
  it does not import spconv, torch-scatter, timm, or the upstream package.

## Pointcept

- Source: `https://github.com/Pointcept/Pointcept`
- Audited revision: `2b97e6e77a5ad731778f268ccea28ed684367b3f`
- License: MIT
- Copyright: Copyright (c) 2023 Pointcept
- License text: `third_party_licenses/Pointcept-MIT.txt`
- Project use: reference for packed offsets, point structures, serialization,
  and PTv3 integration. The project's `PackedPointBatch` is independently
  designed; no Pointcept native library or model source is included.

## RoITr

- Source: `https://github.com/haoyu94/RoITr.git`
- Audited revision: `393539d6709c55b2465231cccb7b951f736a5c72`
- License: MIT
- Copyright: Copyright (c) 2023 Hao Yu
- License text: `third_party_licenses/RoITr-MIT.txt`
- Project use: clean-room point-pair feature formula and optional
  PPF-conditioned attention interface.
- Changes: removed Open3D normal estimation, NumPy runtime calculations,
  pointops CUDA, legacy binaries, and upstream model coupling; added pure torch
  tensor inputs, numerical clamps, registry integration, and explicit masks.

## GeoTransformer

- Source: `https://github.com/qinzheng93/GeoTransformer`
- Audited revision: `e7a135af4c318ff3b8d7f6c963df094d7e4ea540`
- License: MIT
- Copyright: Copyright (c) 2022 Zheng Qin
- License text: `third_party_licenses/GeoTransformer-MIT.txt`
- Project use: geometric structure embedding, dual-normalized coarse matching,
  and log-domain optimal transport contracts.
- Changes: removed the extension package and hard-coded CUDA allocations;
  implemented bounded mask-aware project interfaces using only existing torch
  operations and input-derived device/dtype.

## DFAT

- Source: `https://github.com/fukexue/DFAT.git`
- Audited revision: `884149656199c734e2fceff1eda7d7d3b8ebf8c6`
- License found in checkout: MIT
- Copyright in that license: Copyright (c) 2022 Zheng Qin
- License text: `third_party_licenses/DFAT-MIT.txt`
- Project use: clean-room second-queue focus/saliency interface only.
- Changes: no fine-scale integration, GeoTransformer native extension,
  CPython 3.8 binary, or CUDA chamfer source is included; the interface is not
  enabled by the baseline config.

## TAX-Pose

- Source: `https://github.com/r-pad/taxpose.git`
- Audited revision: `0c4298fa0486fd09e63bf24d618a579b66ba0f18`
- License: MIT
- Copyright: Copyright (c) 2024 Ben Eisner
- License text: `third_party_licenses/taxpose-MIT.txt`
- Project use: clean-room per-point importance/weight head idea.
- Changes: removed the DCP/flow network, PyTorch3D, PyG, DGL, pose solver, and
  Python `<3.10` package coupling; the local head returns only masked logits.
- Metadata note: upstream `pyproject.toml` names `LICENSE.txt`, while this local
  revision contains the license as `LICENSE`.

## DETR

- Source: `https://github.com/facebookresearch/detr.git`
- Audited revision: `29901c51d7fe8712168b8d0d64351170bc0f83e0`
- License: Apache License 2.0
- Copyright headers: Copyright (c) Facebook, Inc. and its affiliates. All Rights
  Reserved.
- License text: `third_party_licenses/detr-Apache-2.0.txt`
- Upstream NOTICE file: none present in the audited checkout.
- Project use: learnable pose-query decoder pattern, no-pose slots, auxiliary
  decoder outputs, and one-to-one set assignment.
- Modification notice: detection boxes/classes and image backbones were replaced
  with direct 6D-rotation/translation/uncertainty outputs over point-token
  memory. The SciPy box matcher was replaced by a project-local exact small-K
  assignment interface. These changes are not endorsed by the upstream authors.

## PointDSC — reference only, no copied source

- Source: `https://github.com/XuyangBai/PointDSC.git`
- Audited revision: `b009d536ac10b570853833f2178397c154745da9`
- License finding: **NOASSERTION**. The local checkout has no repository-level
  `LICENSE`/`COPYING` file and README contains no source-code license grant.
- Project use: the published architectural idea of pairwise Euclidean-distance
  preservation was considered for a second-queue clean-room interface.
- Excluded content: no code from `models/PointDSC.py` or `models/common.py` and
  no `utils/libpmc.so` binary is copied, adapted, linked, loaded, or distributed
  by this repository.

PointDSC's technical source-port status remains `rejected` unless an explicit
license grant is obtained and re-audited. Individual files elsewhere in that
checkout with their own third-party headers do not grant a license to the
PointDSC repository as a whole.

## Distribution obligations

When distributing project source or binaries that contain an adapted derivative:

1. include this notice file;
2. include the applicable texts from `third_party_licenses/`;
3. preserve copyright, attribution, patent, trademark, and modification notices;
4. preserve each derivative file's source repository, full commit, original
   relative path, license, and change summary;
5. do not describe upstream authors as endorsing this project;
6. do not distribute PointDSC source or its prebuilt PMC binary under the current
   `NOASSERTION` finding.

This inventory documents local engineering due diligence; it is not legal advice.
