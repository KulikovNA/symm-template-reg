# Environment compatibility

This document records the active `fracs` environment as observed on
2026-07-15. It is an audit snapshot, not an installation recipe. No package was
installed, removed, downgraded, or rebuilt during the audit.

## Active runtime

| Item | Observed value |
|---|---|
| Conda environment | `fracs` |
| Prefix | `/home/nikita/anaconda3/envs/fracs` |
| Python | CPython 3.10.19 |
| Python executable | `/home/nikita/anaconda3/envs/fracs/bin/python` |
| PyTorch | `2.9.1+cu130` |
| PyTorch CUDA build | 13.0 |
| PyTorch Git revision | `5811a8d7da873dd699ff6687092c225caffcf1bb` |
| cuDNN reported by torch | 9.13 (`91300`) |
| torchvision | `0.24.1+cu130` |
| torchaudio | `2.9.1+cu130` |
| NumPy | `1.24.4` |
| OS/kernel | Linux 6.8.0-134-generic, x86_64 |
| C++ compiler | GCC 13.3.0 at `/usr/bin/c++` |
| CMake | 3.28.3 |

`torch.cuda.is_available()` was false and `torch.cuda.device_count()` was zero
inside the audit process. Torch warned that NVML could not initialize;
`nvidia-smi` existed but returned code 9 because it could not communicate with
the driver. Therefore this audit confirms the CPU path only. It does **not**
claim that the physical workstation has no GPU or that CUDA will remain
unavailable outside this process isolation.

`nvcc` and `ninja` were not found. This makes all proposed in-place CUDA
extension builds unavailable in the audited session even before considering
torch/CUDA ABI compatibility.

## Relevant installed packages

| Package group | State |
|---|---|
| torch / torchvision / torchaudio | installed at the required 2.9.1/0.24.1/cu130 stack |
| NumPy | installed, 1.24.4 |
| h5py | installed, 3.15.1 |
| PyYAML | installed, 6.0.1 |
| SciPy | missing |
| einops | missing |
| Open3D | missing |
| trimesh | missing |
| plyfile | missing |
| addict / timm | missing |
| spconv | missing |
| torch-scatter / torch-cluster / torch-sparse | missing |
| torch-geometric | missing |
| flash-attn | missing |
| MinkowskiEngine | missing |
| PyTorch3D | missing |
| DGL | missing |
| scikit-learn | missing |
| `lap` | missing |

The environment also contains editable/installed packages named
`frag-geometry-engine` and `frag-template-reg`. Their presence is not permission
to import them: this repository's production path must remain independent of
both. The no-production-import check should treat them as forbidden even though
package discovery can see them.

## Compatibility matrix

The labels below distinguish a full upstream checkout from the small source
unit considered for project-local adaptation.

| Source | Full checkout in `fracs` | Small candidate | Required action |
|---|---|---|---|
| RegTR | blocked by MinkowskiEngine, PyTorch3D, SciPy/Open3D, and package auto-import | self/cross attention is compatible | keep only project-local torch attention and heads; use SimplePointEncoder |
| PointTransformerV3 | blocked by addict, spconv, torch-scatter and timm | serialization is potentially pure torch, full encoder is not | expose PTv3 as optional/unavailable for iteration one |
| Pointcept | blocked by absent PyG/spconv/native libs and a torch 2.5/cu124 reference environment | serialization ideas are inspectable | keep project `PackedPointBatch`; do not build Pointcept libs |
| RoITr | blocked by old pointops CUDA and missing SciPy/einops/Open3D | PPF and positional embedding are compatible | use pure torch PPF and caller-provided neighborhoods/masks |
| GeoTransformer | blocked by unbuilt extension and missing SciPy/einops/Open3D | embedding, coarse match and Sinkhorn are portable | remove `.cuda()`, infer device/dtype from inputs, avoid extension imports |
| DFAT | blocked by missing einops/native code and CPython 3.8 binary | focus-selection idea only | retain a second-queue interface; do not enable in baseline |
| taxpose | package declares Python `<3.10`; GPU stack targets torch 2.0/cu118 | per-point MLP/soft-weight idea is portable | implement a small local head without PyTorch3D/PyG/DGL |
| PointDSC | obsolete environment and missing dependencies | core nonlocal block technically runs | no source port because local license is NOASSERTION; clean-room math only |
| DETR | package import blocked by SciPy matcher | transformer decoder is compatible | use torch decoder and project-local small-K assignment |

## Read-only compatibility smokes

These tests loaded local source files with Python bytecode writing disabled. No
adjacent checkout was edited.

| Test | Result |
|---|---|
| RegTR package import | failed at missing `MinkowskiEngine` |
| RegTR isolated `src/models/transformer/transformers.py` | finite outputs `(2,5,2,16)` and `(2,7,2,16)` |
| PointTransformerV3 `model` import | failed at missing `addict` |
| Pointcept point structure import | failed at missing `torch_scatter` |
| RoITr PPF/geometric embedding | finite outputs `(2,8,5,16)` and `(2,8,8,16)` |
| GeoTransformer upstream Sinkhorn on CPU | failed because upstream allocates with `.cuda()` |
| DFAT focus-attention import | failed at missing `einops` |
| TAX-Pose head package path | failed at missing transitive `scipy` |
| PointDSC `NonLocalBlock` | ran with output `(1,16,5)`; source remains license-blocked |
| DETR isolated decoder | finite output `(2,3,2,16)` |
| DETR package import | failed at missing `scipy` |

The isolated RegTR/DETR tests bypassed package `__init__` side effects to answer
the narrow question “does this pure torch source unit execute on torch 2.9?”
They do not validate the full upstream application.

## Safe implementation policy

Iteration one must use only the already installed torch/NumPy stack for the
core model. In particular:

1. Keep `SimplePointEncoder` as the guaranteed backbone.
2. Implement kNN, masking, PPF, matching, and Sinkhorn with project-local torch
   operations.
3. Keep PTv3, DFAT and PointDSC modules optional and outside the baseline.
4. Use a project-local exact small-K assignment implementation instead of
   making SciPy mandatory.
5. Use the internal PLY reader when trimesh, plyfile and Open3D are absent.
6. Never call `.cuda()` inside reusable modules; derive device and dtype from
   input tensors.
7. Do not install or compile anything merely to make an upstream package
   importable.

If a future experiment needs PTv3 or a native point operator, it requires a
separate, explicit compatibility task: identify a torch 2.9/CUDA 13 supported
release or pure-torch replacement, record exact versions, obtain approval, and
only then install/build. It must not modify the existing torch, torchvision,
CUDA, or NumPy versions.

## Reproduce the snapshot

The inspector uses only the standard library plus a best-effort import of the
already installed torch. By default it prints JSON and writes no files:

```bash
python tools/inspect_environment.py \
  --third-party-root "/home/nikita/disser/fragment-template-registration-lab"
```

For an explicit artifact, pass `--json-out <path>`. The command reports Git
revisions, license hashes, native file paths/counts, package versions, build
tools, and torch CUDA visibility. Static source-to-target decisions remain in
`third_party_modules.json` so an environment probe cannot silently change legal
or architectural policy.
