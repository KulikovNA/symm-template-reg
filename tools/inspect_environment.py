#!/usr/bin/env python3
"""Inspect the active Python environment and adjacent reference repositories.

The command is deliberately read-only unless ``--json-out`` is supplied.  It
does not import any adjacent repository, install packages, or build native
extensions.  Only the Python standard library is required; importing torch is
best-effort so the report remains useful in a partially provisioned setup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REFERENCE_REPOSITORIES = (
    "RegTR",
    "PointTransformerV3",
    "Pointcept",
    "RoITr",
    "GeoTransformer",
    "DFAT",
    "taxpose",
    "PointDSC",
    "detr",
)

PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "h5py": "h5py",
    "pyyaml": "PyYAML",
    "scipy": "scipy",
    "einops": "einops",
    "open3d": "open3d",
    "trimesh": "trimesh",
    "plyfile": "plyfile",
    "addict": "addict",
    "timm": "timm",
    "spconv": "spconv",
    "torch_scatter": "torch-scatter",
    "torch_cluster": "torch-cluster",
    "torch_sparse": "torch-sparse",
    "torch_geometric": "torch-geometric",
    "flash_attn": "flash-attn",
    "minkowski_engine": "MinkowskiEngine",
    "pytorch3d": "pytorch3d",
    "dgl": "dgl",
    "scikit_learn": "scikit-learn",
    "lap": "lap",
}

LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
)

MANIFEST_NAMES = (
    "requirements.txt",
    "requirements-gpu.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
)

NATIVE_SUFFIXES = {".cu", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".so"}


def _run(command: list[str], *, cwd: Path | None = None, timeout: float = 8.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": f"timeout after {timeout:g}s",
        }
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_value(path: Path, *args: str) -> str | None:
    result = _run(["git", *args], cwd=path)
    if result["returncode"] != 0:
        return None
    value = result["stdout"].strip()
    return value or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _license_id(path: Path | None) -> str:
    if path is None:
        return "NOASSERTION"
    text = path.read_text(encoding="utf-8", errors="replace")[:4096].lower()
    if "mit license" in text and "permission is hereby granted" in text:
        return "MIT"
    if "apache license" in text and "version 2.0" in text:
        return "Apache-2.0"
    if "gnu general public license" in text:
        return "GPL"
    return "NOASSERTION"


def _iter_native_files(repository: Path) -> Iterable[Path]:
    for path in repository.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in NATIVE_SUFFIXES:
            continue
        relative = path.relative_to(repository)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        yield relative


def _iter_dependency_manifests(repository: Path) -> Iterable[Path]:
    """Yield compact dependency/build manifests, including nested extension setups."""

    for path in repository.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        name = path.name.lower()
        is_manifest = (
            name in {item.lower() for item in MANIFEST_NAMES}
            or (name.startswith("requirements") and name.endswith(".txt"))
            or (name.startswith("environment") and name.endswith((".yml", ".yaml")))
        )
        # Keep the report bounded while still reaching RegTR's nested KPConv
        # wrapper setup files.
        if is_manifest and len(relative.parts) <= 9:
            yield relative


def _repository_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_dir()}
    if not path.is_dir():
        return report

    head = _git_value(path, "rev-parse", "HEAD")
    status = _git_value(path, "status", "--porcelain=v1", "--untracked-files=no")
    license_path = next((path / name for name in LICENSE_NAMES if (path / name).is_file()), None)
    manifests = sorted(str(item) for item in _iter_dependency_manifests(path))
    native_files = sorted(str(item) for item in _iter_native_files(path))
    native_counts = Counter(Path(item).suffix.lower() for item in native_files)

    report.update(
        {
            "is_git_repository": head is not None,
            "head": head,
            "branch": _git_value(path, "branch", "--show-current"),
            "describe": _git_value(path, "describe", "--tags", "--always", "--dirty"),
            "origin": _git_value(path, "remote", "get-url", "origin"),
            "commit_date": _git_value(path, "show", "-s", "--format=%cI", "HEAD"),
            "worktree_clean_tracked": status in (None, ""),
            "license": {
                "spdx": _license_id(license_path),
                "path": license_path.name if license_path else None,
                "sha256": _sha256(license_path) if license_path else None,
            },
            "dependency_manifests": manifests,
            "native_files": native_files,
            "native_file_counts": dict(sorted(native_counts.items())),
        }
    )
    return report


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for key, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            versions[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def _torch_report() -> dict[str, Any]:
    report: dict[str, Any] = {"importable": False}
    captured: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            import torch

            report.update(
                {
                    "importable": True,
                    "version": torch.__version__,
                    "git_version": getattr(torch.version, "git_version", None),
                    "cuda_build": getattr(torch.version, "cuda", None),
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count(),
                    "cudnn_version": torch.backends.cudnn.version(),
                }
            )
            if report["cuda_available"]:
                report["cuda_devices"] = [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                    }
                    for index in range(report["cuda_device_count"])
                ]
            captured.extend(str(record.message) for record in warning_records)
    except Exception as exc:  # pragma: no cover - depends on host installation
        report["error"] = f"{type(exc).__name__}: {exc}"
    if captured:
        report["warnings"] = captured
    return report


def _tool_report(executable: str, args: list[str]) -> dict[str, Any]:
    resolved = shutil.which(executable)
    if resolved is None:
        return {"available": False, "path": None, "version_output": None}
    result = _run([resolved, *args])
    output = result["stdout"] or result["stderr"]
    return {
        "available": result["returncode"] == 0,
        "path": resolved,
        "returncode": result["returncode"],
        "version_output": output,
    }


def inspect(third_party_root: Path) -> dict[str, Any]:
    third_party_root = third_party_root.expanduser().resolve()
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "read_only": True,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "packages": _package_versions(),
        "torch_runtime": _torch_report(),
        "build_and_gpu_tools": {
            "nvidia_smi": _tool_report(
                "nvidia-smi",
                ["--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            ),
            "nvcc": _tool_report("nvcc", ["--version"]),
            "cxx": _tool_report("c++", ["--version"]),
            "ninja": _tool_report("ninja", ["--version"]),
            "cmake": _tool_report("cmake", ["--version"]),
        },
        "third_party_root": str(third_party_root),
        "repositories": {
            name: _repository_report(third_party_root / name) for name in REFERENCE_REPOSITORIES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--third-party-root",
        type=Path,
        required=True,
        help="Directory containing the nine adjacent reference repositories.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional output path. Without it the command only writes JSON to stdout.",
    )
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON instead of indented JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect(args.third_party_root)
    indent = None if args.compact else 2
    payload = json.dumps(report, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json_out is not None:
        output = args.json_out.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    missing = [name for name, value in report["repositories"].items() if not value["exists"]]
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
