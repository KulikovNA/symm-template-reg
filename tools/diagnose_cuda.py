#!/usr/bin/env python3
"""Capture host, Python, driver, PyTorch, and full-model CUDA diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import getpass
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WARNING_FLAGS = {
    "debug_training_on_test_split": True,
    "train_and_validation_use_same_samples": True,
    "results_are_not_final_evaluation": True,
}


class CommandRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def run(self, command: Sequence[str], *, shell_text: str | None = None) -> str:
        display = shell_text or " ".join(command)
        try:
            completed = subprocess.run(
                list(command),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
            record = {
                "command": display,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except Exception:
            record = {
                "command": display,
                "returncode": None,
                "stdout": "",
                "stderr": traceback.format_exc(),
            }
        self.records.append(record)
        return str(record["stdout"])


def _unique_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = f"cuda_diagnostics_{stamp}"
        if suffix:
            name += f"_{suffix:03d}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot create unique CUDA diagnostic below {root}")


def _jsonable_device_properties(properties: Any) -> dict[str, Any]:
    fields = (
        "name",
        "major",
        "minor",
        "total_memory",
        "multi_processor_count",
        "warp_size",
    )
    return {name: getattr(properties, name, None) for name in fields}


def _python_torch_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {
        "sys_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "site_packages": site.getsitepackages(),
        "user": getpass.getuser(),
        "uid": os.getuid(),
        "groups": os.getgroups(),
        "torch_file": torch.__file__,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torchvision_version": None,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available_before_init": torch.cuda.is_available(),
        "cuda_device_count_before_init": torch.cuda.device_count(),
        "cuda_arch_list": torch.cuda.get_arch_list(),
        "torch_config": torch.__config__.show(),
    }
    try:
        import torchvision

        result["torchvision_version"] = torchvision.__version__
        result["torchvision_file"] = torchvision.__file__
    except Exception:
        result["torchvision_import_traceback"] = traceback.format_exc()
    try:
        torch.cuda.init()
        result["cuda_init_ok"] = True
        result["cuda_init_traceback"] = None
    except Exception:
        result["cuda_init_ok"] = False
        result["cuda_init_traceback"] = traceback.format_exc()
    result["cuda_available_after_init"] = torch.cuda.is_available()
    result["cuda_device_count_after_init"] = torch.cuda.device_count()
    if result["cuda_init_ok"] and torch.cuda.device_count() > 0:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_properties": _jsonable_device_properties(properties),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "mem_get_info_bytes": list(torch.cuda.mem_get_info()),
            }
        )
    return result


def _libcuda_diagnostics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        library = ctypes.CDLL("libcuda.so.1")
        result.update(available=True, loaded_object=repr(library), error=None)
    except Exception as exc:
        result.update(available=False, loaded_object=None, error=repr(exc))
    return result


def _finite_gradients(module: torch.nn.Module) -> bool:
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients)


def _cuda_smoke(config_path: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "device_name": None,
        "device_capability": None,
        "torch_cuda_build": torch.version.cuda,
        "driver_version": None,
        "tensor_creation_ok": False,
        "matmul_ok": False,
        "forward_ok": False,
        "backward_ok": False,
        "optimizer_step_ok": False,
        "amp_ok": False,
        "model_forward_ok": False,
        "model_loss_ok": False,
        "model_backward_ok": False,
        "model_optimizer_step_ok": False,
        "model_amp_dtype": None,
        "nonfinite_gradient_parameters": [],
        "device_mismatch_detected": False,
        "finite_values": False,
        "finite_gradients": False,
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
        "error": None,
        "traceback": None,
    }
    if not torch.cuda.is_available():
        metrics["error"] = "CUDA is unavailable to this Python process"
        return metrics
    try:
        device = torch.device("cuda:0")
        metrics["device_name"] = torch.cuda.get_device_name(0)
        metrics["device_capability"] = list(torch.cuda.get_device_capability(0))
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(64, 64, device=device, requires_grad=True)
        metrics["tensor_creation_ok"] = True
        y = x @ x.transpose(0, 1)
        metrics["matmul_ok"] = bool(torch.isfinite(y).all())
        y.square().mean().backward()
        metrics["backward_ok"] = bool(torch.isfinite(x.grad).all())

        linear = torch.nn.Linear(64, 32).to(device)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            simple_loss = linear(x.detach()).square().mean()
        scaler.scale(simple_loss).backward()
        scaler.unscale_(optimizer)
        if not _finite_gradients(linear):
            raise RuntimeError("simple AMP smoke produced non-finite gradients")
        scaler.step(optimizer)
        scaler.update()
        metrics["optimizer_step_ok"] = True
        metrics["amp_ok"] = True

        from copy import deepcopy

        from symm_template_reg.config import load_config
        from symm_template_reg.datasets import FragmentTemplateRegistrationDataset
        from symm_template_reg.engine.evaluator import move_to_device
        from symm_template_reg.engine.trainer import compute_training_losses
        from symm_template_reg.models import build_model, register_all_modules
        from symm_template_reg.models.losses import PoseSetLoss
        from symm_template_reg.registry import COLLATE_FUNCTIONS, build_from_cfg

        config = load_config(config_path)
        register_all_modules()
        dataset_cfg = deepcopy(dict(config["dataset"]))
        dataset_cfg.pop("type", None)
        dataset_cfg["max_samples"] = 1
        dataset = FragmentTemplateRegistrationDataset(**dataset_cfg)
        collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
        batch = move_to_device(collate([dataset[0]]), device)
        model = build_model(config["model"]).to(device).train()
        model_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model_amp_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        metrics["model_amp_dtype"] = str(model_amp_dtype).replace("torch.", "")
        model_scaler = torch.amp.GradScaler(
            "cuda", enabled=model_amp_dtype == torch.float16
        )
        criterion = PoseSetLoss(
            translation_weight=float(config["loss"]["translation_cost_weight"]),
            rotation_weight=float(config["loss"]["rotation_cost_weight"]),
            classification_weight=float(
                config["loss"]["pose_query_classification_weight"]
            ),
            auxiliary_weight=0.0,
        )
        model_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=model_amp_dtype):
            prediction = model(batch)
            metrics["model_forward_ok"] = True
            total, _ = compute_training_losses(
                prediction,
                batch,
                criterion,
                {
                    "auxiliary_registration_losses": False,
                    "pose_decoder_auxiliary_loss": False,
                },
            )
            metrics["model_loss_ok"] = bool(torch.isfinite(total))
        model_scaler.scale(total).backward()
        model_scaler.unscale_(model_optimizer)
        metrics["nonfinite_gradient_parameters"] = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        metrics["finite_gradients"] = _finite_gradients(model)
        if not metrics["finite_gradients"]:
            raise RuntimeError("full model produced missing or non-finite gradients")
        model_scaler.step(model_optimizer)
        model_scaler.update()
        torch.cuda.synchronize(device)
        metrics["model_backward_ok"] = True
        metrics["model_optimizer_step_ok"] = True
        metrics["forward_ok"] = True
        metrics["finite_values"] = metrics["matmul_ok"] and metrics["model_loss_ok"]
        metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / 2**20
    except Exception as exc:
        metrics["error"] = repr(exc)
        metrics["traceback"] = traceback.format_exc()
    return metrics


def run_diagnostics(
    output_root: str | Path,
    config_path: str,
    *,
    prior_restricted_gpu_hidden: bool = False,
) -> Path:
    output = _unique_directory(Path(output_root).expanduser().resolve())
    recorder = CommandRecorder()
    python = sys.executable
    commands: list[tuple[list[str], str | None]] = [
        (["which", "python"], None),
        (["which", "pip"], None),
        ([python, "--version"], None),
        ([python, "-m", "pip", "--version"], None),
        ([python, "-m", "pip", "show", "torch"], None),
        ([python, "-m", "pip", "show", "torchvision"], None),
        (["conda", "info"], None),
        (["conda", "list"], None),
        (["conda", "list", "--explicit"], None),
        (["nvidia-smi"], None),
        (["nvidia-smi", "-L"], None),
        (["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], None),
        (["bash", "-lc", "echo \"$CUDA_VISIBLE_DEVICES\""], "echo $CUDA_VISIBLE_DEVICES"),
        (["bash", "-lc", "echo \"$NVIDIA_VISIBLE_DEVICES\""], "echo $NVIDIA_VISIBLE_DEVICES"),
        (["bash", "-lc", "echo \"$LD_LIBRARY_PATH\""], "echo $LD_LIBRARY_PATH"),
        (["bash", "-lc", "echo \"$PATH\""], "echo $PATH"),
        (["id"], None),
        (["bash", "-lc", "ls -l /dev/nvidia* 2>/dev/null"], "ls -l /dev/nvidia* 2>/dev/null"),
        (["bash", "-lc", "ldconfig -p | grep -E 'libcuda|libnvidia' || true"], "ldconfig -p | grep -E 'libcuda|libnvidia' || true"),
    ]
    environment_sections = []
    for command, shell_text in commands:
        output_text = recorder.run(command, shell_text=shell_text)
        environment_sections.append(f"$ {shell_text or ' '.join(command)}\n{output_text}")
    collect_env = recorder.run([python, "-m", "torch.utils.collect_env"])
    (output / "collect_env.txt").write_text(collect_env, encoding="utf-8")
    env_vars = {
        key: os.environ.get(key)
        for key in (
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "LD_LIBRARY_PATH",
            "PATH",
        )
    }
    python_torch = _python_torch_diagnostics()
    libcuda = _libcuda_diagnostics()
    device_paths = {
        path: {
            "exists": Path(path).exists(),
            "stat": (str(Path(path).stat()) if Path(path).exists() else None),
        }
        for path in ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")
    }
    smoke = _cuda_smoke(config_path)
    query_record = next(
        record
        for record in recorder.records
        if record["command"].startswith("nvidia-smi --query-gpu=")
    )
    if query_record["returncode"] == 0 and query_record["stdout"].strip():
        columns = [value.strip() for value in query_record["stdout"].splitlines()[0].split(",")]
        if len(columns) >= 2:
            smoke["driver_version"] = columns[1]
    (output / "cuda_smoke_metrics.json").write_text(
        json.dumps({**WARNING_FLAGS, **smoke}, indent=2) + "\n", encoding="utf-8"
    )
    nvidia_record = next(
        record for record in recorder.records if record["command"] == "nvidia-smi"
    )
    isolation_evidence = (
        nvidia_record["returncode"] != 0
        and not device_paths["/dev/nvidia0"]["exists"]
        and not device_paths["/dev/nvidiactl"]["exists"]
    )
    unrestricted_gpu_visible = (
        nvidia_record["returncode"] == 0
        and device_paths["/dev/nvidia0"]["exists"]
        and bool(python_torch["cuda_available_after_init"])
    )
    conclusion = (
        "The same fracs Python and PyTorch build sees the GPU when executed with "
        "GPU device access, while the prior restricted Codex process had no "
        "/dev/nvidia* nodes and nvidia-smi could not contact the driver. The exact "
        "cause of the previous false result was Codex sandbox/device-namespace "
        "isolation. No package or driver change is required."
        if prior_restricted_gpu_hidden and unrestricted_gpu_visible
        else (
        "The current process namespace does not expose NVIDIA device nodes; "
        "nvidia-smi cannot communicate with the driver. Because nvidia-smi works "
        "in the user's terminal, this is execution-environment GPU isolation, not "
        "evidence that the host driver or the fracs PyTorch build is broken."
        if isolation_evidence
        else "No GPU-isolation conclusion was proven; inspect cuda_init_traceback and command logs."
        )
    )
    report = {
        **WARNING_FLAGS,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "diagnostic_directory": str(output),
        "repo_root": str(REPO_ROOT),
        "config_path": str(Path(config_path).resolve()),
        "environment": env_vars,
        "python_path_matches_sys_executable": (
            Path(shutil.which("python") or "").resolve() == Path(sys.executable).resolve()
        ),
        "pip_belongs_to_same_conda_prefix": str(
            Path(shutil.which("pip") or "").resolve()
        ).startswith(str(Path(sys.prefix).resolve())),
        "python_and_torch": python_torch,
        "libcuda_so_1": libcuda,
        "device_paths": device_paths,
        "nvidia_smi_returncode": nvidia_record["returncode"],
        "nvidia_smi_stderr": nvidia_record["stderr"],
        "environment_isolation_evidence": isolation_evidence,
        "prior_restricted_gpu_hidden": prior_restricted_gpu_hidden,
        "gpu_visible_with_device_access": unrestricted_gpu_visible,
        "conclusion": conclusion,
        "cuda_smoke": smoke,
    }
    (output / "cuda_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    environment_sections.extend(
        [
            "Environment variables\n" + json.dumps(env_vars, indent=2),
            "Python/Torch\n" + json.dumps(python_torch, indent=2),
            "libcuda.so.1\n" + json.dumps(libcuda, indent=2),
            "Device paths\n" + json.dumps(device_paths, indent=2),
        ]
    )
    (output / "environment.txt").write_text(
        "\n\n".join(environment_sections) + "\n", encoding="utf-8"
    )
    (output / "commands.log").write_text(
        "\n\n".join(
            f"$ {record['command']}\nreturncode={record['returncode']}\n{record['stdout']}"
            for record in recorder.records
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_parts = [
        f"$ {record['command']}\n{record['stderr']}"
        for record in recorder.records
        if record["stderr"]
    ]
    if python_torch.get("cuda_init_traceback"):
        stderr_parts.append(
            "torch.cuda.init()\n" + str(python_torch["cuda_init_traceback"])
        )
    if smoke.get("traceback"):
        stderr_parts.append("CUDA smoke\n" + str(smoke["traceback"]))
    (output / "stderr.log").write_text(
        "\n\n".join(stderr_parts) + "\n", encoding="utf-8"
    )
    markdown = [
        "# CUDA diagnostic",
        "",
        "- `debug_training_on_test_split = true`",
        "- `train_and_validation_use_same_samples = true`",
        "- `results_are_not_final_evaluation = true`",
        f"- Python: `{sys.executable}`",
        f"- Conda environment: `{env_vars.get('CONDA_DEFAULT_ENV')}`",
        f"- Torch: `{torch.__version__}` (CUDA build `{torch.version.cuda}`)",
        f"- `nvidia-smi` return code: `{nvidia_record['returncode']}`",
        f"- `/dev/nvidia0` visible: `{device_paths['/dev/nvidia0']['exists']}`",
        f"- `libcuda.so.1` loadable: `{libcuda['available']}`",
        f"- `torch.cuda.is_available()`: `{python_torch['cuda_available_after_init']}`",
        f"- `torch.cuda.init()` succeeded: `{python_torch['cuda_init_ok']}`",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        "No package versions were changed by this diagnostic.",
    ]
    (output / "cuda_diagnostics.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/home/nikita/disser/fragment-template-registration-lab/work_dirs",
    )
    parser.add_argument(
        "--config", default="configs/debug/test_overfit_faces840_gpu.py"
    )
    parser.add_argument(
        "--prior-restricted-gpu-hidden",
        action="store_true",
        help="Record that a prior restricted Codex process lacked /dev/nvidia*.",
    )
    args = parser.parse_args()
    output = run_diagnostics(
        args.output_root,
        args.config,
        prior_restricted_gpu_hidden=args.prior_restricted_gpu_hidden,
    )
    print(json.dumps({"status": "ok", "output_dir": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
