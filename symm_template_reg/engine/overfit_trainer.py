"""Epoch-based faces840 debug overfit trainer with best-only checkpoints."""

from __future__ import annotations

import csv
import datetime as dt
import gc
import json
import math
import os
import platform
import pprint
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from symm_template_reg.datasets.fragment_mesh_filter import sha256_file
from symm_template_reg.datasets.multi_view_batch_sampler import MultiViewBatchSampler
from symm_template_reg.config import validate_data_policy
from symm_template_reg.engine.evaluator import move_to_device
from symm_template_reg.engine.history import TrainingHistory
from symm_template_reg.engine.metrics import (
    aggregate_metric_rows,
    batch_pose_metric_rows,
)
from symm_template_reg.engine.overfit_manifest import (
    WARNING_FLAGS,
    load_faces840_manifest,
    validate_overfit_flags,
)
from symm_template_reg.engine.multifragment_overfit import (
    multifragment_stage_gates,
    validate_multifragment_config,
    worst_multifragment_sample_score,
)
from symm_template_reg.engine.seed import seed_everything
from symm_template_reg.engine.single_fragment import (
    TrainingCounters,
    apply_trainable_prefixes,
    build_selective_optimizer_parameter_groups,
    load_model_initialization,
    region_class_distribution,
    validate_single_fragment_config,
    world_pose_consistency,
)
from symm_template_reg.engine.training_budget import (
    early_stopping_is_eligible,
    resolve_training_budget,
    sample_exposure_statistics,
)
from symm_template_reg.engine.frozen_feature_cache import (
    cached_fine_coordinate_forward,
    capture_fine_adapter_inputs,
    fine_coordinate_active_loss,
    frozen_module_state_sha256,
)
from symm_template_reg.visualization.multifragment_debug import (
    export_multifragment_overviews,
)
from symm_template_reg.engine.view_ladder import (
    assignment_switch_rate,
    query_assignment_summary,
    query_world_consistency,
)
from symm_template_reg.evaluation.context_conditioning import (
    context_conditioning_diagnostics,
    context_conditioning_metrics,
)
from symm_template_reg.evaluation.diagnostic_gates import (
    evaluate_correspondence_diagnostic_gates,
)
from symm_template_reg.evaluation.plateau import detect_rotation_context_plateau
from symm_template_reg.evaluation.joint_stage import (
    check_joint_stage,
    materialize_best_evaluation,
)
from symm_template_reg.evaluation.active_coordinate import (
    active_row,
    active_world_metrics,
    evaluate_active_sample,
    practical_sample_gate,
    practical_surface_sample_gate,
    pose_placement_sample_gate,
    strict_surface_sample_gate,
    worst_ten_view_sample_score,
    worst_sample_projection_score,
    worst_sample_practical_score,
)
from symm_template_reg.engine.trainer import compute_training_losses, resolve_device
from symm_template_reg.models import build_model, register_all_modules
from symm_template_reg.models.detectors.coordinate_guided_surface_registration_v3 import state_dict_sha256
from symm_template_reg.geometry.triangle_surface import closest_points_on_triangle_mesh
from symm_template_reg.models.pose.pose_representation import transform_points
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance
from symm_template_reg.models.losses import PoseSetLoss
from symm_template_reg.registry import COLLATE_FUNCTIONS, DATASETS, build_from_cfg
from symm_template_reg.visualization.prediction_debug import (
    export_prediction_visualizations,
    select_debug_samples,
)


FINE_GRADIENT_PREFIXES = {
    "fine_adapter": "correspondence_head.fine_feature_adapter",
    "fine_triangle_head": "correspondence_head.fine_candidate_triangle_head",
    "aux_coordinate_head": "correspondence_head.fine_coordinate_auxiliary_head",
    "last_interaction_layer": "interaction_transformer.layers.3",
    "dense_observed_projection": "dense_observed_fine_projection",
    "fine_template_projection": "fine_template_projection",
    "observed_encoder": "observed_encoder",
    "template_encoder": "template_encoder",
    "interaction_transformer": "interaction_transformer",
    "dual_stream_geometry_encoder": "dual_stream_geometry_encoder",
    "v3_fine_adapter": "fine_feature_adapter",
    "v3_template_context_projection": "template_context_projection",
    "v3_coordinate_head": "canonical_coordinate_head",
}


def _module_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    values: dict[str, float] = {}
    named = list(model.named_parameters())
    for label, prefix in FINE_GRADIENT_PREFIXES.items():
        square_sum = sum(
            float(parameter.grad.detach().float().square().sum())
            for name, parameter in named
            if (name == prefix or name.startswith(prefix + "."))
            and parameter.grad is not None
        )
        values[label] = math.sqrt(square_sum)
    return values


def clean_active_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields produced by the clean q/projection path."""

    exact_names = {
        "sample_id", "scene_id", "frame_id", "fragment_id", "num_shell_points",
        "target_leakage_detected", "active_nonfinite_detected",
        "selected_shared_symmetry_element", "correspondence_prediction_summary",
        "correspondence_geometry_rank", "correspondence_covariance_min_max_ratio",
        "predicted_covariance_eigenvalues", "gt_covariance_eigenvalues",
        "eigenvalue_ratio", "rank_margin_m2",
    }
    prefixes = (
        "aux_coordinate_", "exact_global_", "k16_", "loss_", "raw_",
        "weighted_loss_", "fine_feature_", "attention_",
    )
    return {
        key: value
        for key, value in row.items()
        if key in exact_names or key.startswith(prefixes)
    }


def _unique_run_directory(root: Path, experiment_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for suffix in range(1000):
        name = f"{experiment_name}_{stamp}"
        if suffix:
            name += f"_{suffix:03d}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot create unique run directory below {root}")


def validate_initialization_request(
    config: Mapping[str, Any],
    *,
    from_scratch: bool,
    resume: str | Path | None,
    init_checkpoint: str | Path | None,
    init_modules: Sequence[str] | None,
) -> None:
    """Validate scratch/transfer initialization before any run is created."""

    if resume is not None and init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if from_scratch and (
        resume is not None or init_checkpoint is not None or init_modules
    ):
        raise ValueError("--from-scratch excludes checkpoint loading")
    configured_initialization = str(config.get("initialization_mode", "transfer"))
    if configured_initialization == "scratch" and not from_scratch and resume is None:
        raise ValueError(
            "scratch config must be launched with --from-scratch or --resume"
        )
    if from_scratch and configured_initialization != "scratch":
        raise ValueError("--from-scratch requires initialization_mode='scratch'")
    if from_scratch and config.get("pretrained_checkpoint") is not None:
        raise ValueError("scratch config requires pretrained_checkpoint=null")
    if init_modules and init_checkpoint is None:
        raise ValueError("--init-modules requires --init-checkpoint")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Replace a mutable snapshot CSV without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _environment(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "debug_training_on_test_split": True,
        "train_and_validation_use_same_samples": True,
        "results_are_not_final_evaluation": True,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload.update(
            {
                "device_name": properties.name,
                "device_capability": [properties.major, properties.minor],
                "device_total_memory_bytes": properties.total_memory,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return payload


def _build_dataset(config: Mapping[str, Any]) -> Any:
    dataset_cfg = deepcopy(dict(config["dataset"]))
    data = config["data"]
    dataset_cfg["fragment_mesh_filter"] = deepcopy(data["fragment_mesh_filter"])
    dataset_cfg["observed_filter"] = deepcopy(data["observed_filter"])
    dataset_cfg["symmetry_region_activity"] = deepcopy(
        data.get("symmetry_region_activity", {})
    )
    return build_from_cfg(dataset_cfg, DATASETS)


def _select_manifest_samples_by_scene(
    samples: list[Mapping[str, Any]],
    scene_ids: Any,
) -> list[Mapping[str, Any]]:
    """Select manifest entries by explicit scene id, preserving manifest order."""
    if scene_ids is None:
        return list(samples)
    if isinstance(scene_ids, (str, bytes)):
        raise ValueError("data.scene_ids must be a sequence, not a string")

    requested = [str(scene_id) for scene_id in scene_ids]
    if not requested:
        raise ValueError("data.scene_ids must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("data.scene_ids must not contain duplicates")

    available = {str(sample["scene_id"]) for sample in samples}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            "data.scene_ids contains scenes absent from the accepted manifest: "
            + ", ".join(missing)
        )

    requested_set = set(requested)
    selected = [
        sample for sample in samples if str(sample["scene_id"]) in requested_set
    ]
    if not selected:
        raise ValueError("data.scene_ids selected no accepted manifest samples")
    return selected


def _build_pose_criterion(config: Mapping[str, Any]) -> PoseSetLoss:
    loss = config["loss"]
    criterion = PoseSetLoss(
        translation_weight=float(loss.get("translation_cost_weight", 10.0)),
        rotation_weight=float(loss.get("rotation_cost_weight", 1.0)),
        classification_weight=float(
            loss.get("pose_query_classification_weight", 0.2)
        ),
        auxiliary_weight=float(loss.get("pose_decoder_auxiliary_weight", 0.0)),
    )
    criterion.symmetry_pose_weight = float(loss.get("symmetry_pose_weight", 1.0))
    return criterion


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    max_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler_type = str(config.get("type", "cosine"))
    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if scheduler_type == "linear_warmup_constant":
        warmup_steps = int(config.get("warmup_optimizer_steps", 100))
        if warmup_steps < 1:
            raise ValueError("linear warmup requires warmup_optimizer_steps >= 1")
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(float(step + 1) / warmup_steps, 1.0)
        )
    if scheduler_type != "cosine":
        raise ValueError("overfit trainer supports scheduler.type=constant or cosine")
    warmup = int(config.get("warmup_epochs", 0))
    min_lr = float(config.get("min_lr", 1e-6))
    base_lr = float(optimizer.param_groups[0]["lr"])
    minimum_factor = min_lr / max(base_lr, 1e-12)

    def factor(epoch_index: int) -> float:
        completed = epoch_index + 1
        if warmup > 0 and completed <= warmup:
            return completed / warmup
        progress = (completed - warmup) / max(max_epochs - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _amp_settings(
    device: torch.device,
    train_config: Mapping[str, Any],
) -> tuple[bool, torch.dtype, str]:
    enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    requested = str(train_config.get("amp_dtype", "auto")).lower()
    if requested == "auto":
        dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    elif requested in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif requested in {"fp16", "float16"}:
        dtype = torch.float16
    else:
        raise ValueError("train.amp_dtype must be auto, bfloat16, or float16")
    return enabled, dtype, str(dtype).replace("torch.", "")


def _gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_peak_allocated_mb": 0.0,
            "gpu_peak_reserved_mb": 0.0,
        }
    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def _parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "buffers": sum(buffer.numel() for buffer in model.buffers()),
    }


def _print_model_summary(
    model: torch.nn.Module,
    *,
    run_dir: Path,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype_name: str,
    train_samples: int,
    validation_samples: int,
    train_batches: int,
    validation_batches: int,
    max_epochs: int,
    show_model: bool,
) -> None:
    counts = _parameter_counts(model)
    print("\n" + "=" * 88, flush=True)
    print("MODEL", flush=True)
    if show_model:
        print(model, flush=True)
    print("\nPARAMETERS", flush=True)
    print(
        f"  total:     {counts['total']:,}\n"
        f"  trainable: {counts['trainable']:,}\n"
        f"  frozen:    {counts['frozen']:,}\n"
        f"  buffers:   {counts['buffers']:,}\n"
        f"  fp32 parameter size: {counts['total'] * 4 / 2**20:.2f} MiB",
        flush=True,
    )
    print("\nPARAMETERS BY TOP-LEVEL MODULE", flush=True)
    for name, module in model.named_children():
        count = sum(parameter.numel() for parameter in module.parameters())
        print(f"  {name:<32} {count:>14,}", flush=True)
    print("\nRUN", flush=True)
    print(
        f"  output: {run_dir}\n"
        f"  device: {device}\n"
        f"  AMP: {amp_enabled} ({amp_dtype_name})\n"
        f"  epochs: {max_epochs}\n"
        f"  train: {train_samples} samples, {train_batches} batches/epoch\n"
        f"  validation: {validation_samples} samples, {validation_batches} batches/eval",
        flush=True,
    )
    print("=" * 88 + "\n", flush=True)


def _print_eval_metrics(
    epoch: int,
    metrics: Mapping[str, float],
    *,
    improved: bool,
    best_epoch: int | None,
) -> None:
    marker = "NEW BEST" if improved else f"best epoch: {best_epoch}"
    tqdm.write(f"\n[EVAL epoch {epoch:04d}] {marker}")
    for key in sorted(metrics):
        value = float(metrics[key])
        rendered = f"{value:.6g}" if math.isfinite(value) else str(value)
        tqdm.write(f"  {key:<52} {rendered:>14}")
    tqdm.write("")


def _loss_values(
    prediction: Any,
    batch: Mapping[str, Any],
    criterion: PoseSetLoss,
    loss_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return compute_training_losses(
        prediction,
        batch,
        criterion,
        {
            **dict(loss_config),
            "pose_decoder_auxiliary_loss": bool(
                float(loss_config.get("pose_decoder_auxiliary_weight", 0.0))
            ),
        },
    )


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: PoseSetLoss,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    loss_config: Mapping[str, Any],
    active_path_config: Mapping[str, Any] | None = None,
    *,
    epoch: int = 0,
    max_epochs: int | None = None,
    show_progress: bool = True,
    leave_progress: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    world_metadata: dict[tuple[str, int], tuple[Any, Any]] = {}
    loss_sums: dict[str, float] = {}
    sample_count = 0
    rotation_sum = 0.0
    translation_sum = 0.0
    active_path_enabled = bool((active_path_config or {}).get("enabled", False))
    clean_active_only = active_path_enabled and bool(
        (active_path_config or {}).get("clean_active_only", False)
    )
    epoch_label = f"{epoch:04d}"
    if max_epochs is not None:
        epoch_label += f"/{max_epochs:04d}"
    progress = tqdm(
        loader,
        desc=f"val   {epoch_label}",
        unit="batch",
        dynamic_ncols=True,
        leave=leave_progress,
        disable=not show_progress,
    )
    for batch in progress:
        moved = move_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction = model(moved)
            _, losses = _loss_values(prediction, moved, criterion, loss_config)
        batch_size = len(batch["sample_id"])
        for name, value in losses.items():
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * batch_size
        sample_count += batch_size
        batch_rows = batch_pose_metric_rows(
            prediction,
            moved,
            translation_cost_weight=float(
                loss_config.get("translation_cost_weight", 10.0)
            ),
            rotation_cost_weight=float(loss_config.get("rotation_cost_weight", 1.0)),
            ranking_config=loss_config.get("pose_query_ranking"),
            joint_loss_config=(
                loss_config.get("joint_surface_correspondence_pose_v3")
                if loss_config.get("joint_surface_correspondence_pose_v3", {}).get("enabled", False)
                else loss_config.get("joint_correspondence_pose")
            ),
        )
        auxiliary = prediction.correspondence_auxiliary
        if auxiliary is not None and "fine_aux_coordinate_normalized" in auxiliary:
            observed_padded = moved["observed"].to_padded()["points"]
            matched_targets = losses.get("matched_target_points_O")
            matched_poses = losses.get("matched_gt_pose_T_C_from_O")
            for row_index, row in enumerate(batch_rows):
                for component in (
                    "fine_coordinate_aux",
                    "fine_coordinate_aux_tail",
                    "raw_aux_rotation",
                    "raw_aux_translation",
                    "raw_aux_alignment",
                ):
                    for prefix in ("loss", "raw", "weighted_loss"):
                        key = f"per_sample_{prefix}_{component}"
                        if prefix == "loss":
                            key += "_normalized"
                        value = losses.get(key)
                        if isinstance(value, torch.Tensor) and value.ndim == 1:
                            row[f"{prefix}_{component}_per_sample"] = float(
                                value[row_index]
                            )
                mask = prediction.observed_valid_mask[row_index]
                vertices = moved["template_mesh_vertices_O"][row_index].to(device)
                faces = moved["template_mesh_faces"][row_index].to(device=device, dtype=torch.long)
                qn_all = auxiliary["fine_aux_coordinate_normalized"][row_index]
                qn = qn_all[mask]
                extent = (vertices.amax(0) - vertices.amin(0)).clamp_min(1e-8)
                q_aux = .5 * (qn + 1.0) * extent + vertices.amin(0)
                q_aux_all = .5 * (qn_all + 1.0) * extent + vertices.amin(0)
                if active_path_enabled:
                    active_result = evaluate_active_sample(
                        q_aux_O=q_aux_all,
                        valid_mask=mask,
                        target_O=matched_targets[row_index],
                        observed_C=observed_padded[row_index],
                        vertices_O=vertices,
                        faces=faces,
                        equivalent_pose=matched_poses[row_index],
                        procrustes=model.weighted_procrustes,
                        candidate_k=int((active_path_config or {}).get("candidate_k", 16)),
                        projection_chunk_size=int(
                            (active_path_config or {}).get("projection_chunk_size", 256)
                        ),
                    )
                    T_W_from_C = moved["gt"]["T_W_from_C"][row_index]
                    row.update(
                        active_row(
                            active_result,
                            sample_id=str(row["sample_id"]),
                            frame_id=int(row["frame_id"]),
                            T_W_from_C=T_W_from_C,
                            target_leakage_detected=False,
                        )
                    )
                    continue
                exact = closest_points_on_triangle_mesh(q_aux, vertices, faces, point_chunk_size=256)
                q_surface = exact["points"]
                observed = observed_padded[row_index, mask]
                solution = model.weighted_procrustes.solve(
                    q_surface[None], observed[None], q_surface.new_ones((1, len(q_surface))),
                    torch.ones((1, len(q_surface)), dtype=torch.bool, device=device),
                )
                pose = solution["transform"][0]
                target = matched_targets[row_index, mask]
                gt_pose = matched_poses[row_index]
                correspondence = torch.linalg.vector_norm(q_surface - target, dim=-1)
                reconstructed = transform_points(pose[None], q_surface[None])[0]
                alignment = torch.linalg.vector_norm(reconstructed - observed, dim=-1)
                rotation = torch.rad2deg(rotation_geodesic_distance(
                    pose[:3, :3][None], gt_pose[:3, :3][None]
                ))[0]
                translation = torch.linalg.vector_norm(pose[:3, 3] - gt_pose[:3, 3]) * 1000
                corr_p95 = torch.quantile(correspondence.float(), .95) * 1000
                align_p95 = torch.quantile(alignment.float(), .95) * 1000
                score = corr_p95 + align_p95 + rotation + translation
                surface_membership = torch.linalg.vector_norm(
                    q_surface - exact["points"], dim=-1
                )
                passed = bool(
                    (corr_p95 <= 1.0 + 1e-6) & (align_p95 <= 1.0 + 1e-6)
                    & (rotation <= 1.0 + 1e-6) & (translation <= 1.0 + 1e-6)
                    & solution["rank_valid"][0]
                    & (torch.quantile(surface_membership.float(), .95) * 1000 <= .1 + 1e-6)
                )
                row.update(
                    exact_global_projected_correspondence_p95_mm=float(corr_p95),
                    exact_global_projection_alignment_p95_mm=float(align_p95),
                    exact_global_projection_rotation_error_deg=float(rotation),
                    exact_global_projection_translation_error_mm=float(translation),
                    exact_global_projection_rank=int(solution["rank"][0]),
                    exact_global_surface_membership_p95_mm=float(torch.quantile(surface_membership.float(), .95) * 1000),
                    exact_global_projection_score=float(score),
                    exact_global_sample_gate_passed=passed,
                    k16_exact_global_triangle_recall=1.0,
                    k16_fallback_fraction=0.0,
                    k16_sample_gate_passed=passed,
                )
        for row_index, row in enumerate(batch_rows):
            key = (str(row["scene_id"]), int(row["fragment_id"]))
            world_metadata.setdefault(
                key,
                (
                    moved["template_symmetry_metadata"][row_index],
                    moved["gt"]["effective_symmetry_group"][row_index],
                ),
            )
        rows.extend(batch_rows)
        rotation_sum += sum(row["top1_rotation_error_deg"] for row in batch_rows)
        translation_sum += sum(row["translation_total_mm"] for row in batch_rows)
        progress.set_postfix(
            {
                "loss": f"{loss_sums.get('loss_total', 0.0) / max(sample_count, 1):.4f}",
                "rot": f"{rotation_sum / max(sample_count, 1):.1f}deg",
                "trans": f"{translation_sum / max(sample_count, 1):.1f}mm",
            },
            refresh=False,
        )
    raw = aggregate_metric_rows(rows)
    correspondence_summaries = [
        row["correspondence_prediction_summary"]
        for row in rows
        if "correspondence_prediction_summary" in row
    ]
    correspondence_context_distance = (
        float(torch.pdist(torch.as_tensor(correspondence_summaries, dtype=torch.float64)).mean())
        if len(correspondence_summaries) > 1
        else 0.0
    )
    metrics = {
        "eval/symmetry_pose_loss": loss_sums.get("loss_symmetry_pose", 0.0) / max(sample_count, 1),
        "eval/pose_query_classification_loss": loss_sums.get("loss_pose_classification", 0.0) / max(sample_count, 1),
        "eval/top1_rotation_error_deg": raw.get("top1_rotation_error_deg", math.nan),
        "eval/oracle_topK_rotation_error_deg": raw.get("oracle_topk_rotation_error_deg", math.nan),
        "eval/top1_translation_total_mm": raw.get("translation_total_mm", math.nan),
        "eval/oracle_topK_translation_total_mm": raw.get("oracle_translation_total_mm", math.nan),
        "eval/top1_translation_along_axis_mm": raw.get("translation_along_axis_mm", math.nan),
        "eval/top1_translation_perpendicular_to_axis_mm": raw.get("translation_perpendicular_axis_mm", math.nan),
        "eval/top1_pose_success_2deg_2mm": raw.get("success_2deg_2mm", math.nan),
        "eval/top1_pose_success_5deg_5mm": raw.get("success_5deg_5mm", math.nan),
        "eval/top1_pose_success_10deg_10mm": raw.get("success_10deg_10mm", math.nan),
        "eval/oracle_topK_pose_success_5deg_5mm": raw.get("oracle_topk_success_5deg_5mm", math.nan),
        "eval/oracle_topK_pose_success_2deg_2mm": raw.get("oracle_topk_success_2deg_2mm", math.nan),
        "eval/oracle_topK_pose_success_10deg_10mm": raw.get("oracle_topk_success_10deg_10mm", math.nan),
        "eval/query_positive_accuracy": raw.get("query_classification_accuracy", math.nan),
        "eval/oracle_best_pose_cost": raw.get("oracle_best_pose_cost", math.nan),
        "eval/top1_scored_pose_cost": raw.get("top1_scored_pose_cost", math.nan),
        "eval/ranking_regret": raw.get("ranking_regret", math.nan),
        "eval/top1_query_is_oracle": raw.get("top1_query_is_oracle", math.nan),
        "eval/score_pose_cost_spearman": raw.get("score_pose_cost_spearman", math.nan),
        "eval/score_vs_negative_pose_cost_spearman": raw.get("score_vs_negative_pose_cost_spearman", math.nan),
        "eval/pose_cost_min": raw.get("pose_cost_min", math.nan),
        "eval/pose_cost_max": raw.get("pose_cost_max", math.nan),
        "eval/pose_cost_mean": raw.get("pose_cost_mean", math.nan),
        "eval/pose_cost_std": raw.get("pose_cost_std", math.nan),
        "eval/ranking_target_entropy": raw.get("ranking_target_entropy", math.nan),
        "eval/ranking_predicted_entropy": raw.get("ranking_predicted_entropy", math.nan),
        "eval/ranking_target_max_probability": raw.get("ranking_target_max_probability", math.nan),
        "eval/ranking_predicted_max_probability": raw.get("ranking_predicted_max_probability", math.nan),
        "eval/observed_region_point_accuracy": raw.get("observed_region_point_accuracy", math.nan),
        "eval/observed_region_macro_f1": raw.get("observed_region_macro_f1", math.nan),
        "eval/observed_region_ignored_point_count": raw.get("observed_region_ignored_point_count", math.nan),
        "eval/active_region_exact_set_accuracy": raw.get("active_region_exact_match", math.nan),
        "eval/active_region_macro_f1": raw.get("active_region_macro_f1", math.nan),
        "eval/effective_group_accuracy": raw.get("learned_effective_group_accuracy", math.nan),
        "eval/pose_conditioned_effective_group_accuracy": raw.get("pose_conditioned_effective_group_accuracy", math.nan),
        "eval/predicted_hypothesis_count_accuracy": raw.get("pose_conditioned_hypothesis_count_accuracy", math.nan),
        "eval/pose_conditioned_out_of_bounds_fraction": raw.get("pose_conditioned_out_of_bounds_fraction", math.nan),
        "eval/pose_conditioned_group_unresolved": raw.get("pose_conditioned_group_unresolved", math.nan),
        "eval/learned_effective_group_accuracy": raw.get("learned_effective_group_accuracy", math.nan),
        "eval/learned_hypothesis_count_accuracy": raw.get("learned_hypothesis_count_accuracy", math.nan),
        "eval/duplicate_query_fraction": raw.get("duplicate_pose_query_fraction", math.nan),
        "eval/fragment_num_faces": raw.get("fragment_num_faces", math.nan),
        "eval/fragment_surface_area_m2": raw.get("fragment_surface_area_m2", math.nan),
        "eval/fragment_bbox_diagonal_m": raw.get("fragment_bbox_diagonal_m", math.nan),
        "eval/correspondence_point_rmse_mm": raw.get("correspondence_point_rmse_mm", math.nan),
        "eval/correspondence_point_p95_mm": raw.get("correspondence_point_p95_mm", math.nan),
        "eval/correspondence_pose_rotation_error_deg": raw.get("correspondence_pose_rotation_error_deg", math.nan),
        "eval/correspondence_pose_translation_error_mm": raw.get("correspondence_pose_translation_error_mm", math.nan),
        "eval/correspondence_pose_success_2deg_2mm": raw.get("correspondence_pose_success_2deg_2mm", math.nan),
        "eval/correspondence_pose_success_5deg_5mm": raw.get("correspondence_pose_success_5deg_5mm", math.nan),
        "eval/direct_vs_correspondence_rotation_deg": raw.get("direct_vs_correspondence_rotation_deg", math.nan),
        "eval/direct_vs_correspondence_translation_mm": raw.get("direct_vs_correspondence_translation_mm", math.nan),
        "eval/confidence_entropy": raw.get("confidence_entropy", math.nan),
        "eval/effective_correspondence_count": raw.get("effective_correspondence_count", math.nan),
        "eval/maximum_normalized_correspondence_weight": raw.get("maximum_normalized_correspondence_weight", math.nan),
        "eval/correspondence_context_pairwise_distance": correspondence_context_distance,
        "eval/correspondence_pose_rank_valid": raw.get("correspondence_pose_rank_valid", math.nan),
        "eval/correspondence_pose_valid_solution": raw.get("correspondence_pose_valid_solution", math.nan),
        "eval/hybrid_residual_rotation_deg": raw.get("hybrid_residual_rotation_deg", math.nan),
        "eval/hybrid_residual_translation_mm": raw.get("hybrid_residual_translation_mm", math.nan),
    }
    projection_scores = [row["exact_global_projection_score"] for row in rows if "exact_global_projection_score" in row]
    projection_gates = [row["exact_global_sample_gate_passed"] for row in rows if "exact_global_sample_gate_passed" in row]
    if projection_scores:
        metrics["eval/worst_sample_projection_score"] = max(projection_scores)
        metrics["eval/all_samples_projection_gate_passed"] = float(all(projection_gates))
    for name, value in loss_sums.items():
        metrics[f"eval/{name}"] = value / max(sample_count, 1)
    if (
        loss_config.get("joint_correspondence_pose", {}).get("enabled", False)
        or loss_config.get("joint_surface_correspondence_pose_v3", {}).get("enabled", False)
    ):
        for name in (
            "correspondence_rmse_mm", "correspondence_p50_mm",
            "correspondence_p95_mm", "correspondence_max_mm",
            "predicted_to_template_surface_p50_mm", "predicted_to_template_surface_p95_mm",
            "predicted_to_template_surface_max_mm", "template_visible_patch_to_predicted_p95_mm",
            "symmetric_chamfer_p95_mm", "rotation_error_deg",
            "translation_total_mm", "translation_along_axis_mm",
            "translation_perpendicular_to_axis_mm", "visible_alignment_rmse_mm",
            "visible_alignment_p95_mm", "visible_alignment_max_mm",
            "pose_success_1deg_1mm", "pose_success_2deg_2mm",
            "pose_success_5deg_5mm", "effective_correspondence_count",
            "effective_correspondence_fraction", "max_correspondence_weight",
            "procrustes_rank", "procrustes_rank_valid",
            "procrustes_determinant", "procrustes_orthogonality_error",
            "procrustes_reflection_corrected", "physical_normalized_score",
            "coarse_patch_top1_accuracy", "coarse_patch_top4_recall",
            "coarse_patch_top8_recall", "gt_patch_in_candidate_set_fraction",
            "valid_patch_set_top1_accuracy", "valid_patch_set_top4_recall",
            "valid_patch_set_top8_recall",
            "valid_patch_set_in_candidate_set_fraction",
            "mean_valid_patch_count", "max_valid_patch_count",
            "fraction_with_multiple_valid_patches",
            "wrong_top1_but_same_triangle_available_fraction",
            "triangle_top1_accuracy", "gt_triangle_in_local_candidates_fraction",
            "single_owner_triangle_top1", "valid_triangle_set_top1",
            "valid_triangle_set_top4", "valid_triangle_set_top1_accuracy",
            "valid_triangle_set_top4_recall", "mean_valid_triangle_count",
            "fraction_with_multiple_valid_triangles",
            "valid_triangle_candidate_recall", "local_triangle_set_ce",
            "local_triangle_random_ce",
            "local_triangle_classifier_worse_than_uniform",
            "triangle_target_index_mismatch_fraction",
            "covariance_penalty_active", "rank_margin_m2",
            "mean_local_candidate_count", "min_local_candidate_count",
            "max_local_candidate_count", "invalid_candidate_count_fraction",
            "duplicate_local_candidate_fraction",
            "teacher_forcing_selected_symmetry_element",
            "barycentric_reconstruction_p50_mm", "barycentric_reconstruction_p95_mm",
            "unique_predicted_patches", "unique_predicted_triangles",
            "most_popular_patch_fraction", "most_popular_triangle_fraction",
            "correspondence_rank", "rank_invalid_fraction",
            "candidate_recall", "aux_coordinate_rmse_mm",
            "aux_coordinate_p95_mm", "fine_feature_variance",
            "fine_feature_effective_rank", "fine_feature_pairwise_distance",
            "fine_feature_collision_fraction", "fine_candidate_logit_variance",
        ):
            metrics[f"eval/{name}"] = raw.get(name, math.nan)
        metrics.update(
            {
                "eval/all_samples_pose_success_2deg_2mm": float(all(bool(row["pose_success_2deg_2mm"]) for row in rows)),
                "eval/all_samples_correspondence_p95_under_2mm": float(all(float(row["correspondence_p95_mm"]) <= 2.0 for row in rows)),
                "eval/all_samples_alignment_p95_under_2mm": float(all(float(row["visible_alignment_p95_mm"]) <= 2.0 for row in rows)),
                "eval/all_samples_predicted_to_template_surface_p95_under_1mm": float(all(float(row["predicted_to_template_surface_p95_mm"]) <= 1.0 for row in rows)),
                "eval/all_samples_procrustes_rank_valid": float(all(bool(row["procrustes_rank_valid"]) for row in rows)),
            }
        )
        selected_counts: dict[int, int] = {}
        for row in rows:
            selected = int(row["selected_shared_symmetry_element"])
            selected_counts[selected] = selected_counts.get(selected, 0) + 1
        for selected, count in selected_counts.items():
            metrics[f"eval/selected_symmetry_element_{selected}_fraction"] = count / max(len(rows), 1)
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(
            (str(row["scene_id"]), int(row["fragment_id"])), []
        ).append(row)
    consistency_values: dict[str, list[float]] = {}
    for key, group_rows in grouped_rows.items():
        metadata, effective_group = world_metadata[key]
        for source, prefix in (
            ("top1_T_W_from_O", ""),
            ("oracle_T_W_from_O", "oracle_"),
            ("correspondence_T_W_from_O", "correspondence_"),
        ):
            if not all(source in row for row in group_rows):
                continue
            transforms = torch.as_tensor(
                [row[source] for row in group_rows], dtype=torch.float64
            )
            values = world_pose_consistency(transforms, metadata, effective_group)
            for name, value in values.items():
                consistency_values.setdefault(prefix + name, []).append(value)
    for name, values in consistency_values.items():
        metrics[f"eval/{name}"] = sum(values) / len(values)
    if "eval/correspondence_world_translation_range_mm" in metrics:
        metrics["eval/correspondence_world_translation_spread_mm"] = metrics[
            "eval/correspondence_world_translation_range_mm"
        ]
    for name, value in context_conditioning_metrics(rows).items():
        metrics[f"eval/{name}"] = value
    for key, value in raw.items():
        if key.startswith("observed_region_") or key.startswith("active_region_region_"):
            metrics.setdefault(f"eval/{key}", value)
    if active_path_enabled:
        active_world: dict[str, float] = {}
        for key, group_rows in grouped_rows.items():
            metadata, effective_group = world_metadata[key]
            if all("exact_global_T_W_from_O" in row for row in group_rows):
                for name, value in active_world_metrics(
                    group_rows, metadata, effective_group
                ).items():
                    active_world[name] = value
        clean: dict[str, float] = {
            "eval/active/exact_global/worst_sample_projection_score": (
                worst_sample_projection_score(rows)
            ),
            "eval/active/exact_global/all_samples_gate_passed": float(
                all(bool(row["exact_global_sample_gate_passed"]) for row in rows)
            ),
            "eval/active/exact_global/worst_correspondence_p95_mm": max(
                float(row["exact_global_projected_correspondence_p95_mm"])
                for row in rows
            ),
            "eval/active/exact_global/worst_alignment_p95_mm": max(
                float(row["exact_global_projection_alignment_p95_mm"])
                for row in rows
            ),
            "eval/active/worst_sample_practical_score": (
                worst_sample_practical_score(rows)
            ),
            "eval/active/practical_passed_sample_count": float(
                sum(practical_sample_gate(row)["passed"] for row in rows)
            ),
            "eval/active/strict_passed_sample_count": float(
                sum(bool(row["exact_global_sample_gate_passed"]) for row in rows)
            ),
            "eval/active/k16/minimum_exact_global_triangle_recall": min(
                float(row["k16_exact_global_triangle_recall"]) for row in rows
            ),
            "eval/active/k16/maximum_fallback_fraction": max(
                float(row["k16_fallback_fraction"]) for row in rows
            ),
            "eval/active/nonfinite_detected": float(
                any(bool(row["active_nonfinite_detected"]) for row in rows)
            ),
        }
        if bool((active_path_config or {}).get("ten_view_gates", False)):
            clean.update(
                **{
                    "eval/active/worst_sample_score": worst_ten_view_sample_score(rows),
                    "eval/active/pose_ready_sample_count": float(sum(
                        pose_placement_sample_gate(row)["passed"] for row in rows
                    )),
                    "eval/active/practical_surface_passed_sample_count": float(sum(
                        practical_surface_sample_gate(row)["passed"] for row in rows
                    )),
                    "eval/active/strict_surface_passed_sample_count": float(sum(
                        strict_surface_sample_gate(row)["passed"] for row in rows
                    )),
                }
            )
        elif bool((active_path_config or {}).get("multifragment_gates", False)):
            multifragment_gates = multifragment_stage_gates(rows)
            clean.update(
                **{
                    "eval/active/worst_sample_multifragment_score": worst_multifragment_sample_score(rows),
                    "eval/active/pose_ready_sample_count": float(
                        multifragment_gates["pose_placement_gate"]["passed_sample_count"]
                    ),
                    "eval/active/practical_surface_passed_sample_count": float(
                        multifragment_gates["practical_surface_gate"]["passed_sample_count"]
                    ),
                    "eval/active/strict_surface_passed_sample_count": float(
                        multifragment_gates["strict_surface_gate"]["passed_sample_count"]
                    ),
                }
            )
        else:
            clean.update({
                "eval/inactive/legacy_triangle/active": 0.0,
                "eval/inactive/legacy_barycentric/active": 0.0,
                "eval/inactive/legacy_pose_query/active": 0.0,
                "eval/inactive/regions/active": 0.0,
                "eval/inactive/ranking/active": 0.0,
            })
        for name, value in active_world.items():
            clean[f"eval/active/world/{name}"] = float(value)
        active_loss_names = (
            "loss_total",
            "loss_fine_coordinate_aux_normalized",
            "loss_fine_coordinate_aux_tail_normalized",
            "loss_raw_aux_rotation_normalized",
            "loss_raw_aux_translation_normalized",
            "loss_raw_aux_alignment_normalized",
            "weighted_loss_fine_coordinate_aux",
            "weighted_loss_fine_coordinate_aux_tail",
            "weighted_loss_raw_aux_rotation",
            "weighted_loss_raw_aux_translation",
            "weighted_loss_raw_aux_alignment",
        )
        for name in active_loss_names:
            if name in loss_sums:
                clean[f"eval/active/loss/{name}"] = (
                    loss_sums[name] / max(sample_count, 1)
                )
        # Compatibility aliases remain finite and point to the same active
        # exact-global result; no inactive metric participates in selection.
        clean["eval/active/worst_sample_projection_score"] = clean[
            "eval/active/exact_global/worst_sample_projection_score"
        ]
        clean["eval/active/all_samples_gate_passed"] = clean[
            "eval/active/exact_global/all_samples_gate_passed"
        ]
        clean["eval/worst_sample_projection_score"] = clean[
            "eval/active/exact_global/worst_sample_projection_score"
        ]
        clean["eval/all_samples_projection_gate_passed"] = clean[
            "eval/active/exact_global/all_samples_gate_passed"
        ]
        metrics = clean
    if clean_active_only:
        rows = [clean_active_metric_row(row) for row in rows]
    if was_training:
        model.train()
    return metrics, rows


def _write_evaluation(
    run_dir: Path,
    epoch: int,
    metrics: Mapping[str, Any],
    rows: list[dict[str, Any]],
    *,
    clean_active_only: bool = False,
) -> Path:
    destination = run_dir / "evaluations" / f"epoch_{epoch:04d}"
    destination.mkdir(parents=True, exist_ok=False)
    _atomic_json(destination / "metrics.json", {**WARNING_FLAGS, "epoch": epoch, **metrics})
    diagnostics = context_conditioning_diagnostics(rows)
    if diagnostics:
        _atomic_json(
            destination / "context_conditioning_diagnostics.json",
            {**WARNING_FLAGS, "epoch": epoch, **diagnostics},
        )
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (destination / "per_sample_metrics.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if rows and "query_pose_costs" in rows[0]:
        query_count = len(rows[0]["query_pose_costs"])
        matrix_fields = [
            "sample_id",
            "frame_id",
            "oracle_query_index",
            *(f"query_{index}_pose_cost" for index in range(query_count)),
        ]
        matrix_rows = []
        current_assignments: dict[int, int] = {}
        for row in sorted(rows, key=lambda value: int(value["frame_id"])):
            frame = int(row["frame_id"])
            assigned = int(row["oracle_query_index"])
            current_assignments[frame] = assigned
            matrix_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "frame_id": frame,
                    "oracle_query_index": assigned,
                    **{
                        f"query_{index}_pose_cost": float(cost)
                        for index, cost in enumerate(row["query_pose_costs"])
                    },
                }
            )
        for matrix_path in (
            destination / "query_assignment_matrix.csv",
            run_dir / "query_assignment_matrix.csv",
        ):
            temporary = matrix_path.with_suffix(matrix_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=matrix_fields)
                writer.writeheader()
                writer.writerows(matrix_rows)
            os.replace(temporary, matrix_path)
        history_assignments: list[dict[int, int]] = []
        for previous_path in sorted(
            (run_dir / "evaluations").glob(
                "epoch_*/query_assignment_diagnostics.json"
            )
        ):
            payload = json.loads(previous_path.read_text(encoding="utf-8"))
            history_assignments.append(
                {
                    int(frame): int(query)
                    for frame, query in payload.get("assignments", {}).items()
                }
            )
        history_assignments.append(current_assignments)
        switch_rate, switch_count, comparison_count = assignment_switch_rate(
            history_assignments
        )
        assignment_payload = {
            **WARNING_FLAGS,
            "epoch": epoch,
            "assignments": {
                str(frame): query for frame, query in current_assignments.items()
            },
            **query_assignment_summary(rows, num_queries=query_count),
            "query_assignment_switch_rate": switch_rate,
            "query_assignment_switch_count": switch_count,
            "query_assignment_comparison_count": comparison_count,
            "query_world_pose_consistency": query_world_consistency(rows),
        }
        _atomic_json(
            destination / "query_assignment_diagnostics.json", assignment_payload
        )
        _atomic_json(run_dir / "query_assignment_diagnostics.json", assignment_payload)
    snapshots = {
        "oracle_pose_metrics.json": {
            key: value
            for key, value in metrics.items()
            if "oracle" in key
        },
        "ranking_diagnostics.json": {
            key: value
            for key, value in metrics.items()
            if any(
                token in key
                for token in (
                    "pose_cost_",
                    "ranking_",
                    "top1_query_is_oracle",
                    "spearman",
                )
            )
        },
        "ranking_target_statistics.json": {
            "target_distributions": [
                row["ranking_target_distribution"]
                for row in rows
                if "ranking_target_distribution" in row
            ],
            "nearly_uniform_target_sample_ids": [
                row["sample_id"]
                for row in rows
                if "ranking_target_distribution" in row
                and row["ranking_target_distribution"]
                and max(row["ranking_target_distribution"])
                - min(row["ranking_target_distribution"])
                < 0.05
            ],
        },
        "region_metrics.json": {
            key: value
            for key, value in metrics.items()
            if "region" in key or "effective_group" in key
        },
        "cross_view_consistency.json": {
            key: value for key, value in metrics.items() if "world_" in key
        },
        "top1_vs_oracle_summary.json": {
            key: value
            for key, value in metrics.items()
            if "top1" in key or "oracle" in key or "regret" in key
        },
        "effective_group_metrics.json": {
            key: value for key, value in metrics.items() if "group" in key
        },
    }
    if clean_active_only:
        snapshots = {
            "cross_view_consistency.json": snapshots["cross_view_consistency.json"],
            "active_conditioning_summary.json": {
                key: value
                for key, value in metrics.items()
                if "conditioning" in key or "context" in key or "feature" in key
            },
        }
    confusion = {
        key: sum(float(row.get(key, 0)) for row in rows)
        for key in sorted({key for row in rows for key in row})
        if "confusion" in key
        and key != "patch_confusion_matrix"
        and all(
            isinstance(row.get(key, 0), (bool, int, float)) for row in rows
        )
    }
    if not clean_active_only:
        snapshots["region_confusion_matrix.json"] = confusion
        snapshots["patch_confusion_matrix.json"] = {
            str(row.get("sample_id", index)): row["patch_confusion_matrix"]
            for index, row in enumerate(rows)
            if "patch_confusion_matrix" in row
        }
    for filename, values in snapshots.items():
        payload = {**WARNING_FLAGS, "epoch": epoch, "metrics": values}
        _atomic_json(destination / filename, payload)
        _atomic_json(run_dir / filename, payload)
    return destination


def _write_conditioning_gradient_audit(
    run_dir: Path,
    *,
    label: str,
    source_epoch: int,
    gradient_norm: float,
    module_gradient_norms: Mapping[str, float],
    nonzero_gradient_parameter_fraction: float,
    counters: TrainingCounters,
    status: str,
) -> Path:
    """Snapshot input-conditioning and the most recent real backward audit."""

    source = (
        run_dir / "evaluations" / f"epoch_{int(source_epoch):04d}"
        / "context_conditioning_diagnostics.json"
    )
    conditioning = (
        json.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}
    )
    destination = run_dir / "conditioning_gradient_audits" / str(label)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
            **WARNING_FLAGS,
            "label": str(label),
            "source_epoch": int(source_epoch),
            "gradient_status": str(status),
            "gradient_norm": float(gradient_norm),
            "module_gradient_norms": dict(module_gradient_norms),
            "nonzero_gradient_parameter_fraction": float(
                nonzero_gradient_parameter_fraction
            ),
            "conditioning": conditioning,
            **counters.to_dict(),
        }
    _atomic_json(destination / "audit.json", payload)
    _atomic_json(destination / "fragment_conditioning_audit.json", payload)
    matrix = conditioning.get("q_aux_summary_pairwise_distance_matrix")
    sample_csv = source.parent / "per_sample_metrics.csv"
    if isinstance(matrix, list) and sample_csv.is_file():
        with sample_csv.open("r", encoding="utf-8", newline="") as stream:
            identities = list(csv.DictReader(stream))
        pairwise_path = destination / "fragment_conditioning_pairwise.csv"
        fields = ["sample_a", "sample_b", "fragment_a", "fragment_b", "frame_a", "frame_b", "q_aux_summary_distance"]
        rows = []
        for left in range(len(identities)):
            for right in range(left + 1, len(identities)):
                rows.append({
                        "sample_a": identities[left]["sample_id"], "sample_b": identities[right]["sample_id"],
                        "fragment_a": identities[left].get("fragment_id"), "fragment_b": identities[right].get("fragment_id"),
                        "frame_a": identities[left].get("frame_id"), "frame_b": identities[right].get("frame_id"),
                        "q_aux_summary_distance": matrix[left][right],
                })
        _atomic_csv_rows(pairwise_path, fields, rows)
    return destination


def _is_improvement(
    current: float,
    best: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    if not math.isfinite(current):
        return False
    if best is None:
        return True
    if mode == "min":
        return current <= best - min_delta
    if mode == "max":
        return current >= best + min_delta
    raise ValueError("best_metric_mode must be min or max")


def is_checkpoint_improvement(
    metrics: Mapping[str, float],
    best_metrics: Mapping[str, float] | None,
    *,
    metric_name: str,
    mode: str,
    min_delta: float,
    tie_breaker_name: str | None = None,
    tie_breaker_mode: str = "max",
    tie_breakers: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Apply the primary metric, then lexicographic configured tie breakers."""

    current = float(metrics[metric_name])
    if best_metrics is None:
        return math.isfinite(current)
    best = float(best_metrics[metric_name])
    if _is_improvement(current, best, mode=mode, min_delta=min_delta):
        return True
    if not math.isfinite(current) or abs(current - best) > min_delta:
        return False
    configured = list(tie_breakers or ())
    if not configured and tie_breaker_name is not None:
        configured = [{"metric": tie_breaker_name, "mode": tie_breaker_mode}]
    for item in configured:
        name = str(item["metric"])
        tie_mode = str(item.get("mode", "max"))
        current_tie = float(metrics[name])
        best_tie = float(best_metrics[name])
        if not math.isfinite(current_tie):
            return False
        if abs(current_tie - best_tie) <= min_delta:
            continue
        if tie_mode == "max":
            return current_tie > best_tie
        if tie_mode == "min":
            return current_tie < best_tie
        raise ValueError("best metric tie-breaker mode must be min or max")
    return False


def _save_best_checkpoint(
    run_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    epoch: int,
    counters: TrainingCounters,
    checkpoint_filename: str,
    best_metric_name: str,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    environment: Mapping[str, Any],
    sample_exposures: Mapping[str, int],
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / checkpoint_filename
    temporary = checkpoint_dir / (checkpoint_filename + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": scaler.state_dict(),
        "epoch": epoch,
        **counters.to_dict(),
        "best_metric_name": best_metric_name,
        "best_metric": float(metrics[best_metric_name]),
        "resolved_config": dict(config),
        "manifest_sha256": manifest_file_sha256,
        "template_sha256": manifest["template_sha256"],
        "sidecar_sha256": manifest["symmetry_sidecar_sha256"],
        "environment": dict(environment),
        "sample_exposures": {
            str(sample_id): int(exposures)
            for sample_id, exposures in sample_exposures.items()
        },
        **WARNING_FLAGS,
    }
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    best_metrics = {
        **WARNING_FLAGS,
        "epoch": epoch,
        **counters.to_dict(),
        "best_metric_name": best_metric_name,
        "best_metric": float(metrics[best_metric_name]),
        **{
            key: metrics[key]
            for key in (
                "eval/top1_scored_pose_cost",
                "eval/oracle_best_pose_cost",
                "eval/ranking_regret",
                "eval/top1_pose_success_5deg_5mm",
                "eval/oracle_topK_pose_success_5deg_5mm",
                "eval/active_region_exact_set_accuracy",
                "eval/effective_group_accuracy",
            )
            if key in metrics
        },
        "metrics": dict(metrics),
    }
    best_manifest = {
        **WARNING_FLAGS,
        "checkpoint_path": str(destination),
        "checkpoint_policy": "best_only_atomic_replace_on_configured_metric_improvement",
        "manifest_sha256": manifest_file_sha256,
        "manifest_internal_sha256": manifest["manifest_sha256"],
        "template_sha256": manifest["template_sha256"],
        "symmetry_sidecar_sha256": manifest["symmetry_sidecar_sha256"],
        "epoch": epoch,
        **counters.to_dict(),
        "best_metric_name": best_metric_name,
        "best_metric": float(metrics[best_metric_name]),
        "strict_load_status": "saved_for_strict_load",
        "environment": dict(environment),
    }
    _atomic_json(checkpoint_dir / "best_metrics.json", best_metrics)
    _atomic_json(checkpoint_dir / "best_manifest.json", best_manifest)
    return destination


def run_overfit_training(
    config: Mapping[str, Any],
    *,
    device_name: str = "auto",
    work_dir_override: str | Path | None = None,
    resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    init_modules: list[str] | None = None,
    from_scratch: bool = False,
) -> dict[str, Any]:
    config = deepcopy(dict(config))
    dependencies = dict(config.get("stage_gate_dependencies", {}))
    if bool(dependencies.get("require_parameterization_capacity", False)):
        configured = dependencies.get("parameterization_capacity_path")
        paths = configured if isinstance(configured, list) else [configured]
        required_field = str(
            dependencies.get(
                "parameterization_capacity_required_field", "audit_passed"
            )
        )
        failures = []
        for raw_path in paths:
            path = Path(str(raw_path)).expanduser() if raw_path else None
            payload = None
            if path is not None and path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
            if payload is None or not bool(payload.get(required_field, False)):
                failures.append(
                    {
                        "path": None if path is None else str(path),
                        "required_field": required_field,
                        "actual": None if payload is None else payload.get(required_field),
                    }
                )
        if failures:
            raise ValueError(
                "training stopped before epoch 0: parameterization capacity gate "
                f"is missing or failed: {failures}"
            )
    leakage_policy = dict(config.get("target_leakage_policy", {}))
    audit_path = leakage_policy.get("audit_path")
    joint_baseline = bool(
        config.get("loss", {}).get("joint_correspondence_pose", {}).get("enabled", False)
        or config.get("loss", {}).get("joint_surface_correspondence_pose_v3", {}).get("enabled", False)
    )
    if joint_baseline and not audit_path:
        raise ValueError(
            "joint baseline training requires target_leakage_policy.audit_path "
            "from a passing pre-training audit"
        )
    if audit_path:
        audit = json.loads(Path(str(audit_path)).expanduser().read_text(encoding="utf-8"))
        if bool(audit.get("target_leakage_detected", False)):
            raise ValueError("training forbidden: target leakage audit detected leakage")
        if joint_baseline and not bool(audit.get("audit_passed", False)):
            raise ValueError("joint baseline training forbidden: target leakage audit did not pass")
    validate_data_policy(config)
    validate_overfit_flags(config)
    validate_single_fragment_config(config)
    validate_multifragment_config(config)
    validate_initialization_request(
        config,
        from_scratch=from_scratch,
        resume=resume,
        init_checkpoint=init_checkpoint,
        init_modules=init_modules,
    )
    train_cfg = config["train"]
    if not bool(train_cfg.get("save_best_only", True)):
        raise ValueError("faces840 trainer requires train.save_best_only=True")
    if bool(train_cfg.get("save_periodic_checkpoints", False)) or bool(
        train_cfg.get("save_final_checkpoint", False)
    ):
        raise ValueError("faces840 trainer forbids periodic/final checkpoints")
    device = resolve_device(device_name)
    if device.type == "cuda":
        # Required by deterministic cuBLAS kernels when deterministic algorithms
        # are enabled by seed_everything(). Record it in the resolved environment.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    register_all_modules()
    experiment = config["experiment"]
    output_root = Path(
        work_dir_override or experiment["work_dir_root"]
    ).expanduser().resolve()
    run_dir = _unique_run_directory(output_root, str(experiment["name"]))
    run_id = run_dir.name
    history = TrainingHistory(run_dir, run_id, config.get("history", {}))
    environment = _environment(device)
    counters = TrainingCounters()
    global_step = 0
    best_value: float | None = None
    best_epoch: int | None = None
    best_checkpoint: Path | None = None
    best_metrics_snapshot: dict[str, float] | None = None
    try:
        config["dataset"]["fragment_mesh_cache_dir"] = str(
            run_dir / "cache" / "fragment_mesh_metadata"
        )
        dataset = _build_dataset(config)
        manifest_path = Path(str(config["data"]["train_manifest"])).expanduser()
        manifest, manifest_file_sha = load_faces840_manifest(
            manifest_path, config, dataset
        )
        record_indices = {
            record.sample_id: index for index, record in enumerate(dataset.sample_records)
        }
        data_cfg = config["data"]
        selected_manifest_samples = _select_manifest_samples_by_scene(
            manifest["samples"], data_cfg.get("scene_ids")
        )
        expected_selected = data_cfg.get("expected_selected_samples")
        if (
            expected_selected is not None
            and len(selected_manifest_samples) != int(expected_selected)
        ):
            raise ValueError(
                "scene-filtered manifest sample count mismatch: "
                f"expected {int(expected_selected)}, "
                f"got {len(selected_manifest_samples)}"
            )
        all_indices = [
            record_indices[sample["sample_id"]]
            for sample in selected_manifest_samples
        ]
        max_train = data_cfg.get("max_train_samples")
        max_validation = data_cfg.get("max_validation_samples")
        train_indices = all_indices[: int(max_train)] if max_train is not None else all_indices
        train_manifest_samples = selected_manifest_samples[: len(train_indices)]
        validation_indices = (
            all_indices[: int(max_validation)] if max_validation is not None else all_indices
        )
        if not train_indices or not validation_indices:
            raise ValueError("train and validation subsets must be non-empty")
        collate = build_from_cfg(config["collate"], COLLATE_FUNCTIONS)
        workers = int(data_cfg.get("num_workers", 4))
        generator = torch.Generator().manual_seed(seed)
        common_loader = {
            "num_workers": workers,
            "persistent_workers": bool(data_cfg.get("persistent_workers", True)) and workers > 0,
            "pin_memory": bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
            "collate_fn": collate,
        }
        drop_last = bool(data_cfg.get("drop_last", False))
        multi_view_cfg = dict(config.get("multi_view_batch", {}))
        grouped_sampler = None
        if bool(multi_view_cfg.get("enabled", False)):
            grouped_sampler = MultiViewBatchSampler(
                train_manifest_samples,
                views_per_group=int(multi_view_cfg.get("views_per_group", len(train_indices))),
                group_by=tuple(multi_view_cfg.get("group_by", ("scene_id", "fragment_id"))),
                require_same_fragment_mesh=bool(
                    multi_view_cfg.get("require_same_fragment_mesh", True)
                ),
                shuffle=bool(data_cfg.get("shuffle_train", True)),
                drop_last=drop_last,
                seed=seed,
            )
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_sampler=grouped_sampler,
                **common_loader,
            )
            budget_batch_size = int(multi_view_cfg.get("views_per_group", len(train_indices)))
        else:
            budget_batch_size = int(data_cfg.get("train_batch_size", 2))
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=budget_batch_size,
                shuffle=bool(data_cfg.get("shuffle_train", True)),
                generator=generator,
                drop_last=drop_last,
                **common_loader,
            )
        validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=int(data_cfg.get("validation_batch_size", 2)),
            shuffle=bool(data_cfg.get("shuffle_validation", False)),
            drop_last=False,
            **common_loader,
        )
        if bool(data_cfg.get("shuffle_validation", False)):
            raise ValueError("validation DataLoader must not shuffle")
        accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
        training_budget = resolve_training_budget(
            config.get("train_budget"),
            selected_samples=len(train_indices),
            batch_size=budget_batch_size,
            gradient_accumulation_steps=accumulation,
            drop_last=drop_last,
            configured_max_optimizer_steps=(
                int(train_cfg["max_optimizer_steps"])
                if train_cfg.get("max_optimizer_steps") is not None
                else None
            ),
            configured_max_epochs=int(train_cfg["max_epochs"]),
        )
        max_optimizer_steps = training_budget.computed_max_optimizer_steps
        effective_views_per_optimizer_step = min(
            len(train_indices), budget_batch_size * accumulation
        )
        expected_effective_views = data_cfg.get(
            "effective_views_per_optimizer_step"
        )
        if (
            expected_effective_views is not None
            and effective_views_per_optimizer_step != int(expected_effective_views)
        ):
            raise ValueError(
                "batch/accumulation contract mismatch: expected "
                f"{int(expected_effective_views)} views per optimizer step, got "
                f"{effective_views_per_optimizer_step}"
            )
        train_cfg["max_optimizer_steps"] = max_optimizer_steps
        train_cfg["max_epochs"] = max(
            int(train_cfg["max_epochs"]),
            math.ceil(max_optimizer_steps / max(len(train_loader), 1)),
        )
        sample_exposures = {
            str(sample["sample_id"]): 0 for sample in train_manifest_samples
        }
        region_report: dict[str, Any] | None = None
        if bool(data_cfg.get("single_fragment_contract", False)) and not bool(
            config.get("active_coordinate_path", {}).get(
                "clean_active_only", False
            )
        ):
            point_loss_config = config["loss"].get("observed_region_loss", {})
            region_report = region_class_distribution(
                (dataset[index] for index in all_indices),
                max_class_weight=float(
                    point_loss_config.get("max_class_weight", 5.0)
                ),
            )
            region_ids = region_report["region_ids"]
            point_loss_cfg = config["loss"].setdefault(
                "observed_region_loss", {}
            )
            if (
                str(point_loss_cfg.get("class_balancing", "none"))
                == "inverse_sqrt_frequency"
            ):
                point_loss_cfg["class_weights"] = [
                    region_report["inverse_sqrt_frequency_weights"][name]
                    for name in region_ids
                ]
            active_loss_cfg = config["loss"].setdefault(
                "active_region_loss", {}
            )
            if str(active_loss_cfg.get("pos_weight_source", "")) == "manifest":
                active_loss_cfg["pos_weight"] = [
                    region_report["active_pos_weight"][name]
                    for name in region_ids
                ]
            _atomic_json(
                run_dir / "region_class_distribution.json",
                {**WARNING_FLAGS, **region_report},
            )
            with (run_dir / "region_class_distribution.csv").open(
                "x", encoding="utf-8", newline=""
            ) as stream:
                fields = [
                    "region_id",
                    "point_frequency",
                    "class_weight",
                    "active_positive_samples",
                    "active_valid_samples",
                    "active_pos_weight",
                ]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for name in region_ids:
                    writer.writerow(
                        {
                            "region_id": name,
                            "point_frequency": region_report["point_frequency"][name],
                            "class_weight": region_report["inverse_sqrt_frequency_weights"][name],
                            "active_positive_samples": region_report["active_positive_samples"][name],
                            "active_valid_samples": region_report["active_valid_samples"][name],
                            "active_pos_weight": region_report["active_pos_weight"][name],
                        }
                    )
        model = build_model(config["model"]).to(device)
        runtime_static_cache = getattr(model, "_static_geometry_cache", None)
        if runtime_static_cache is not None:
            runtime_static_cache.manifest_sha256 = str(
                manifest.get("manifest_sha256", manifest_file_sha)
            )
        resume_path: Path | None = None
        resumed: Mapping[str, Any] | None = None
        initialization_report: dict[str, Any] = {
            "mode": "random_initialization",
            "loaded_module_prefixes": [],
            "missing_keys": [],
            "unexpected_keys": [],
        }
        if from_scratch:
            initialization_report = {
                "mode": "scratch",
                "initialization_mode": "scratch",
                "pretrained_checkpoint": None,
                "checkpoint_sources": [],
                "loaded_module_prefixes": [],
                "optimizer_loaded": False,
                "scheduler_loaded": False,
                "scaler_loaded": False,
                "counters_loaded": False,
                "seed": seed,
                "initial_state_dict_sha256": state_dict_sha256(model),
                "rules": {
                    "linear": "xavier_uniform; bias_zero",
                    "conv1d": "kaiming_uniform_relu; bias_zero",
                    "layer_norm": "weight_one; bias_zero",
                    "final_q_aux": "normal_std_1e-3; bias_zero; tanh",
                },
            }
        if resume is not None:
            # The frozen-feature cache is captured below, before the optimizer
            # exists.  Restore the model now so cached upstream tensors come
            # from the resumed model rather than from random initialization.
            # Optimizer/scheduler/scaler/counters are restored after those
            # objects have been constructed.
            resume_path = Path(resume).expanduser().resolve()
            resumed = torch.load(
                resume_path, map_location=device, weights_only=False
            )
            model.load_state_dict(resumed["model"], strict=True)
        elif init_checkpoint is not None:
            initialization_report = load_model_initialization(
                model,
                init_checkpoint,
                module_prefixes=init_modules,
                strict=bool(
                    config.get("stage", {}).get("strict_initialization", True)
                ),
            )
        stage_cfg = dict(config.get("stage", {}))
        stage_name = str(stage_cfg.get("name", "faces840_joint"))
        freeze_report = apply_trainable_prefixes(
            model, stage_cfg.get("trainable_module_prefixes")
        )
        frozen_cache_cfg = dict(config.get("frozen_feature_cache", {}))
        if bool(frozen_cache_cfg.get("enabled", False)):
            fine_only = (
                "fine_feature_adapter", "fine_coordinate_auxiliary_head"
            )
            upstream_trainable = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and not any(token in name for token in fine_only)
            ]
            if upstream_trainable:
                raise ValueError(
                    "frozen_feature_cache.enabled must be false when upstream modules are trainable"
                )
        frozen_cache_payload: dict[str, Any] | None = None
        frozen_cache_sample_ids: tuple[str, ...] | None = None
        frozen_cache_runtime = {
            "requested": bool(frozen_cache_cfg.get("enabled", False)),
            "mode": "online",
            "audit_path": frozen_cache_cfg.get("audit_path"),
            "fallback_to_online": bool(
                frozen_cache_cfg.get("fallback_to_online", True)
            ),
        }
        if frozen_cache_runtime["requested"]:
            audit_path = frozen_cache_cfg.get("audit_path")
            audit = None
            if audit_path and Path(str(audit_path)).expanduser().is_file():
                audit = json.loads(
                    Path(str(audit_path)).expanduser().read_text(encoding="utf-8")
                )
            cache_allowed = bool(audit and audit.get("cache_allowed", False))
            provenance_verified = True
            if bool(frozen_cache_cfg.get("verify_provenance", False)):
                payload = dict((audit or {}).get("cache_key_payload", {}))
                checkpoint_source = init_checkpoint or resume
                provenance_verified = bool(
                    checkpoint_source
                    and payload.get("initialization_checkpoint_sha256")
                    == sha256_file(Path(checkpoint_source).expanduser().resolve())
                    and payload.get("manifest_file_sha256") == manifest_file_sha
                    and payload.get("template_sha256") == manifest["template_sha256"]
                    and payload.get("sidecar_sha256")
                    == manifest["symmetry_sidecar_sha256"]
                    and payload.get("frozen_module_state_sha256")
                    == frozen_module_state_sha256(model)
                )
                cache_allowed = cache_allowed and provenance_verified
            deterministic_batch = (
                len(train_loader) == 1
                and not bool(data_cfg.get("shuffle_train", True))
                and not bool(config.get("augmentations", {}).get("enabled", False))
            )
            if cache_allowed and deterministic_batch:
                cache_batch = next(iter(train_loader))
                frozen_cache_sample_ids = tuple(map(str, cache_batch["sample_id"]))
                moved_cache_batch = move_to_device(cache_batch, device)
                was_training_for_cache = model.training
                model.eval()
                with torch.no_grad():
                    captured_prediction, captured = capture_fine_adapter_inputs(
                        model, moved_cache_batch
                    )
                frozen_cache_payload = {
                    name: value.detach()
                    for name, value in captured.items()
                    if isinstance(value, torch.Tensor)
                }
                if was_training_for_cache:
                    model.train()
                # The online capture transiently materializes all frozen
                # activations for the full eight-view batch.  Only the small
                # adapter-input payload above survives; release allocator
                # reservations before the independent online evaluator runs.
                del moved_cache_batch, captured, captured_prediction
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                frozen_cache_runtime.update(
                    mode="cached_fine_coordinate_active_loss",
                    cache_allowed=True,
                    audit_cache_key=audit.get("cache_key"),
                    provenance_verified=provenance_verified,
                    sample_ids=list(frozen_cache_sample_ids),
                )
            else:
                frozen_cache_runtime.update(
                    cache_allowed=False,
                    reason=(
                        "audit_missing_or_failed"
                        if not bool(audit and audit.get("cache_allowed", False))
                        else "cache_provenance_mismatch"
                        if not provenance_verified
                        else "training_batch_is_not_single_deterministic_batch"
                    ),
                )
                if not frozen_cache_runtime["fallback_to_online"]:
                    raise ValueError(
                        "frozen feature cache requested but its audit/batch contract failed"
                    )
        config["resolved_frozen_feature_cache"] = frozen_cache_runtime
        criterion = _build_pose_criterion(config)
        optimizer_cfg = train_cfg["optimizer"]
        if optimizer_cfg.get("type") != "AdamW":
            raise ValueError("faces840 trainer supports AdamW only")
        parameter_groups = build_selective_optimizer_parameter_groups(
            model,
            default_lr=float(optimizer_cfg["lr"]),
            prefix_learning_rates=stage_cfg.get("prefix_learning_rates"),
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            lr=float(optimizer_cfg["lr"]),
            weight_decay=float(optimizer_cfg["weight_decay"]),
        )
        max_epochs = int(train_cfg["max_epochs"])
        scheduler = _build_scheduler(optimizer, train_cfg["scheduler"], max_epochs)
        amp_enabled, amp_dtype, amp_dtype_name = _amp_settings(device, train_cfg)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16
        )
        start_epoch = 1
        if resume is not None:
            if resume_path is None or resumed is None:
                raise RuntimeError("resume checkpoint was not loaded before cache capture")
            optimizer.load_state_dict(resumed["optimizer"])
            scheduler.load_state_dict(resumed["scheduler"])
            scaler_state = resumed.get("amp_scaler")
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            counters = TrainingCounters.from_checkpoint(resumed)
            global_step = counters.optimizer_step
            resumed_epoch = int(resumed.get("epoch", 0))
            start_epoch = resumed_epoch + 1
            saved_sample_exposures = resumed.get("sample_exposures")
            if isinstance(saved_sample_exposures, Mapping):
                if set(saved_sample_exposures) != set(sample_exposures):
                    raise ValueError(
                        "resume checkpoint sample_exposures do not match the "
                        "selected training samples"
                    )
                sample_exposures = {
                    sample_id: int(saved_sample_exposures[sample_id])
                    for sample_id in sample_exposures
                }
            elif (
                training_budget.mode == "epochs"
                and (
                    not training_budget.drop_last
                    or training_budget.selected_samples % training_budget.batch_size == 0
                )
                and counters.samples_seen
                == resumed_epoch * training_budget.selected_samples
            ):
                # Older checkpoints did not persist per-sample exposure counts.
                # A checkpoint written after a complete epoch has presented each
                # selected sample exactly once, so this legacy state is exact.
                sample_exposures = {
                    sample_id: resumed_epoch for sample_id in sample_exposures
                }
            initialization_report = {
                "mode": "resume_full_state",
                "checkpoint_path": str(resume_path),
                "strict": True,
                "loaded_module_prefixes": ["<all>"],
                "missing_keys": [],
                "unexpected_keys": [],
                "optimizer_loaded": True,
                "scheduler_loaded": True,
                "scaler_loaded": True,
                "counters_loaded": True,
            }
        terminal_cfg = dict(config.get("terminal_output", {}))
        show_progress = bool(terminal_cfg.get("progress_bars", True))
        leave_progress = bool(terminal_cfg.get("leave_progress_bars", True))
        _print_model_summary(
            model,
            run_dir=run_dir,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype_name=amp_dtype_name,
            train_samples=len(train_indices),
            validation_samples=len(validation_indices),
            train_batches=len(train_loader),
            validation_batches=len(validation_loader),
            max_epochs=max_epochs,
            show_model=bool(terminal_cfg.get("show_model", True)),
        )
        config["resolved_runtime"] = {
            "device": str(device),
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype_name,
            "scene_ids": data_cfg.get("scene_ids"),
            "selected_manifest_samples": len(selected_manifest_samples),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "train_batch_size": int(data_cfg.get("train_batch_size", 2)),
            "actual_batch_size": budget_batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_views_per_optimizer_step": effective_views_per_optimizer_step,
            "peak_gpu_memory_mb": None,
            "validation_batch_size": int(data_cfg.get("validation_batch_size", 2)),
            "num_workers": workers,
            "stage_name": stage_name,
            "max_optimizer_steps": train_cfg.get("max_optimizer_steps"),
            "training_budget": training_budget.to_dict(),
            "multi_view_batch": multi_view_cfg,
            "initialization": initialization_report,
            "freezing": freeze_report,
        }
        _atomic_json(
            run_dir / "initialization_summary.json",
            {**WARNING_FLAGS, **initialization_report,
             "total_parameter_count": sum(p.numel() for p in model.parameters()),
             "trainable_parameter_count": freeze_report["trainable_parameter_count"]},
        )
        (run_dir / "resolved_config.py").write_text(
            "config = " + pprint.pformat(config, sort_dicts=False) + "\n",
            encoding="utf-8",
        )
        _atomic_json(run_dir / "resolved_config.json", config)
        _atomic_json(run_dir / "environment.json", environment)
        _atomic_json(run_dir / "cuda_environment.json", environment)
        _atomic_json(
            run_dir / "training_budget.json",
            {
                **training_budget.to_dict(),
                **sample_exposure_statistics(
                    sample_exposures,
                    target=training_budget.target_sample_exposures,
                ),
                "per_sample_exposures": sample_exposures,
            },
        )
        dataset.write_filter_artifacts(run_dir / "data_filter")
        run_manifest = {
            **WARNING_FLAGS,
            "run_id": run_id,
            "dataset_root": manifest["dataset_root"],
            "train_manifest": str(manifest_path.resolve()),
            "validation_manifest": str(manifest_path.resolve()),
            "manifest_file_sha256": manifest_file_sha,
            "manifest_internal_sha256": manifest["manifest_sha256"],
            "template_path": manifest["template_path"],
            "template_sha256": manifest["template_sha256"],
            "symmetry_sidecar_path": manifest["symmetry_sidecar_path"],
            "symmetry_sidecar_sha256": manifest["symmetry_sidecar_sha256"],
            "manifest_type": manifest.get("manifest_type"),
            "physical_fragments_total": manifest.get(
                "physical_fragments_total", manifest.get("physical_fragment_count")
            ),
            "accepted_physical_fragments": manifest.get(
                "accepted_physical_fragments", manifest.get("physical_fragment_count")
            ),
            "rejected_physical_fragments": manifest.get("rejected_physical_fragments", 0),
            "observations_total": manifest.get(
                "observations_total", manifest.get("accepted_observations")
            ),
            "accepted_observations": manifest.get("accepted_observations"),
            "rejected_observations": manifest.get("rejected_observations", 0),
            "selected_scene_ids": data_cfg.get("scene_ids"),
            "selected_manifest_samples": len(selected_manifest_samples),
            "train_samples_this_run": len(train_indices),
            "validation_samples_this_run": len(validation_indices),
            "model_parameter_count": sum(p.numel() for p in model.parameters()),
            "stage_name": stage_name,
            "initialization": initialization_report,
            "loaded_module_prefixes": initialization_report.get(
                "loaded_module_prefixes", []
            ),
            "frozen_module_prefixes": freeze_report["frozen_module_prefixes"],
            "trainable_module_prefixes": freeze_report["trainable_module_prefixes"],
            "trainable_parameter_count": freeze_report["trainable_parameter_count"],
            "checkpoint_policy": "best_only",
            "initialization_mode": initialization_report["mode"],
            "pretrained_checkpoint": None if from_scratch else init_checkpoint,
            "checkpoint_sources": initialization_report.get("checkpoint_sources", []),
        }
        _atomic_json(run_dir / "run_manifest.json", run_manifest)
        if bool(data_cfg.get("single_fragment_contract", False)):
            debug_indices = sorted(all_indices, key=lambda index: dataset[index]["frame_id"])
            debug_entries = [
                {
                    "dataset_index": index,
                    "sample_id": dataset[index]["sample_id"],
                    "scene_id": dataset[index]["scene_id"],
                    "frame_id": dataset[index]["frame_id"],
                    "fragment_id": dataset[index]["fragment_id"],
                }
                for index in debug_indices
            ]
        else:
            debug_indices, debug_entries = select_debug_samples(
                dataset,
                all_indices,
                count=min(
                    int(config["debug_visualization"].get("num_samples", 8)),
                    len(all_indices),
                ),
                seed=seed,
            )
        _atomic_json(
            run_dir / "debug_visualization_manifest.json",
            {
                **WARNING_FLAGS,
                "seed": seed,
                "selection_is_fixed_for_all_epochs": True,
                "samples": debug_entries,
            },
        )
        history.record(
            "run_start",
            epoch=0,
            **counters.to_dict(),
            phase="setup",
            stage_name=stage_name,
            frozen_module_prefixes=freeze_report["frozen_module_prefixes"],
            trainable_parameter_count=freeze_report["trainable_parameter_count"],
            warnings=["train and validation use the same test observations"],
        )
        best_metric_name = str(train_cfg["best_metric"])
        best_mode = str(train_cfg.get("best_metric_mode", "min"))
        min_delta = float(train_cfg.get("best_metric_min_delta", 1e-6))
        tie_breaker_name = train_cfg.get("best_metric_tie_breaker")
        tie_breaker_name = str(tie_breaker_name) if tie_breaker_name else None
        tie_breaker_mode = str(
            train_cfg.get("best_metric_tie_breaker_mode", "max")
        )
        tie_breakers = train_cfg.get("best_metric_tie_breakers")
        if tie_breakers is not None and not isinstance(tie_breakers, Sequence):
            raise ValueError("train.best_metric_tie_breakers must be a sequence")
        checkpoint_filename = str(
            stage_cfg.get("checkpoint_filename", "best.pth")
        )
        early_stopping_patience = int(
            train_cfg.get("early_stopping_patience_evals", 0)
        )
        minimum_exposures_before_early_stop = int(
            train_cfg.get("min_sample_exposures_before_early_stop", 750)
        )
        evals_without_improvement = 0
        stop_reason: str | None = None
        diagnostic_failure: dict[str, Any] | None = None
        evaluation_diagnostic_records: list[dict[str, Any]] = []
        consecutive_rank_invalid_evals = 0

        def evaluate_epoch(epoch: int) -> tuple[dict[str, float], bool]:
            nonlocal best_value, best_epoch, best_checkpoint, best_metrics_snapshot
            nonlocal evals_without_improvement, stop_reason
            nonlocal diagnostic_failure
            nonlocal consecutive_rank_invalid_evals
            surface_loss_cfg = config["loss"].get("joint_surface_correspondence_pose_v3")
            if isinstance(surface_loss_cfg, dict):
                surface_loss_cfg["_runtime_epoch"] = int(epoch)
            metrics, rows = _evaluate(
                model,
                validation_loader,
                device,
                criterion,
                amp_enabled,
                amp_dtype,
                config["loss"],
                active_path_config=config.get("active_coordinate_path"),
                epoch=epoch,
                max_epochs=max_epochs,
                show_progress=show_progress,
                leave_progress=leave_progress,
            )
            # Exact/K16 projection temporarily reserves substantially more
            # memory than the differentiable q_aux path.  It has no tensors
            # needed by training, so release those allocator blocks before the
            # next full-view optimizer step.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            correspondence_head = getattr(model, "correspondence_head", None)
            if correspondence_head is not None and hasattr(
                correspondence_head, "set_patch_recall"
            ):
                correspondence_head.set_patch_recall(
                    metrics.get("eval/coarse_patch_top4_recall", 0.0)
                )
            _write_evaluation(
                run_dir,
                epoch,
                metrics,
                rows,
                clean_active_only=bool(
                    config.get("active_coordinate_path", {}).get(
                        "clean_active_only", False
                    )
                ),
            )
            exposure_values = sample_exposure_statistics(
                sample_exposures, target=training_budget.target_sample_exposures
            )
            evaluation_diagnostic_records.append(
                {**metrics, "optimizer_step": counters.optimizer_step}
            )
            if best_metric_name not in metrics:
                raise KeyError(f"configured best metric is missing: {best_metric_name}")
            current = float(metrics[best_metric_name])
            improved = is_checkpoint_improvement(
                metrics,
                best_metrics_snapshot,
                metric_name=best_metric_name,
                mode=best_mode,
                min_delta=min_delta,
                tie_breaker_name=tie_breaker_name,
                tie_breaker_mode=tie_breaker_mode,
                tie_breakers=tie_breakers,
            )
            if improved:
                best_value = current
                best_epoch = epoch
                best_metrics_snapshot = dict(metrics)
                best_checkpoint = _save_best_checkpoint(
                    run_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    counters=counters,
                    checkpoint_filename=checkpoint_filename,
                    best_metric_name=best_metric_name,
                    metrics=metrics,
                    config=config,
                    manifest=manifest,
                    manifest_file_sha256=manifest_file_sha,
                    environment=environment,
                    sample_exposures=sample_exposures,
                )
                evals_without_improvement = 0
            elif early_stopping_is_eligible(
                sample_exposures, minimum_exposures_before_early_stop
            ):
                evals_without_improvement += 1
                if (
                    early_stopping_patience > 0
                    and evals_without_improvement >= early_stopping_patience
                ):
                    stop_reason = "early_stopping"
            else:
                evals_without_improvement = 0
            plateau_config = config.get("plateau_detection")
            plateau = detect_rotation_context_plateau(
                evaluation_diagnostic_records,
                min_sample_exposures=float(exposure_values["min_sample_exposures"]),
                config=plateau_config,
            ) if isinstance(plateau_config, Mapping) else {"detected": False}
            gates = evaluate_correspondence_diagnostic_gates(
                metrics,
                config.get("diagnostic_gates"),
                min_sample_exposures=float(exposure_values["min_sample_exposures"]),
                model_config=config.get("model"),
            )
            rank_collapse_patience = int(
                train_cfg.get("rank_collapse_patience_evals", 0)
            )
            rank_valid = metrics.get("eval/all_samples_procrustes_rank_valid")
            if rank_valid is not None and float(rank_valid) < 1.0:
                consecutive_rank_invalid_evals += 1
            else:
                consecutive_rank_invalid_evals = 0
            rank_collapsed = (
                rank_collapse_patience > 0
                and consecutive_rank_invalid_evals >= rank_collapse_patience
            )
            if rank_collapsed:
                gates = {
                    "failed": True,
                    "diagnosis": "correspondence_rank_collapse",
                    "consecutive_rank_invalid_evals": consecutive_rank_invalid_evals,
                    "configured_patience_evals": rank_collapse_patience,
                }
            if plateau.get("detected") or gates.get("failed"):
                diagnostic_failure = plateau if plateau.get("detected") else gates
                stop_reason = f"diagnostic:{diagnostic_failure['diagnosis']}"
                failure_payload = {
                    **WARNING_FLAGS,
                    "status": "stopped_with_diagnosis",
                    "diagnosis": diagnostic_failure["diagnosis"],
                    "optimizer_step": counters.optimizer_step,
                    "sample_exposures": exposure_values,
                    "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
                    "best_epoch": best_epoch,
                    "current_checkpoint_saved": False,
                    "current_state_metadata_only": True,
                    "details": diagnostic_failure,
                }
                _atomic_json(run_dir / "diagnostic_failure.json", failure_payload)
                (run_dir / "failure_report.md").write_text(
                    "# Diagnostic stop\n\n"
                    f"- diagnosis: `{diagnostic_failure['diagnosis']}`\n"
                    f"- optimizer step: `{counters.optimizer_step}`\n"
                    f"- best checkpoint: `{failure_payload['best_checkpoint']}`\n\n"
                    "This is an intentional diagnostic stop, not a crash. The next stage must not start.\n",
                    encoding="utf-8",
                )
            history.record(
                "eval_epoch",
                epoch=epoch,
                **counters.to_dict(),
                phase="eval",
                stage_name=stage_name,
                frozen_module_prefixes=freeze_report["frozen_module_prefixes"],
                trainable_parameter_count=freeze_report["trainable_parameter_count"],
                is_best=improved,
                early_stopping_eligible=early_stopping_is_eligible(
                    sample_exposures, minimum_exposures_before_early_stop
                ),
                evals_without_improvement=evals_without_improvement,
                **{
                    key: value
                    for key, value in sample_exposure_statistics(
                        sample_exposures,
                        target=training_budget.target_sample_exposures,
                    ).items()
                    if key != "samples_seen"
                },
                current_best_metric=best_value,
                best_epoch=best_epoch,
                best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
                **metrics,
                **_gpu_memory(device),
            )
            if improved:
                history.record(
                    "best_checkpoint",
                    epoch=epoch,
                    **counters.to_dict(),
                    phase="checkpoint",
                    stage_name=stage_name,
                    is_best=True,
                    current_best_metric=best_value,
                    best_epoch=best_epoch,
                    best_checkpoint=str(best_checkpoint),
                )
            if bool(terminal_cfg.get("print_eval_metrics", True)):
                _print_eval_metrics(
                    epoch,
                    metrics,
                    improved=improved,
                    best_epoch=best_epoch,
                )
            return metrics, improved

        gradient_norm = 0.0
        last_module_gradient_norms = {
            label: 0.0 for label in FINE_GRADIENT_PREFIXES
        }
        last_nonzero_gradient_parameter_fraction = 0.0
        conditioning_required = {
            int(value)
            for value in config.get("conditioning_audit", {}).get(
                "required_epochs", ()
            )
            if isinstance(value, int)
        }
        baseline_epoch = start_epoch - 1
        if bool(train_cfg.get("evaluate_before_training", True)):
            baseline_metrics, baseline_improved = evaluate_epoch(baseline_epoch)
            if from_scratch:
                baseline_rows = (
                    run_dir / "evaluations" / f"epoch_{baseline_epoch:04d}"
                    / "per_sample_metrics.csv"
                )
                (run_dir / "scratch_initialization_per_sample.csv").write_bytes(
                    baseline_rows.read_bytes()
                )
                _atomic_json(
                    run_dir / "scratch_initialization_summary.json",
                    {
                        **WARNING_FLAGS,
                        "training_performed": False,
                        "initialization_mode": "scratch",
                        "pretrained_checkpoint": None,
                        "checkpoint_sources": [],
                        "initial_state_dict_sha256": initialization_report[
                            "initial_state_dict_sha256"
                        ],
                        "sample_count": len(validation_indices),
                        "frames": sorted({
                            int(sample["frame_id"])
                            for sample in selected_manifest_samples
                        }),
                        "fragments": sorted({
                            int(sample["fragment_id"])
                            for sample in selected_manifest_samples
                        }),
                        "metrics": baseline_metrics,
                    },
                )
            if baseline_epoch in conditioning_required:
                _write_conditioning_gradient_audit(
                    run_dir,
                    label=f"epoch_{baseline_epoch:04d}",
                    source_epoch=baseline_epoch,
                    gradient_norm=gradient_norm,
                    module_gradient_norms=last_module_gradient_norms,
                    nonzero_gradient_parameter_fraction=0.0,
                    counters=counters,
                    status="before_first_optimizer_step; see active graph audit",
                )
            if baseline_improved:
                _write_conditioning_gradient_audit(
                    run_dir,
                    label="best",
                    source_epoch=baseline_epoch,
                    gradient_norm=gradient_norm,
                    module_gradient_norms=last_module_gradient_norms,
                    nonzero_gradient_parameter_fraction=0.0,
                    counters=counters,
                    status="best_before_first_optimizer_step",
                )
        if bool(train_cfg.get("visualize_before_training", True)):
            paths = export_prediction_visualizations(
                model,
                dataset,
                debug_indices,
                collate,
                device,
                epoch=baseline_epoch,
                output_dir=(
                    run_dir
                    / "debug_visualizations"
                    / f"epoch_{baseline_epoch:04d}"
                ),
                config={
                    **config["debug_visualization"],
                    "reference_root": str(
                        run_dir / "debug_visualizations" / "reference"
                    ),
                    "pose_query_ranking": config["loss"].get(
                        "pose_query_ranking", {}
                    ),
                    "joint_correspondence_pose": config["loss"].get(
                        "joint_correspondence_pose", {}
                    ),
                    "joint_surface_correspondence_pose_v3": config["loss"].get(
                        "joint_surface_correspondence_pose_v3", {}
                    ),
                    "symmetry_region_activity": config["data"].get(
                        "symmetry_region_activity", {}
                    ),
                },
            )
            if bool(config["debug_visualization"].get("multifragment_layout", False)):
                paths.extend(
                    export_multifragment_overviews(
                        model, dataset, debug_indices, collate, device,
                        run_dir / "debug_visualizations" / f"epoch_{baseline_epoch:04d}" / "multifragment_overview",
                    )
                )
            history.record(
                "debug_visualization",
                epoch=baseline_epoch,
                **counters.to_dict(),
                phase="debug_visualization",
                stage_name=stage_name,
                current_best_metric=best_value,
                best_epoch=best_epoch,
                best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
                debug_visualization_paths=paths,
            )

        gradient_clip_value = train_cfg.get("gradient_clip_norm", 1.0)
        gradient_clip = (
            None
            if gradient_clip_value is None or float(gradient_clip_value) <= 0.0
            else float(gradient_clip_value)
        )
        log_interval = int(
            train_cfg.get(
                "log_interval_optimizer_steps",
                train_cfg.get("log_interval_steps", 10),
            )
        )
        performance_logging = dict(config.get("performance_logging", {}))
        progress_update_interval = int(
            performance_logging.get("progress_update_interval_steps", 1)
        )
        module_gradient_interval = int(
            performance_logging.get("per_module_gradient_norm_interval_steps", 1)
        )
        eval_step_interval = int(
            train_cfg.get("eval_interval_optimizer_steps", 0)
        )
        visualization_step_interval = int(
            train_cfg.get("debug_visualization_interval_optimizer_steps", 0)
        )
        eval_epoch_interval = int(train_cfg.get("eval_interval_epochs", 0))
        visualization_epoch_interval = int(
            train_cfg.get("debug_visualization_interval_epochs", 0)
        )
        scheduler_type = str(train_cfg["scheduler"].get("type", "cosine"))
        last_eval_step = counters.optimizer_step
        last_visualization_step = counters.optimizer_step
        completed_epoch = start_epoch - 1
        # Keep terminal artifact generation defined even if the configured
        # budget was already exhausted by the resumed checkpoint.
        optimizer.zero_grad(set_to_none=True)
        run_peak_gpu_memory_mb = 0.0
        for epoch in range(start_epoch, max_epochs + 1):
            surface_loss_cfg = config["loss"].get("joint_surface_correspondence_pose_v3")
            if isinstance(surface_loss_cfg, dict):
                surface_loss_cfg["_runtime_epoch"] = int(epoch)
            correspondence_head = getattr(model, "correspondence_head", None)
            if correspondence_head is not None and hasattr(correspondence_head, "set_epoch"):
                correspondence_head.set_epoch(epoch)
            if grouped_sampler is not None:
                grouped_sampler.set_epoch(epoch)
            if max_optimizer_steps is not None and counters.optimizer_step >= max_optimizer_steps:
                stop_reason = "max_optimizer_steps"
                break
            epoch_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model.train()
            loss_sums: dict[str, torch.Tensor] = {}
            sample_count = 0
            data_time = 0.0
            gradient_norm = 0.0
            batch_ready = time.perf_counter()
            train_progress = tqdm(
                train_loader,
                desc=(
                    f"train {epoch:04d}/{max_epochs:04d} "
                    f"opt={counters.optimizer_step}/{max_optimizer_steps or '-'}"
                ),
                unit="batch",
                dynamic_ncols=True,
                leave=leave_progress,
                disable=not show_progress,
            )
            for batch_index, batch in enumerate(train_progress):
                data_time += time.perf_counter() - batch_ready
                moved = move_to_device(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    if frozen_cache_payload is None:
                        prediction = model(moved)
                        total, losses = _loss_values(
                            prediction, moved, criterion, config["loss"]
                        )
                    else:
                        current_ids = tuple(map(str, batch["sample_id"]))
                        if current_ids != frozen_cache_sample_ids:
                            raise RuntimeError(
                                "cached four-view batch order changed after cache creation"
                            )
                        _, q_normalized = cached_fine_coordinate_forward(
                            model.correspondence_head.fine_feature_adapter,
                            model.correspondence_head.fine_coordinate_auxiliary_head,
                            frozen_cache_payload,
                        )
                        total, losses = fine_coordinate_active_loss(
                            q_normalized,
                            moved,
                            frozen_cache_payload["observed_valid_mask"],
                            config["loss"],
                        )
                    scaled = total / accumulation
                if not bool(torch.isfinite(total)):
                    raise RuntimeError(
                        f"non-finite training loss at epoch {epoch}, "
                        f"batch_step {counters.batch_step + 1}"
                    )
                scaler.scale(scaled).backward()
                batch_size = len(batch["sample_id"])
                counters.record_batch(batch_size)
                for sample_id in batch["sample_id"]:
                    key = str(sample_id)
                    if key not in sample_exposures:
                        raise KeyError(f"unexpected training sample ID: {key}")
                    sample_exposures[key] += 1
                sample_count += batch_size
                should_step = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(train_loader)
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    if not gradients:
                        raise RuntimeError("optimizer step has no gradients")
                    parameters_with_grad = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    trainable_numel = sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                    next_optimizer_step = counters.optimizer_step + 1
                    detailed_gradient_due = (
                        module_gradient_interval > 0
                        and next_optimizer_step % module_gradient_interval == 0
                    )
                    if detailed_gradient_due:
                        last_module_gradient_norms = _module_gradient_norms(model)
                        nonzero_gradient_numel = sum(
                            int(parameter.grad.detach().ne(0).sum())
                            for parameter in parameters_with_grad
                        )
                        last_nonzero_gradient_parameter_fraction = (
                            nonzero_gradient_numel / max(trainable_numel, 1)
                        )
                    if gradient_clip is None:
                        gradient_norm = math.sqrt(
                            sum(
                                float(parameter.grad.detach().float().square().sum())
                                for parameter in parameters_with_grad
                            )
                        )
                    else:
                        gradient_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                parameters_with_grad, gradient_clip,
                                error_if_nonfinite=True,
                            )
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    counters.record_optimizer_step()
                    global_step = counters.optimizer_step
                    if scheduler_type in {"constant", "linear_warmup_constant"}:
                        scheduler.step()
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        contribution = value.detach() * batch_size
                        loss_sums[key] = loss_sums.get(key, contribution.new_zeros(())) + contribution
                update_progress = (
                    progress_update_interval > 0
                    and counters.optimizer_step % progress_update_interval == 0
                )
                if update_progress:
                    train_progress.set_postfix(
                    {
                        "loss": f"{float(loss_sums.get('loss_total', total.detach()) / max(sample_count, 1)):.4f}",
                        "batch": counters.batch_step,
                        "opt": counters.optimizer_step,
                        "seen": counters.samples_seen,
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    },
                    refresh=False,
                    )
                if should_step and (
                    counters.optimizer_step % max(log_interval, 1) == 0
                    or batch_index + 1 == len(train_loader)
                ):
                    history.record(
                        "train_step",
                        epoch=epoch,
                        **counters.to_dict(),
                        phase="train",
                        stage_name=stage_name,
                        frozen_module_prefixes=freeze_report["frozen_module_prefixes"],
                        trainable_parameter_count=freeze_report["trainable_parameter_count"],
                        learning_rate=optimizer.param_groups[0]["lr"],
                        gradient_norm=gradient_norm,
                        **{
                            f"gradient_norm/{name}": value
                            for name, value in last_module_gradient_norms.items()
                        },
                        amp_scale=float(scaler.get_scale()),
                        current_best_metric=best_value,
                        best_epoch=best_epoch,
                        best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
                        warmup_progress=float(losses.get("warmup_progress", 1.0)),
                        current_pose_loss_weight=float(
                            losses.get("current_pose_loss_weight", 0.0)
                        ),
                        current_alignment_loss_weight=float(
                            losses.get("current_alignment_loss_weight", 0.0)
                        ),
                        **{
                            f"train/{key}": float(value.detach())
                            for key, value in losses.items()
                            if isinstance(value, torch.Tensor) and value.ndim == 0
                        },
                        **_gpu_memory(device),
                    )
                if (
                    max_optimizer_steps is not None
                    and counters.optimizer_step >= max_optimizer_steps
                ):
                    stop_reason = "max_optimizer_steps"
                    break
                batch_ready = time.perf_counter()
            if scheduler_type == "cosine":
                scheduler.step()
            completed_epoch = epoch
            # One synchronization point converts all detached train scalars.
            epoch_metric_tensor = torch.stack(
                [value / max(sample_count, 1) for value in loss_sums.values()]
            ) if loss_sums else torch.empty(0, device=device)
            epoch_metric_values = epoch_metric_tensor.cpu().tolist()
            epoch_metrics = {
                f"train/{key}": value
                for key, value in zip(loss_sums, epoch_metric_values)
            }
            if bool(terminal_cfg.get("print_train_epoch_summary", True)):
                tqdm.write(
                    f"[TRAIN epoch {epoch:04d}] "
                    f"batch_step={counters.batch_step} "
                    f"optimizer_step={counters.optimizer_step} "
                    f"samples_seen={counters.samples_seen} "
                    f"loss={epoch_metrics.get('train/loss_total', math.nan):.6f} "
                    f"sym={epoch_metrics.get('train/loss_symmetry_pose', math.nan):.6f} "
                    f"rank={epoch_metrics.get('train/loss_pose_query_ranking', 0.0):.6f} "
                    f"regP={epoch_metrics.get('train/loss_observed_regions', 0.0):.6f} "
                    f"regA={epoch_metrics.get('train/loss_active_regions', 0.0):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                )
            due_eval = (
                eval_step_interval > 0
                and counters.optimizer_step - last_eval_step >= eval_step_interval
            ) or (eval_epoch_interval > 0 and epoch % eval_epoch_interval == 0)
            if due_eval:
                _, improved = evaluate_epoch(epoch)
                last_eval_step = counters.optimizer_step
                if epoch in conditioning_required:
                    _write_conditioning_gradient_audit(
                        run_dir,
                        label=f"epoch_{epoch:04d}",
                        source_epoch=epoch,
                        gradient_norm=gradient_norm,
                        module_gradient_norms=last_module_gradient_norms,
                        nonzero_gradient_parameter_fraction=last_nonzero_gradient_parameter_fraction,
                        counters=counters,
                        status="after_real_backward",
                    )
                if improved:
                    _write_conditioning_gradient_audit(
                        run_dir,
                        label="best",
                        source_epoch=epoch,
                        gradient_norm=gradient_norm,
                        module_gradient_norms=last_module_gradient_norms,
                        nonzero_gradient_parameter_fraction=last_nonzero_gradient_parameter_fraction,
                        counters=counters,
                        status="best_after_real_backward",
                    )
            visualization_paths: list[str] = []
            due_visualization = (
                visualization_step_interval > 0
                and counters.optimizer_step - last_visualization_step
                >= visualization_step_interval
            ) or (
                visualization_epoch_interval > 0
                and epoch % visualization_epoch_interval == 0
            )
            if due_visualization:
                label = (
                    f"step_{counters.optimizer_step:06d}"
                    if bool(data_cfg.get("single_fragment_contract", False))
                    else f"epoch_{epoch:04d}"
                )
                visualization_paths = export_prediction_visualizations(
                    model,
                    dataset,
                    debug_indices,
                    collate,
                    device,
                    epoch=epoch,
                    output_dir=run_dir / "debug_visualizations" / label,
                    config={
                        **config["debug_visualization"],
                        "reference_root": str(
                            run_dir / "debug_visualizations" / "reference"
                        ),
                        "pose_query_ranking": config["loss"].get(
                            "pose_query_ranking", {}
                        ),
                        "joint_correspondence_pose": config["loss"].get(
                            "joint_correspondence_pose", {}
                        ),
                        "joint_surface_correspondence_pose_v3": config["loss"].get(
                            "joint_surface_correspondence_pose_v3", {}
                        ),
                        "symmetry_region_activity": config["data"].get(
                            "symmetry_region_activity", {}
                        ),
                    },
                )
                if bool(config["debug_visualization"].get("multifragment_layout", False)):
                    visualization_paths.extend(
                        export_multifragment_overviews(
                            model, dataset, debug_indices, collate, device,
                            run_dir / "debug_visualizations" / label / "multifragment_overview",
                        )
                    )
                last_visualization_step = counters.optimizer_step
                history.record(
                    "debug_visualization",
                    epoch=epoch,
                    **counters.to_dict(),
                    phase="debug_visualization",
                    stage_name=stage_name,
                    current_best_metric=best_value,
                    best_epoch=best_epoch,
                    best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
                    debug_visualization_paths=visualization_paths,
                )
            history.record(
                "train_epoch",
                epoch=epoch,
                **counters.to_dict(),
                phase="train",
                stage_name=stage_name,
                frozen_module_prefixes=freeze_report["frozen_module_prefixes"],
                trainable_parameter_count=freeze_report["trainable_parameter_count"],
                learning_rate=optimizer.param_groups[0]["lr"],
                gradient_norm=gradient_norm,
                **{
                    f"gradient_norm/{name}": value
                    for name, value in last_module_gradient_norms.items()
                },
                amp_scale=float(scaler.get_scale()),
                epoch_time_sec=time.perf_counter() - epoch_start,
                data_time_sec=data_time,
                current_best_metric=best_value,
                best_epoch=best_epoch,
                best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
                warmup_progress=float(losses.get("warmup_progress", 1.0)),
                current_pose_loss_weight=float(
                    losses.get("current_pose_loss_weight", 0.0)
                ),
                current_alignment_loss_weight=float(
                    losses.get("current_alignment_loss_weight", 0.0)
                ),
                debug_visualization_paths=visualization_paths,
                **epoch_metrics,
                **_gpu_memory(device),
            )
            if device.type == "cuda":
                run_peak_gpu_memory_mb = max(
                    run_peak_gpu_memory_mb,
                    torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                )
            if stop_reason is not None:
                break
        if stop_reason is None:
            stop_reason = "max_epochs"
        if counters.optimizer_step != last_eval_step:
            _, improved = evaluate_epoch(completed_epoch)
            last_eval_step = counters.optimizer_step
            if improved:
                _write_conditioning_gradient_audit(
                    run_dir,
                    label="best",
                    source_epoch=completed_epoch,
                    gradient_norm=gradient_norm,
                    module_gradient_norms=last_module_gradient_norms,
                    nonzero_gradient_parameter_fraction=last_nonzero_gradient_parameter_fraction,
                    counters=counters,
                    status="best_after_real_backward",
                )
        _write_conditioning_gradient_audit(
            run_dir,
            label="final",
            source_epoch=completed_epoch,
            gradient_norm=gradient_norm,
            module_gradient_norms=last_module_gradient_norms,
            nonzero_gradient_parameter_fraction=last_nonzero_gradient_parameter_fraction,
            counters=counters,
            status="final_after_real_backward",
        )
        joint_stage_gate = None
        if bool(
            config.get("loss", {}).get("joint_correspondence_pose", {}).get("enabled", False)
            or config.get("loss", {}).get("joint_surface_correspondence_pose_v3", {}).get("enabled", False)
        ):
            if best_epoch is None:
                raise RuntimeError("joint baseline finished without a best evaluation")
            materialize_best_evaluation(run_dir, best_epoch)
            joint_stage_gate = check_joint_stage(run_dir)
            history.record(
                "stage_gate",
                epoch=completed_epoch,
                **counters.to_dict(),
                phase="gate",
                stage_name=stage_name,
                stage_gate_status=("passed" if joint_stage_gate["stage_passed"] else "failed"),
                next_stage_allowed=joint_stage_gate["next_stage_allowed"],
                warnings=[] if joint_stage_gate["stage_passed"] else ["stage readiness gate failed; do not start the next stage"],
            )
        history.write_epoch_csv()
        checkpoints = sorted(path.name for path in (run_dir / "checkpoints").glob("*"))
        expected_checkpoints = sorted(
            [checkpoint_filename, "best_manifest.json", "best_metrics.json"]
        )
        if checkpoints != expected_checkpoints:
            raise RuntimeError(f"best-only checkpoint invariant violated: {checkpoints}")
        gate_failed = bool(
            joint_stage_gate is not None and not joint_stage_gate["stage_passed"]
        )
        diagnostic_failure_path = (
            str(run_dir / "diagnostic_failure.json")
            if (run_dir / "diagnostic_failure.json").is_file()
            else None
        )
        effective_diagnostic_failure = (
            diagnostic_failure
            if diagnostic_failure is not None
            else joint_stage_gate if gate_failed else None
        )
        summary = {
            **WARNING_FLAGS,
            "status": "ok",
            "run_status": "ok",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "device": str(device),
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype_name,
            "epochs": completed_epoch,
            **counters.to_dict(),
            "stop_reason": stop_reason,
            "stage_name": stage_name,
            "frozen_module_prefixes": freeze_report["frozen_module_prefixes"],
            "trainable_parameter_count": freeze_report["trainable_parameter_count"],
            "best_metric_name": best_metric_name,
            "best_metric": best_value,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_checkpoint),
            "history": str(history.path),
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "diagnostic_failure": effective_diagnostic_failure,
            "diagnostic_failure_path": diagnostic_failure_path,
            "stage_readiness": (
                None if joint_stage_gate is None
                else "passed" if joint_stage_gate["stage_passed"] else "failed"
            ),
            "stage_gate_status": (
                None if joint_stage_gate is None
                else "passed" if joint_stage_gate["stage_passed"] else "failed"
            ),
            "next_stage_allowed": (
                None if joint_stage_gate is None else joint_stage_gate["next_stage_allowed"]
            ),
            "computed_max_optimizer_steps": max_optimizer_steps,
            "actual_batch_size": budget_batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_views_per_optimizer_step": effective_views_per_optimizer_step,
            "peak_gpu_memory_mb": run_peak_gpu_memory_mb,
            **sample_exposure_statistics(
                sample_exposures,
                target=training_budget.target_sample_exposures,
            ),
        }
        config["resolved_runtime"]["peak_gpu_memory_mb"] = run_peak_gpu_memory_mb
        (run_dir / "resolved_config.py").write_text(
            "config = " + pprint.pformat(config, sort_dicts=False) + "\n",
            encoding="utf-8",
        )
        _atomic_json(run_dir / "resolved_config.json", config)
        _atomic_json(
            run_dir / "training_budget.json",
            {
                **training_budget.to_dict(),
                **sample_exposure_statistics(
                    sample_exposures,
                    target=training_budget.target_sample_exposures,
                ),
                "per_sample_exposures": sample_exposures,
                "early_stopping_minimum_exposures": minimum_exposures_before_early_stop,
                "early_stopping_eligible": early_stopping_is_eligible(
                    sample_exposures, minimum_exposures_before_early_stop
                ),
                "valid_early_stop_after_minimum_exposures": (
                    stop_reason == "early_stopping"
                    and early_stopping_is_eligible(
                        sample_exposures, minimum_exposures_before_early_stop
                    )
                ),
                "stop_reason": stop_reason,
            },
        )
        _atomic_json(run_dir / "final_summary.json", summary)
        _atomic_json(run_dir / "stage_summary.json", summary)
        _atomic_json(
            run_dir / "gradient_summary.json",
            {
                **WARNING_FLAGS,
                "stage_name": stage_name,
                "last_gradient_norm": gradient_norm,
                "all_observed_gradients_finite": math.isfinite(gradient_norm),
                "trainable_parameter_count": freeze_report["trainable_parameter_count"],
                "module_gradient_norms": last_module_gradient_norms,
                "nonzero_gradient_parameter_fraction": (
                    last_nonzero_gradient_parameter_fraction
                ),
                **counters.to_dict(),
            },
        )
        if joint_stage_gate is not None:
            archive = run_dir.parent / f"{run_dir.name}_report.tar.gz"
            print("\nJoint stage report command:", flush=True)
            print(
                f"python tools/package_joint_stage_report.py --run-dir {run_dir} --output {archive}",
                flush=True,
            )
            print(f"Expected archive: {archive}\n", flush=True)
            if config.get("stage_gate_dependencies", {}).get("local_substage"):
                compact_archive = run_dir.parent / f"{run_dir.name}_compact_report.tar.gz"
                print("Local substage compact package command:", flush=True)
                print(
                    "python tools/package_correspondence_head_stage.py "
                    f"--input {run_dir} --output {compact_archive}",
                    flush=True,
                )
                print(
                    "STOP: do not start the next local substage before external analysis.\n",
                    flush=True,
                )
        history.record(
            "run_end",
            epoch=completed_epoch,
            **counters.to_dict(),
            phase="complete",
            stage_name=stage_name,
            current_best_metric=best_value,
            best_epoch=best_epoch,
            best_checkpoint=str(best_checkpoint),
        )
        return summary
    except Exception as exc:
        history.record(
            "error",
            epoch=None,
            **counters.to_dict(),
            phase="error",
            current_best_metric=best_value,
            best_epoch=best_epoch,
            best_checkpoint=str(best_checkpoint) if best_checkpoint else None,
            warnings=[repr(exc), traceback.format_exc()],
        )
        history.write_epoch_csv()
        _atomic_json(
            run_dir / "final_summary.json",
            {
                **WARNING_FLAGS,
                "status": "error",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


__all__ = ["run_overfit_training"]
