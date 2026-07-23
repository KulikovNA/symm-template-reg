#!/usr/bin/env python3
"""Построить консервативный граф активных и устаревших файлов репозитория."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


SOURCE_ROOTS = ("symm_template_reg", "tools", "tests", "configs", "docs")
ACTIVE_ENTRY_PATHS = {
    "symm_template_reg/models/detectors/coordinate_guided_surface_registration_v3.py",
    "symm_template_reg/models/losses/clean_coordinate_pose_loss_v3.py",
    "symm_template_reg/datasets/split_directory_fragment_dataset.py",
    "symm_template_reg/datasets/boundary_augmentation.py",
    "symm_template_reg/datasets/template_contract.py",
    "symm_template_reg/datasets/collate.py",
    "symm_template_reg/datasets/template_repository.py",
    "symm_template_reg/models/symmetry/metadata.py",
    "symm_template_reg/models/symmetry/groups.py",
    "symm_template_reg/models/symmetry/region_assignment.py",
    "symm_template_reg/models/symmetry/hypothesis_expander.py",
    "symm_template_reg/geometry/triangle_surface.py",
    "symm_template_reg/models/geometry/aux_guided_triangle_candidates.py",
    "symm_template_reg/models/pose/weighted_procrustes.py",
    "symm_template_reg/engine/runtime.py",
    "symm_template_reg/engine/production_trainer.py",
    "symm_template_reg/engine/production_evaluator.py",
    "tools/train.py",
    "tools/evaluate.py",
    "tools/inspect_dataset.py",
    "tools/visualize_predictions.py",
    "tools/visualize_boundary_augmentation.py",
    "tools/visualize_template_symmetry.py",
    "tools/profile_training.py",
    "tools/package_training_report.py",
    "tools/export_model.py",
    "tools/check_cuda.py",
    "tools/audit_active_repository_graph.py",
    "configs/train/coordinate_guided_surface_v3.py",
    "configs/eval/coordinate_guided_surface_v3.py",
    "configs/debug/smoke.py",
    "configs/debug/tiny_overfit.py",
    "configs/debug/augmentation_preview.py",
    "configs/debug/four_fragments_four_frames_overfit.py",
    "tests/test_split_directory_fragment_dataset.py",
    "tests/test_production_registry.py",
    "tests/test_package_training_report.py",
    "tests/boundary_augmentation_test_utils.py",
    "tests/test_boundary_augmentation_erosion.py",
    "tests/test_boundary_augmentation_dilation.py",
    "tests/test_boundary_augmentation_mixed.py",
    "tests/test_augmentation_updates_points_and_targets_together.py",
    "tests/test_added_points_have_valid_template_targets.py",
    "tests/test_added_points_use_gt_only_for_target.py",
    "tests/test_fracture_candidates_are_gated.py",
    "tests/test_depth_ring_candidates_are_gated.py",
    "tests/test_min_points_preserved.py",
    "tests/test_max_fraction_respected.py",
    "tests/test_deterministic_seed.py",
    "tests/test_validation_has_no_augmentation.py",
    "tests/test_test_has_no_augmentation.py",
    "tests/test_augmentation_does_not_modify_gt_pose.py",
    "tests/test_active_parameter_graph.py",
    "tests/test_all_active_modules_receive_gradients.py",
    "tests/test_clean_v3_checkpoint_has_no_legacy_keys.py",
    "tests/test_clean_v3_model_has_no_legacy_heads.py",
    "tests/test_fine_coordinate_auxiliary_head.py",
    "tests/test_fine_local_feature_adapter.py",
    "tests/test_weighted_procrustes.py",
    "tests/test_weighted_procrustes_masked.py",
    "tests/test_weighted_procrustes_degenerate.py",
    "tests/test_exact_global_projection_chunked.py",
    "tests/test_aux_guided_triangle_candidates.py",
    "tests/test_projection_global_vs_candidate.py",
    "tests/test_symmetry_groups.py",
    "tests/test_symmetry_metadata.py",
    "tests/test_fragment_symmetry_targets.py",
    "tests/test_collate.py",
    "tests/test_template_repository.py",
    "tests/test_fragment_mesh_filter.py",
    "tests/test_registration_point_selection_shell_only.py",
    "tests/test_four_fragments_four_frames_overfit_config.py",
    "tools/README.md",
    "tests/README.md",
    "configs/debug/README.md",
    "docs/ARCHITECTURE_RU.md",
    "docs/DATASET_RU.md",
    "docs/AUGMENTATION_RU.md",
    "docs/TRAINING_RU.md",
    "docs/INFERENCE_RU.md",
    "docs/SYMMETRY_RU.md",
    "docs/THIRD_PARTY_NOTICES.md",
    "docs/CLEANUP_REPORT.md",
}
LEGAL_NAMES = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "third_party_modules.json",
    "third_party_revisions.json",
}
GENERATED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
GENERATED_SUFFIXES = {".pyc", ".pyo"}
LEGACY_TOKENS = (
    "pose_query",
    "ranking",
    "region_head",
    "region_loss",
    "patch",
    "triangle_head",
    "triangle_classifier",
    "barycentric",
    "view_ladder",
    "conditioned_pose",
    "conditioned_symm",
    "residual_pose",
    "overlap_head",
    "overlap_loss",
    "uncertainty_head",
    "point_weight_head",
    "correspondence_confidence",
    "joint_correspondence_pose",
    "joint_surface_correspondence_pose",
    "surface_constrained_correspondence",
    "soft_coarse_local",
    "uniform_correspondence_procrustes",
)
PLANNED_KEEP_NAMES = {
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "environment_fracs.txt",
}


def _module_name(path: Path, repo: Path) -> str | None:
    relative = path.relative_to(repo)
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_imports(
    path: Path,
    repo: Path,
    module_paths: dict[str, str],
) -> tuple[set[str], list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set(), []
    module = _module_name(path, repo) or ""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    dependencies: set[str] = set()
    raw: list[str] = []

    def add(name: str) -> None:
        raw.append(name)
        candidate = name
        while candidate:
            if candidate in module_paths:
                dependencies.add(module_paths[candidate])
                return
            candidate = candidate.rpartition(".")[0]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                drop = node.level - 1
                if drop:
                    base_parts = base_parts[:-drop]
                base = ".".join(
                    [*base_parts, *([] if node.module is None else node.module.split("."))]
                )
            else:
                base = node.module or ""
            if base:
                add(base)
            for alias in node.names:
                if alias.name != "*":
                    add(".".join(part for part in (base, alias.name) if part))
    return dependencies, sorted(set(raw))


def _registry_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    decorator = re.compile(
        r"@(?P<registry>[A-Z_]+)\.register_module\([^)]*\)\s*"
        r"(?:class|def)\s+(?P<name>[A-Za-z_]\w*)",
        re.MULTILINE,
    )
    for match in decorator.finditer(text):
        entries.append(match.groupdict())
    return entries


def _transitive_closure(entries: set[str], imports: dict[str, set[str]]) -> set[str]:
    closure: set[str] = set()
    queue = deque(sorted(entries))
    while queue:
        current = queue.popleft()
        if current in closure:
            continue
        closure.add(current)
        queue.extend(sorted(imports.get(current, ())))
    return closure


def _status(
    relative: str,
    *,
    active_closure: set[str],
    registry_entries: list[dict[str, str]],
) -> tuple[str, str]:
    path = Path(relative)
    lower = relative.lower()
    if path.name in LEGAL_NAMES or "third_party_licenses/" in relative:
        return "legal_notice_keep", "Юридический notice/license нельзя удалять."
    if any(part in GENERATED_PARTS for part in path.parts) or path.suffix in GENERATED_SUFFIXES:
        return "generated", "Генерируемый cache/bytecode не является исходным кодом."
    if relative in ACTIVE_ENTRY_PATHS:
        return "keep", "Явная точка входа production CoordinateGuidedSurfaceRegistrationV3."
    legacy = any(token in lower for token in LEGACY_TOKENS)
    if registry_entries and legacy:
        if relative in active_closure:
            return (
                "keep",
                "Legacy registry-файл пока достижим из active graph; сначала требуется "
                "вынести активную часть или удалить принудительную регистрацию.",
            )
        return "delete", "Registry entry относится только к запрещённой legacy-архитектуре."
    if legacy:
        if relative in active_closure:
            return (
                "keep",
                "Смешанный/legacy файл пока импортируется active graph; удаление небезопасно.",
            )
        return "delete", "Файл относится к legacy experiment и не достижим из active entries."
    if relative in active_closure:
        return "keep", "Транзитивная Python-зависимость production entry."
    if path.name in PLANNED_KEEP_NAMES:
        return "keep", "Корневой metadata/documentation файл."
    if relative.startswith("schemas/") or relative.startswith("examples/"):
        return "keep", "Формат/пример сохранён консервативно до production Dataset audit."
    if relative.startswith("symm_template_reg/"):
        return (
            "delete",
            "Python-модуль не достижим из повторно проверенного production graph.",
        )
    if relative.startswith("configs/") or relative.startswith("docs/"):
        return "delete", "Не входит в минимальный production набор; заменить актуальным файлом."
    if relative.startswith("tests/"):
        return "delete", "Тест не покрывает достижимый production graph; проверить перед cleanup."
    if relative.startswith("tools/"):
        return "delete", "Диагностический tool не является production entry."
    return "keep", "Консервативно оставлен: недостаточно доказательств для удаления."


def audit_repository(repo_root: Path) -> dict[str, Any]:
    repo = repo_root.expanduser().resolve()
    files = sorted(
        path
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(repo).parts
    )
    module_paths = {
        module: str(path.relative_to(repo))
        for path in files
        if (module := _module_name(path, repo)) is not None
    }
    imports: dict[str, set[str]] = {}
    raw_imports: dict[str, list[str]] = {}
    imported_by: dict[str, set[str]] = defaultdict(set)
    registry_by_path: dict[str, list[dict[str, str]]] = {}
    texts: dict[str, str] = {}
    for path in files:
        relative = str(path.relative_to(repo))
        if path.suffix in {".py", ".md", ".json"} or path.name == "README.md":
            try:
                texts[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass
        if path.suffix == ".py":
            dependencies, raw = _resolve_imports(path, repo, module_paths)
            imports[relative] = dependencies
            raw_imports[relative] = raw
            for dependency in dependencies:
                imported_by[dependency].add(relative)
            registry_by_path[relative] = _registry_entries(texts.get(relative, ""))
    active_entries = {path for path in ACTIVE_ENTRY_PATHS if (repo / path).exists()}
    closure = _transitive_closure(active_entries, imports)
    config_texts = {
        path: text for path, text in texts.items() if path.startswith("configs/")
    }
    test_texts = {
        path: text for path, text in texts.items() if path.startswith("tests/")
    }
    rows: list[dict[str, Any]] = []
    all_registry: list[dict[str, str]] = []
    for path in files:
        relative = str(path.relative_to(repo))
        registry_entries = registry_by_path.get(relative, [])
        for entry in registry_entries:
            all_registry.append({"path": relative, **entry})
        symbols = [entry["name"] for entry in registry_entries]
        module = _module_name(path, repo)
        needles = {
            path.stem,
            *(symbols or ()),
            *((module,) if module else ()),
        }
        configs = sorted(
            candidate
            for candidate, text in config_texts.items()
            if any(needle and needle in text for needle in needles)
        )
        tests = sorted(
            candidate
            for candidate, text in test_texts.items()
            if any(needle and needle in text for needle in needles)
        )
        status, reason = _status(
            relative,
            active_closure=closure,
            registry_entries=registry_entries,
        )
        rows.append(
            {
                "path": relative,
                "status": status,
                "reason": reason,
                "imported_by": sorted(imported_by.get(relative, ())),
                "imports": sorted(imports.get(relative, ())),
                "raw_imports": raw_imports.get(relative, []),
                "registry_references": registry_entries,
                "configs_referencing_file": configs,
                "tests_covering_file": tests,
            }
        )
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["status"]] += 1
    return {
        "repo_root": str(repo),
        "active_entries": sorted(active_entries),
        "active_transitive_closure": sorted(closure),
        "counts": dict(sorted(counts.items())),
        "files": rows,
        "registry_entries": sorted(
            all_registry, key=lambda row: (row["registry"], row["name"], row["path"])
        ),
    }


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = result["files"]
    keep = [
        row for row in files
        if row["status"] in {"keep", "legal_notice_keep"}
    ]
    delete = [row for row in files if row["status"] == "delete"]
    generated = [row for row in files if row["status"] == "generated"]
    payloads = {
        "active_repository_graph.json": result,
        "keep_manifest.json": {"files": keep},
        "delete_manifest.json": {"files": delete},
        "active_registry_entries.json": {"entries": result["registry_entries"]},
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    lines = [
        "# План минимизации репозитория",
        "",
        f"- Active entries: {len(result['active_entries'])}",
        f"- Keep/legal: {len(keep)}",
        f"- Delete candidates: {len(delete)}",
        f"- Generated: {len(generated)}",
        "",
        "## Правило безопасности",
        "",
        "Удалять разрешено только пути из `delete_manifest.json`. Файлы со статусом "
        "`keep` из-за смешанной зависимости сначала разделяются, затем audit запускается повторно.",
        "",
        "## Активные точки входа",
        "",
        *[f"- `{path}`" for path in result["active_entries"]],
        "",
        "## Кандидаты на удаление",
        "",
        *[f"- `{row['path']}` — {row['reason']}" for row in delete],
        "",
    ]
    (output_dir / "cleanup_plan.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_repository(Path(args.repo_root))
    _write_outputs(result, Path(args.output_dir).expanduser().resolve())
    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                "counts": result["counts"],
                "active_entries": len(result["active_entries"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
