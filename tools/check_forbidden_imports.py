#!/usr/bin/env python
"""Fail when production code imports a neighbouring or legacy runtime package."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


FORBIDDEN = {
    "RegTR",
    "regtr",
    "GeoTransformer",
    "geotransformer",
    "RoITr",
    "roitr",
    "PointTransformerV3",
    "pointtransformerv3",
    "Pointcept",
    "pointcept",
    "taxpose",
    "PointDSC",
    "pointdsc",
    "detr",
    "DFAT",
    "dfat",
    "frag_template_reg",
    "frag_geometry_engine",
}


def scan(root: Path) -> list[dict[str, object]]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            for module in modules:
                root_name = module.split(".", 1)[0]
                if root_name in FORBIDDEN:
                    violations.append(
                        {"file": str(path), "line": node.lineno, "module": module}
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="symm_template_reg")
    args = parser.parse_args()
    violations = scan(Path(args.root))
    print(json.dumps({"status": "ok" if not violations else "failed", "violations": violations}, indent=2))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
