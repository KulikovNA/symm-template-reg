#!/usr/bin/env python3
"""Merge B1 fine-query and B2 barycentric weights into a Stage-A checkpoint."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a", required=True)
    parser.add_argument("--b1", required=True)
    parser.add_argument("--b2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {name: Path(value).expanduser().resolve() for name, value in (("stage_a", args.stage_a), ("b1", args.b1), ("b2", args.b2))}
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    payloads = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    merged = deepcopy(payloads["stage_a"])
    copied = []
    for source, prefix in (("b1", "correspondence_head.fine_query."), ("b2", "correspondence_head.barycentric_head.")):
        for key, value in payloads[source]["model"].items():
            if key.startswith(prefix):
                merged["model"][key] = value
                copied.append({"source": source, "key": key})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    manifest = {"output": str(output), "sources": {key: str(value) for key, value in paths.items()}, "copied": copied, "semantics": "model-only initialization; optimizer/scheduler state must not be resumed"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
