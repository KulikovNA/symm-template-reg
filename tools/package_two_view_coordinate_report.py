#!/usr/bin/env python3
"""Package two-view coordinate reports without binary artifacts."""
from __future__ import annotations
import argparse,io,json,shlex,tarfile
from pathlib import Path
ALLOWED={".json",".csv",".jsonl",".md"};EXCLUDED={".pth",".ply",".pt",".npy",".npz"}
def package(inputs,output):
    if output.exists():raise FileExistsError(output)
    files=[];output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,"w:gz") as a:
        for root in inputs:
            if not root.is_dir():raise FileNotFoundError(root)
            for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in ALLOWED):
                name=Path(root.name)/p.relative_to(root);a.add(p,arcname=str(name),recursive=False);files.append(str(name))
        data=json.dumps({"inputs":list(map(str,inputs)),"files":files,"allowed":sorted(ALLOWED),"excluded":sorted(EXCLUDED)},indent=2).encode();info=tarfile.TarInfo("packaging_manifest.json");info.size=len(data);a.addfile(info,io.BytesIO(data))
    return {"archive":str(output),"file_count":len(files),"excluded":sorted(EXCLUDED)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",action="append",required=True);p.add_argument("--output",required=True);a=p.parse_args();inputs=[Path(x).expanduser().resolve() for x in a.input];output=Path(a.output).expanduser().resolve();r=package(inputs,output);cmd="python tools/package_two_view_coordinate_report.py "+" ".join(f"--input {shlex.quote(str(x))}" for x in inputs)+f" --output {shlex.quote(str(output))}";print(json.dumps({**r,"exact_package_command":cmd},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
