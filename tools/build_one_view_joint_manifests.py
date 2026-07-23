#!/usr/bin/env python3
"""Add shell-only frame04/frame08 manifests without touching existing manifests."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from symm_template_reg.engine.single_fragment import manifest_content_sha256  # noqa:E402
from symm_template_reg.engine.view_ladder import subset_view_manifest  # noqa:E402

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--overwrite-shell-only',action='store_true');a=p.parse_args();src=Path(a.source_manifest).expanduser().resolve();out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True);source=json.loads(src.read_text());written=[]
 for frame in (4,8):
  payload=subset_view_manifest(source,(frame,));payload['registration_point_selection']='shell_only';payload['deterministic_point_selection']=True
  for sample in payload['samples']:
   with np.load(sample['visible_points_path'],allow_pickle=False) as arrays:
    rows=(arrays['fragment_id']==int(sample['fragment_id'])) & (arrays['surface_label']==0)
    sample['num_observed_points']=int(rows.sum());sample['registration_point_selection']='shell_only'
  payload['manifest_sha256']=manifest_content_sha256(payload);path=out/f'fragment0002_frame{frame:02d}_only.json'
  if path.exists() and not a.overwrite_shell_only:raise FileExistsError(path)
  if path.exists():
   existing=json.loads(path.read_text())
   if existing.get('registration_point_selection')!='shell_only':raise ValueError(f'refusing to overwrite non-shell manifest: {path}')
  encoded=(json.dumps(payload,indent=2)+'\n').encode();path.write_bytes(encoded);digest=hashlib.sha256(encoded).hexdigest();path.with_suffix('.json.sha256').write_text(f'{digest}  {path.name}\n');written.append({'path':str(path),'frame_id':frame,'sample_count':1,'effective_group':'C2','registration_point_selection':'shell_only','train_validation_same_samples':True,'file_sha256':digest})
 combined=subset_view_manifest(source,(4,8));combined['registration_point_selection']='shell_only';combined['deterministic_point_selection']=True
 for sample in combined['samples']:
  with np.load(sample['visible_points_path'],allow_pickle=False) as arrays:
   rows=(arrays['fragment_id']==int(sample['fragment_id'])) & (arrays['surface_label']==0)
   sample['num_observed_points']=int(rows.sum());sample['registration_point_selection']='shell_only'
 combined['manifest_sha256']=manifest_content_sha256(combined);combined_path=out/'fragment0002_views02_shell_only.json'
 if combined_path.exists() and not a.overwrite_shell_only:raise FileExistsError(combined_path)
 encoded=(json.dumps(combined,indent=2)+'\n').encode();combined_path.write_bytes(encoded);digest=hashlib.sha256(encoded).hexdigest();combined_path.with_suffix('.json.sha256').write_text(f'{digest}  {combined_path.name}\n');written.append({'path':str(combined_path),'frame_ids':[4,8],'sample_count':2,'effective_group':'C2','registration_point_selection':'shell_only','train_validation_same_samples':True,'file_sha256':digest})
 print(json.dumps({'source_manifest':str(src),'written':written},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
