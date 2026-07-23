#!/usr/bin/env python3
"""Audit authoritative shell-only registration rows for frames 4 and 8."""

from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa: E402
from symm_template_reg.models.pose.pose_representation import transform_points  # noqa: E402
from _correspondence_diagnostics import build_dataset,manifest_samples,statistics_mm  # noqa: E402

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    config,dataset=build_dataset(a.config,out,shell_only=True);_,samples=manifest_samples(dataset,a.manifest)
    rows=[]
    for sample in samples:
        with np.load(sample['meta']['visible_points_path'],allow_pickle=False) as arrays:
            fragment=arrays['fragment_id']==int(sample['fragment_id']); labels=arrays['surface_label'][fragment]
            counts={str(int(k)):int(v) for k,v in zip(*np.unique(labels,return_counts=True))}
        selected_labels=sample['observed']['surface_labels']; q=sample['gt']['points_O_corresponding']; pc=sample['observed']['points_C']; template=sample['template']
        closest=closest_points_on_triangle_mesh(q,template['points_O'],template['faces'])
        reconstructed=transform_points(sample['gt']['T_C_from_O'],q)
        reconstruction=torch.linalg.vector_norm(reconstructed-pc,dim=-1)
        q_metrics=statistics_mm(closest['distances']); reconstruction_metrics=statistics_mm(reconstruction)
        count=len(pc)
        rows.append({
            'sample_id':sample['sample_id'],'frame_id':int(sample['frame_id']),
            'total_points_in_npz_for_fragment':int(fragment.sum()),'shell_point_count':counts.get('0',0),
            'fracture_point_count':counts.get('1',0),'other_surface_label_counts':{k:v for k,v in counts.items() if k not in {'0','1'}},
            'points_passed_to_model':count,'points_in_correspondence_loss':count,'points_in_procrustes':count,'points_in_evaluation':count,
            'selected_non_shell_count':int(selected_labels.ne(0).sum()),
            **{f'gt_q_template_surface_{k}':v for k,v in q_metrics.items()},
            **{f'gt_reconstruction_{k}':v for k,v in reconstruction_metrics.items()},
        })
    passed=all(r['selected_non_shell_count']==0 and r['gt_q_template_surface_p95_mm']<.5 and r['gt_reconstruction_p95_mm']<.1 for r in rows)
    report={'audit_passed':passed,'registration_point_selection':'shell_only','gates':{'gt_q_template_surface_p95_mm':.5,'gt_reconstruction_p95_mm':.1},'frames':rows}
    (out/'registration_point_contract_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'registration_point_contract_report.md').write_text('# Registration point contract\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    with (out/'registration_point_contract_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    print(json.dumps({'output_dir':str(out),**report},indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
