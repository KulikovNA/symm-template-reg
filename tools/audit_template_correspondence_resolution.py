#!/usr/bin/env python3
"""Measure triangle and deterministic template-anchor quantization floors."""

from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from symm_template_reg.datasets.transforms import farthest_point_indices as numpy_fps  # noqa:E402
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa:E402
from _correspondence_diagnostics import actual_template_anchors,build_dataset,manifest_samples,nearest_sample_distance,statistics_mm,tensor_sha256  # noqa:E402

def _samples(vertices:torch.Tensor,faces:torch.Tensor,count:int)->tuple[torch.Tensor,str]:
    if count<=len(vertices):
        idx=np.sort(numpy_fps(vertices.numpy(),count));return vertices[idx],f'fps_template_vertices_count_{count}'
    tri=vertices[faces];area=torch.linalg.vector_norm(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)/2
    cdf=torch.cumsum(area,0)/area.sum();u=(torch.arange(count,dtype=torch.float64)+.5)/count;ids=torch.searchsorted(cdf.double(),u).clamp_max(len(tri)-1)
    # Deterministic low-discrepancy barycentric samples.
    r1=torch.frac(torch.arange(count,dtype=torch.float64)*0.6180339887498949+.5);r2=torch.frac(torch.arange(count,dtype=torch.float64)*0.4142135623730950+.25);s=torch.sqrt(r1)
    bary=torch.stack((1-s,s*(1-r2),s*r2),-1).float();return (tri[ids]*bary[:,:,None]).sum(1),f'area_cdf_golden_barycentric_count_{count}'

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=False)
    _,dataset=build_dataset(a.config,out,shell_only=True);_,samples=manifest_samples(dataset,a.manifest);template=samples[0]['template'];vertices=template['points_O'];faces=template['faces'];anchors=actual_template_anchors(samples[0],512)
    np.savetxt(out/'actual_template_anchors_512.csv',anchors.numpy(),delimiter=',',header='x,y,z',comments='')
    supports={'anchors_512':(anchors,'repository_fine4096_then_model_fps512')}
    for count in (2048,4096,8192):supports[f'samples_{count}']=_samples(vertices,faces,count)
    rows=[]
    for sample in samples:
        q=sample['gt']['points_O_corresponding'];triangle=closest_points_on_triangle_mesh(q,vertices,faces)['distances'];rows.append({'frame_id':int(sample['frame_id']),'support':'mesh_triangles',**statistics_mm(triangle)})
        for name,(support,policy) in supports.items():rows.append({'frame_id':int(sample['frame_id']),'support':name,'sampling_policy':policy,'support_count':len(support),**statistics_mm(nearest_sample_distance(q,support))})
    anchor_rows=[r for r in rows if r['support']=='anchors_512']; impossible=any(r['p95_mm']>2 for r in anchor_rows)
    policies={name:{'policy':policy,'coordinate_sha256':tensor_sha256(value),'count':len(value)} for name,(value,policy) in supports.items()};policy_encoded=json.dumps(policies,sort_keys=True).encode();summary={'audit_passed':True,'hard_current_anchor_gate_impossible':impossible,'criterion':'nearest current anchor p95 > 2 mm','sampling_policy_sha256':hashlib.sha256(policy_encoded).hexdigest(),'sampling_policies':policies,'metrics':rows}
    with (out/'template_resolution_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    (out/'template_resolution_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'template_resolution_report.md').write_text('# Template correspondence resolution\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n');print(json.dumps({'output_dir':str(out),**summary},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
