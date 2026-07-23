#!/usr/bin/env python3
"""Build deterministic shell-only coarse-patch/triangle targets for V3 loss."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa:E402
from symm_template_reg.models.geometry.point_ops import farthest_point_indices  # noqa:E402
from _correspondence_diagnostics import actual_template_anchors,build_dataset,manifest_samples,statistics_mm,tensor_sha256  # noqa:E402

def main()->int:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--num-patches',type=int,default=64);p.add_argument('--local-candidates',type=int,default=32);a=p.parse_args();out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=False);_,dataset=build_dataset(a.config,out,shell_only=True);_,samples=manifest_samples(dataset,a.manifest);records=[]
 for sample in samples:
  template=sample['template'];anchors=actual_template_anchors(sample,512);ids,mask=farthest_point_indices(anchors[None],torch.ones((1,len(anchors)),dtype=torch.bool),a.num_patches);patch_points=anchors[ids[0,mask[0]]];triangles=template['points_O'][template['faces']];centroids=triangles.mean(1);candidate_faces=torch.cdist(patch_points.float(),centroids.float()).topk(min(a.local_candidates,len(centroids)),largest=False).indices;q=sample['gt']['points_O_corresponding'];nearest=closest_points_on_triangle_mesh(q,template['points_O'],template['faces']);face_ids=nearest['face_ids'];patch_ids=torch.cdist(centroids[face_ids].float(),patch_points.float()).argmin(-1);local_target=torch.full_like(face_ids,-1)
  for i,(patch,face) in enumerate(zip(patch_ids.tolist(),face_ids.tolist())):
   matches=torch.nonzero(candidate_faces[patch].eq(face),as_tuple=False).flatten();local_target[i]=int(matches[0]) if len(matches) else int(torch.cdist(centroids[face:face+1],centroids[candidate_faces[patch]]).argmin())
  path=out/f'frame_{int(sample["frame_id"]):06d}_patch_targets.npz';np.savez_compressed(path,coarse_patch_id=patch_ids.numpy(),fine_local_target=local_target.numpy(),nearest_triangle_id=face_ids.numpy(),barycentric=nearest['barycentric'].numpy(),target_projection_error_m=nearest['distances'].numpy());metrics=statistics_mm(nearest['distances']);records.append({'frame_id':int(sample['frame_id']),'sample_id':sample['sample_id'],'path':str(path),'point_count':len(q),'patch_count':len(patch_points),'local_candidates':candidate_faces.shape[1],'target_projection':metrics,'patch_points_sha256':tensor_sha256(patch_points),'target_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 passed=all(r['target_projection']['p95_mm']<.5 for r in records);summary={'targets_passed':passed,'target_projection_p95_gate_mm':.5,'records':records};(out/'template_patch_targets_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'output_dir':str(out),**summary},indent=2));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
