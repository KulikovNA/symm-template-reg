#!/usr/bin/env python3
"""Inspect the trained legacy global-soft correspondence head on frames 4/8."""

from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from symm_template_reg.engine.evaluator import move_to_device  # noqa:E402
from symm_template_reg.evaluation.correspondence_diagnostics import attention_distribution_metrics,covariance_geometry,local_rigidity_errors,pairwise_distance_correlation,rowwise_and_chamfer  # noqa:E402
from symm_template_reg.geometry import closest_points_on_triangle_mesh  # noqa:E402
from symm_template_reg.models import build_model  # noqa:E402
from symm_template_reg.models.losses import JointCorrespondencePoseLoss  # noqa:E402
from symm_template_reg.registry import COLLATE_FUNCTIONS,build_from_cfg  # noqa:E402
from symm_template_reg.visualization.ply import write_colored_ply  # noqa:E402
from _correspondence_diagnostics import actual_template_anchors,build_dataset,manifest_samples,statistics_mm  # noqa:E402

def _plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist() if value.ndim else float(value)
    return value

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--output-dir',required=True);a=p.parse_args();device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=False);run=Path(a.run_dir).expanduser().resolve();resolved=json.loads((run/'resolved_config.json').read_text());model=build_model(resolved['model']).to(device).eval();payload=torch.load(run/'checkpoints/best.pth',map_location=device,weights_only=False);model.load_state_dict(payload['model'],strict=True)
    criterion_cfg=dict(resolved['loss']['joint_correspondence_pose']);criterion_cfg.pop('enabled',None);criterion=JointCorrespondencePoseLoss(**criterion_cfg)
    rows=[];details=[]
    for selection,shell_only in (('original_all_fragment_points',False),('authoritative_shell_only',True)):
        config,dataset=build_dataset(a.config,out/f'cache_{selection}',shell_only=shell_only);_,samples=manifest_samples(dataset,a.manifest);collate=build_from_cfg(config['collate'],COLLATE_FUNCTIONS)
        for sample in samples:
            batch=move_to_device(collate([sample]),device)
            with torch.no_grad():
                prediction=model(batch);valid=prediction.observed_valid_mask[0];q_pred=prediction.correspondence_points_O[0,valid];p_c=batch['observed'].to_padded()['points'][0,valid];q_gt=batch['gt']['points_O_corresponding'].to_padded()['points'];template=batch['template'].to_padded();diag=criterion(prediction.correspondence_points_O,prediction.correspondence_pose,batch['gt']['T_C_from_O'],batch['observed'].to_padded()['points'],q_gt,prediction.observed_valid_mask,template['points'],template['valid_mask'],batch['template_symmetry_metadata'],batch['gt']['effective_symmetry_group']);matched=diag['matched_target_points_O'][0,valid]
            logits=prediction.correspondence_logits[0,valid];attention=attention_distribution_metrics(logits);rowset=rowwise_and_chamfer(q_pred,matched);row_distance=rowset['rowwise_distance'];rigidity=local_rigidity_errors(q_pred,p_c,8);axis=torch.as_tensor(sample['template']['symmetry_metadata'].axis.direction,device=device);pred_geometry=covariance_geometry(q_pred,axis);gt_geometry=covariance_geometry(matched,axis);surface=closest_points_on_triangle_mesh(q_pred,sample['template']['points_O'].to(device),sample['template']['faces'].to(device))['distances'];anchors=actual_template_anchors(sample,logits.shape[-1]).to(device)
            row={
                'selection':selection,'sample_id':sample['sample_id'],'frame_id':int(sample['frame_id']),'attention_matrix_rows':logits.shape[0],'attention_matrix_columns':logits.shape[1],
                'observed_points':int(valid.sum()),'observed_interaction_tokens':int(model.max_observed_tokens),'template_tokens':logits.shape[-1],'softmax_temperature':model.correspondence_head.temperature,
                'attention_entropy_mean':float(attention['entropy'].mean()),'attention_normalized_entropy_mean':float(attention['normalized_entropy'].mean()),'top1_probability_mean':float(attention['top1_mass'].mean()),'top5_probability_mass_mean':float(attention['top5_mass'].mean()),'top16_probability_mass_mean':float(attention['top16_mass'].mean()),
                'unique_argmax_template_anchors':int(attention['unique_argmax_anchors']),'collision_ratio':float(attention['collision_ratio']),'most_popular_anchor_fraction':float(attention['most_popular_anchor_fraction']),
                'row_correspondence_p50_mm':statistics_mm(row_distance)['p50_mm'],'row_correspondence_p95_mm':statistics_mm(row_distance)['p95_mm'],'symmetric_chamfer_mm':float(rowset['symmetric_chamfer_distance']*1000),
                'local_rigidity_p95_mm':float(torch.quantile(rigidity,.95)*1000),'pairwise_distance_correlation':pairwise_distance_correlation(q_pred,p_c),
                'q_pred_rank':int(pred_geometry['rank']),'q_gt_rank':int(gt_geometry['rank']),'q_pred_axial_extent_mm':float(pred_geometry['axial_extent']*1000),'q_gt_axial_extent_mm':float(gt_geometry['axial_extent']*1000),'q_pred_radial_extent_mm':float(pred_geometry['radial_extent']*1000),'q_gt_radial_extent_mm':float(gt_geometry['radial_extent']*1000),
                **{f'q_pred_template_surface_{k}':v for k,v in statistics_mm(surface).items()},
            };rows.append(row);details.append({**row,'q_pred_bbox':[_plain(pred_geometry['bbox_min']),_plain(pred_geometry['bbox_max'])],'q_gt_bbox':[_plain(gt_geometry['bbox_min']),_plain(gt_geometry['bbox_max'])],'q_pred_covariance_eigenvalues':_plain(pred_geometry['covariance_eigenvalues']),'q_gt_covariance_eigenvalues':_plain(gt_geometry['covariance_eigenvalues']),'selected_shared_symmetry_element':int(diag['selected_shared_symmetry_element'][0])})
            counts=attention['anchor_counts'].detach().cpu().numpy();scale=counts/max(int(counts.max()),1);anchor_colors=np.stack((255*scale,80*(1-scale),255*(1-scale)),1).astype(np.uint8);bad=(row_distance>0.002).detach().cpu().numpy();pred_colors=np.tile(np.array([230,40,40],dtype=np.uint8),(len(q_pred),1));pred_colors[bad]=[255,0,255];points=np.concatenate((sample['template']['points_O'].numpy(),anchors.cpu().numpy(),q_pred.detach().cpu().numpy()));colors=np.concatenate((np.tile([200,200,200],(len(sample['template']['points_O']),1)),anchor_colors,pred_colors));faces=sample['template']['faces'].numpy();visdir=out/selection/f'frame_{int(sample["frame_id"]):06d}';visdir.mkdir(parents=True,exist_ok=True);write_colored_ply(visdir/'attention_anchor_usage.ply',points,colors,faces=faces,comments=(f'collision_ratio={row["collision_ratio"]}', 'bad_row_error_magenta_threshold_mm=2'))
    shell=[r for r in rows if r['selection']=='authoritative_shell_only'];diagnoses=[]
    if any(r['row_correspondence_p95_mm']>2 and r['symmetric_chamfer_mm']<=2 for r in shell):diagnoses.append('point_order_or_correspondence_permutation_failure')
    if any(r['row_correspondence_p95_mm']>2 and r['symmetric_chamfer_mm']>2 for r in shell):diagnoses.append('wrong_template_region_failure')
    if any(r['attention_normalized_entropy_mean']>.7 for r in shell):diagnoses.append('diffuse_soft_matching_failure')
    if any(r['unique_argmax_template_anchors']/max(r['observed_points'],1)<.1 for r in shell):diagnoses.append('anchor_collapse')
    if any(min(json_detail['q_pred_covariance_eigenvalues'])/max(max(json_detail['q_pred_covariance_eigenvalues']),1e-12)<1e-3 for json_detail in details if json_detail['selection']=='authoritative_shell_only'):diagnoses.append('correspondence_geometry_collapse')
    summary={'audit_passed':True,'checkpoint':str(run/'checkpoints/best.pth'),'best_epoch':payload.get('epoch'),'diagnoses':diagnoses or ['no_threshold_diagnosis'],'metrics':details}
    with (out/'soft_correspondence_head_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
    (out/'soft_correspondence_head_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'soft_correspondence_head_report.md').write_text('# Soft correspondence head audit\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n');print(json.dumps({'output_dir':str(out),**summary},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
