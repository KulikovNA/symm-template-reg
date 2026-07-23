"""Naming contract for compact four-fragment/four-view visualization exports."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from symm_template_reg.engine.evaluator import move_to_device
from symm_template_reg.evaluation.active_coordinate import evaluate_active_sample
from symm_template_reg.models.pose.pose_representation import transform_points
from symm_template_reg.visualization.ply import write_colored_ply


def required_multifragment_visualization_names(fragments=(0, 1, 2, 3), frames=(2, 4, 5, 8)):
    return {
        "per_frame": [f"frame_{frame:04d}_all_fragments_pred_vs_gt.ply" for frame in frames],
        "per_fragment": [f"fragment_{fragment:04d}_four_view_world_pose_comparison.ply" for fragment in fragments],
        "worst_sample": ["q_aux_vs_global_projection.ply", "q_aux_vs_k16_projection.ply", "reconstructed_visible_shell.ply", "visualization_summary.json"],
    }


@torch.no_grad()
def export_multifragment_overviews(model, dataset, indices, collate, device, output_dir):
    """Export per-frame, per-fragment and worst-sample clean active-path views."""
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    palette = np.asarray([[230,60,70],[40,190,90],[70,120,245],[230,170,40]], dtype=np.uint8)
    frame_items = defaultdict(list); fragment_items = defaultdict(list); evaluated = []
    was_training = model.training; model.eval()
    for index in indices:
        sample = dataset[index]
        batch = move_to_device(collate([sample]), device)
        prediction = model(batch)
        valid = prediction.observed_valid_mask[0]
        result = evaluate_active_sample(
            q_aux_O=prediction.correspondence_points_O[0], valid_mask=valid,
            target_O=sample["gt"]["points_O_corresponding"].to(device),
            observed_C=sample["observed"]["points_C"].to(device),
            vertices_O=sample["template"]["points_O"].to(device),
            faces=sample["template"]["faces"].to(device),
            equivalent_pose=sample["gt"]["T_C_from_O"].to(device),
            procrustes=model.weighted_procrustes, candidate_k=16, projection_chunk_size=256,
        )
        predicted_pose = result["T_C_from_O"]["exact_global"]
        gt_pose = sample["gt"]["T_C_from_O"].to(device)
        world = sample["gt"]["T_W_from_C"].to(device)
        vertices = sample["template"]["points_O"].to(device)
        item = {
            "sample": sample, "prediction": prediction, "result": result,
            "predicted_C": transform_points(predicted_pose[None], vertices[None])[0],
            "gt_C": transform_points(gt_pose[None], vertices[None])[0],
            "predicted_W": transform_points((world @ predicted_pose)[None], vertices[None])[0],
        }
        frame_items[int(sample["frame_id"])].append(item)
        fragment_items[int(sample["fragment_id"])].append(item)
        score = (
            result["exact_global"]["projected_correspondence_p95_mm"] / 2.5
            + result["exact_global"]["projection_alignment_p95_mm"] / 2.5
            + result["exact_global"]["projection_rotation_error_deg"]
            + result["exact_global"]["projection_translation_error_mm"] / 0.5
        )
        evaluated.append((score, item))
    written = []
    for frame, items in sorted(frame_items.items()):
        points=[]; colors=[]
        for item in items:
            fragment=int(item["sample"]["fragment_id"]); color=palette[fragment]
            observed=item["sample"]["observed"]["points_C"].cpu().numpy()
            predicted=item["predicted_C"].cpu().numpy(); gt=item["gt_C"].cpu().numpy()
            points.extend((observed,predicted,gt)); colors.extend((np.broadcast_to(color,(len(observed),3)),np.broadcast_to(color,(len(predicted),3)),np.broadcast_to((color*0.35).astype(np.uint8),(len(gt),3))))
        path=destination/f"frame_{frame:04d}_all_fragments_pred_vs_gt.ply"
        write_colored_ply(path,np.concatenate(points),np.concatenate(colors),comments=("bright=predicted template; dark=GT template; fragment color identifies physical fragment",)); written.append(str(path))
    view_palette=np.asarray([[230,60,70],[40,190,90],[70,120,245],[230,170,40]],dtype=np.uint8)
    for fragment,items in sorted(fragment_items.items()):
        points=[]; colors=[]
        for view,item in enumerate(sorted(items,key=lambda x:int(x["sample"]["frame_id"]))):
            value=item["predicted_W"].cpu().numpy(); points.append(value); colors.append(np.broadcast_to(view_palette[view],(len(value),3)))
        path=destination/f"fragment_{fragment:04d}_four_view_world_pose_comparison.ply"
        write_colored_ply(path,np.concatenate(points),np.concatenate(colors),comments=("one color per frame; exact-global uniform-Procrustes pose",)); written.append(str(path))
    score,worst=max(evaluated,key=lambda value:value[0]); result=worst["result"]
    q=worst["prediction"].correspondence_points_O[0,worst["prediction"].observed_valid_mask[0]].cpu().numpy()
    for name,projected in (("global",result["projected_points_O"]["exact_global"]),("k16",result["projected_points_O"]["k16"])):
        value=projected.cpu().numpy(); path=destination/f"q_aux_vs_{name}_projection.ply"
        write_colored_ply(path,np.concatenate((q,value)),np.concatenate((np.broadcast_to([220,50,210],(len(q),3)),np.broadcast_to([40,210,80],(len(value),3))))); written.append(str(path))
    reconstructed=transform_points(result["T_C_from_O"]["exact_global"][None],result["projected_points_O"]["exact_global"][None])[0]
    path=destination/"reconstructed_visible_shell.ply"; write_colored_ply(path,reconstructed.cpu(),[230,60,70]); written.append(str(path))
    summary=destination/"visualization_summary.json"; summary.write_text(json.dumps({"worst_sample":worst["sample"]["sample_id"],"worst_sample_score":score,"pose_source":"exact_global_uniform_procrustes","legacy_visualization":False},indent=2)+"\n"); written.append(str(summary))
    if was_training: model.train()
    return written


__all__ = ["export_multifragment_overviews", "required_multifragment_visualization_names"]
