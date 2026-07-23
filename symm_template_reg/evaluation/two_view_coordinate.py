"""Pure helpers for two-view coordinate-stage scoring and gates."""
from __future__ import annotations
import torch
from symm_template_reg.models.pose.pose_representation import invert_transform
from symm_template_reg.models.pose.rotation import rotation_geodesic_distance

def projection_score(row):
    return sum(float(row[k]) for k in ("projected_correspondence_p95_mm","projection_alignment_p95_mm","projection_pose_rotation_error_deg","projection_pose_translation_total_mm"))

def worst_sample_projection_score(rows):
    if not rows: raise ValueError("at least one sample is required")
    return max(projection_score(row) for row in rows)

def two_view_gate(rows):
    per=[]
    for row in rows:
        checks={"correspondence":float(row["projected_correspondence_p95_mm"])<=1+1e-6,"alignment":float(row["projection_alignment_p95_mm"])<=1+1e-6,"rotation":float(row["projection_pose_rotation_error_deg"])<=1+1e-6,"translation":float(row["projection_pose_translation_total_mm"])<=1+1e-6,"rank":int(row["projection_correspondence_rank"])==3,"surface":float(row["surface_membership_p95_mm"])<=.1+1e-6,"recall":float(row["exact_global_selected_triangle_in_shortlist_fraction"])>=.995-1e-6,"fallback":float(row["fallback_fraction"])<=1e-6,"no_leakage":not bool(row.get("target_leakage_detected",False)),"no_nonfinite":not bool(row.get("nonfinite_detected",False))}
        per.append({"frame_id":row.get("frame_id"),"checks":checks,"passed":all(checks.values())})
    return {"stage_passed":len(per)==2 and all(x["passed"] for x in per),"per_sample":per}

def two_view_world_pose_metrics(T_W_from_C,T_C_from_O,symmetry_rotation=None):
    world=T_W_from_C@T_C_from_O;delta=invert_transform(world[0:1])[0]@world[1]
    rotation=torch.rad2deg(rotation_geodesic_distance(torch.eye(3,device=world.device)[None],delta[:3,:3][None]))[0]
    if symmetry_rotation is not None:rotation=torch.minimum(rotation,torch.as_tensor(symmetry_rotation,device=rotation.device,dtype=rotation.dtype))
    axis0=world[0,:3,2];axis1=world[1,:3,2];axis=torch.rad2deg(torch.acos(torch.dot(axis0,axis1).abs().clamp(-1,1)))
    return {"world_translation_difference_mm":float(torch.linalg.vector_norm(world[0,:3,3]-world[1,:3,3])*1000),"world_symmetry_aware_rotation_difference_deg":float(rotation),"world_axis_difference_deg":float(axis)}

__all__=["projection_score","worst_sample_projection_score","two_view_gate","two_view_world_pose_metrics"]
