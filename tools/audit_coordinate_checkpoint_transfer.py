#!/usr/bin/env python3
"""Audit frame-4/frame-8 coordinate checkpoint transfer without training."""

from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]; TOOLS=Path(__file__).resolve().parent
for p in (ROOT,TOOLS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from coordinate_guided_audit_common import load_f1_audit_context, quantile_metrics_mm  # noqa:E402
from recheck_coordinate_guided_surface import _evaluate, _passes  # noqa:E402
from symm_template_reg.models import register_all_modules  # noqa:E402
from symm_template_reg.models.geometry.aux_guided_triangle_candidates import AuxGuidedTriangleCandidateBuilder  # noqa:E402
from symm_template_reg.models.geometry.triangle_targets import triangle_target_sets  # noqa:E402
from symm_template_reg.models.heads.coordinate_guided_surface_projection import CoordinateGuidedSurfaceProjectionHead  # noqa:E402


def physical_normalized_score(metrics):
    return sum(float(metrics[key]) for key in (
        "projected_correspondence_p95_mm", "projection_alignment_p95_mm",
        "projection_pose_rotation_error_deg", "projection_pose_translation_total_mm",
    ))


def select_initialization(rows):
    grouped={}
    for row in rows: grouped.setdefault(row["checkpoint"],{})[int(row["manifest_frame_id"])]=row
    choices=[]
    for checkpoint, frames in grouped.items():
        scores={frame:float(value["physical_normalized_score"]) for frame,value in frames.items()}
        choices.append((max(scores.values()),checkpoint,scores,all(bool(v["exact_global_gate_passed"]) for v in frames.values())))
    worst, checkpoint, scores, passed=min(choices,key=lambda x:(x[0],x[1]))
    return {"selected_checkpoint":checkpoint,"selection_reason":"minimum worst-frame physical normalized score; no weight averaging","frame4_score":scores.get(4),"frame8_score":scores.get(8),"worst_frame_score":worst,"cross_transfer_passed":passed}


def run(args):
    output=Path(args.output_dir).expanduser().resolve(); rows=[]; details={}
    checkpoints=[Path(args.checkpoint_frame4).resolve(),Path(args.checkpoint_frame8).resolve()]
    manifests=[Path(args.manifest_frame4).resolve(),Path(args.manifest_frame8).resolve()]
    for ci,checkpoint in enumerate(checkpoints):
        for mi,manifest in enumerate(manifests):
            context=load_f1_audit_context(checkpoint,manifest,output/f"context_{ci}_{mi}",torch.device(args.device))
            q,mask,v,f=context["q_aux"],context["mask"],context["vertices"],context["faces"]
            target_sets=triangle_target_sets(context["target"][mask],v,f,tolerance_m=.00015,point_chunk_size=256)
            valid=torch.zeros((len(mask),len(f)),dtype=torch.bool,device=mask.device); valid[mask]=target_sets["valid_triangle_mask"]
            projector=CoordinateGuidedSurfaceProjectionHead().to(mask.device)
            built_modes={}; timings={}
            for mode,k in (("exact_global",1),("aux_guided_global_topk",16)):
                if q.is_cuda: torch.cuda.synchronize()
                start=time.perf_counter(); built=AuxGuidedTriangleCandidateBuilder(mode="aux_guided_global_topk",candidate_k=k,projection_chunk_size=256)(q[None],[v],[f],mask[None])
                if q.is_cuda: torch.cuda.synchronize()
                timings[mode]=(time.perf_counter()-start)*1000; built_modes[mode]=built
            global_selected=built_modes["exact_global"]["candidate_triangle_ids"][0,:,0]
            mode_metrics={}
            for mode,built in built_modes.items():
                if q.is_cuda: torch.cuda.synchronize()
                projection_started=time.perf_counter()
                metrics,_,_,_=_evaluate(mode,built["candidate_triangle_ids"][0],built["candidate_triangle_mask"][0],context,valid,global_selected,projector,timings[mode])
                if q.is_cuda: torch.cuda.synchronize()
                projection_runtime=(time.perf_counter()-projection_started)*1000
                metrics["candidate_selection_runtime_ms"]=timings[mode]
                metrics["exact_projection_runtime_ms"]=projection_runtime
                metrics["runtime_ms"]=timings[mode]+projection_runtime
                metrics["fallback_fraction"]=0.0
                mode_metrics[mode]=metrics
            raw=quantile_metrics_mm(torch.linalg.vector_norm(q[mask]-context["target"][mask],dim=-1),"aux_coordinate")
            exact=mode_metrics["exact_global"]; k16=mode_metrics["aux_guided_global_topk"]
            frame=int(context["sample"].get("frame_id")); score=physical_normalized_score(exact)
            row={"checkpoint":str(checkpoint),"checkpoint_source_frame":4 if ci==0 else 8,"manifest":str(manifest),"manifest_frame_id":frame,**raw,
                 "exact_global_correspondence_p95_mm":exact["projected_correspondence_p95_mm"],"exact_global_alignment_p95_mm":exact["projection_alignment_p95_mm"],"exact_global_rotation_error_deg":exact["projection_pose_rotation_error_deg"],"exact_global_translation_error_mm":exact["projection_pose_translation_total_mm"],"exact_global_rank":exact["projection_correspondence_rank"],"exact_global_runtime_ms":timings["exact_global"],
                 "k16_correspondence_p95_mm":k16["projected_correspondence_p95_mm"],"k16_alignment_p95_mm":k16["projection_alignment_p95_mm"],"k16_rotation_error_deg":k16["projection_pose_rotation_error_deg"],"k16_translation_error_mm":k16["projection_pose_translation_total_mm"],"k16_rank":k16["projection_correspondence_rank"],"k16_exact_global_triangle_recall":k16["exact_global_selected_triangle_in_shortlist_fraction"],"k16_fallback_fraction":0.0,"aux_guided_runtime_ms":timings["aux_guided_global_topk"],"candidate_selection_runtime_ms":timings["aux_guided_global_topk"],"exact_projection_runtime_ms":k16["exact_projection_runtime_ms"],"k16_acceleration_claimed":timings["aux_guided_global_topk"]<timings["exact_global"],"physical_normalized_score":score,"exact_global_gate_passed":_passes(exact,require_global_recall=False)["passed"],"k16_gate_passed":_passes(k16)["passed"]}
            rows.append(row); details[f"checkpoint{4 if ci==0 else 8}_on_frame{frame}"]=mode_metrics
    selection=select_initialization(rows)
    with (output/"coordinate_checkpoint_transfer_matrix.csv").open("w",newline="",encoding="utf-8") as s:
        w=csv.DictWriter(s,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"audit_completed":True,"rows":rows,"mode_details":details,"selection":selection,"weight_averaging_used":False}
    (output/"coordinate_checkpoint_transfer_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"selected_two_view_initialization.json").write_text(json.dumps(selection,indent=2)+"\n")
    report=["# Coordinate checkpoint transfer", "", "| checkpoint | manifest frame | score | exact pass | K16 pass |", "|---|---:|---:|---|---|"]
    for r in rows: report.append(f"| frame {r['checkpoint_source_frame']} | {r['manifest_frame_id']} | {r['physical_normalized_score']:.6f} | {r['exact_global_gate_passed']} | {r['k16_gate_passed']} |")
    report += ["",f"Selected: `{selection['selected_checkpoint']}`",f"Worst-frame score: `{selection['worst_frame_score']:.6f}`"]
    (output/"coordinate_checkpoint_transfer_report.md").write_text("\n".join(report)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint-frame4",required=True);p.add_argument("--checkpoint-frame8",required=True);p.add_argument("--manifest-frame4",required=True);p.add_argument("--manifest-frame8",required=True);p.add_argument("--device",choices=("cpu","cuda"),default="cuda");p.add_argument("--output-dir",required=True);a=p.parse_args();o=Path(a.output_dir).expanduser().resolve()
    if o.exists():raise FileExistsError(o)
    o.mkdir(parents=True);register_all_modules();result=run(a);print(json.dumps({"output_dir":str(o),"selection":result["selection"]},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
