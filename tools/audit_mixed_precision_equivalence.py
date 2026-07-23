#!/usr/bin/env python3
"""Optional bf16/fp16 one-step ablation, gated by a passing fp32 audit."""

from __future__ import annotations

import argparse, copy, gc, json, math, os, sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT=Path(__file__).resolve().parents[1]; TOOLS=Path(__file__).resolve().parent
for value in (ROOT,TOOLS):
    if str(value) not in sys.path: sys.path.insert(0,str(value))
from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.config import load_config  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion,_loss_values  # noqa: E402
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.engine.single_fragment import build_selective_optimizer_parameter_groups  # noqa: E402
from symm_template_reg.models import build_model,register_all_modules  # noqa: E402

def _state(model): return {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
def _optimizer(cfg,model):
    groups=build_selective_optimizer_parameter_groups(model,default_lr=float(cfg["train"]["optimizer"]["lr"]),prefix_learning_rates=cfg["stage"]["prefix_learning_rates"])
    return torch.optim.AdamW(groups,lr=float(cfg["train"]["optimizer"]["lr"]),weight_decay=0.)
def _branch(cfg,state,host,device,dtype=None):
    seed_everything(int(cfg.get("seed",0))); model=build_model(cfg["model"]).to(device); model.load_state_dict(state); model.train()
    optimizer=_optimizer(cfg,model); criterion=_build_pose_criterion(cfg); batch=move_to_device(host,device); optimizer.zero_grad(set_to_none=True)
    enabled=dtype is not None; scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    cfg["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"]=1
    with torch.autocast("cuda",dtype=dtype or torch.float16,enabled=enabled):
        prediction=model(batch); loss,losses=_loss_values(prediction,batch,criterion,cfg["loss"])
    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
    gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); scaler.step(optimizer); scaler.update()
    result={"q":prediction.correspondence_points_O.detach().cpu(),"pose":prediction.correspondence_pose.detach().cpu(),"loss":float(loss.detach()),"selected":losses["selected_shared_symmetry_element"].detach().cpu(),"rotation_deg":float(losses["rotation_error_deg"].detach()),"translation_mm":float(losses["translation_total_mm"].detach()),"gradients":gradients,"updated":_state(model)}
    del model,optimizer,batch; gc.collect(); torch.cuda.empty_cache(); return result
def _cmp(a,b):
    maximum=0.; dot=na=nb=0.
    for name,x in a.items():
        x=x.double().flatten(); y=b[name].double().flatten(); maximum=max(maximum,float((x-y).abs().max())); dot+=float(x@y); na+=float(x@x); nb+=float(y@y)
    return {"max_abs_diff":maximum,"cosine_similarity":dot/max(math.sqrt(na*nb),1e-30)}
def run(args):
    gate=json.loads(Path(args.fp32_audit).read_text())
    if not gate.get("audit_passed"): raise ValueError("AMP audit requires fp32 audit_passed=true")
    output=Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True,exist_ok=False); device=torch.device("cuda")
    if hasattr(torch.backends.cuda,"enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    cfg,_,_,_,samples,collate,initial=load_multifragment_context(args.baseline_config,args.manifest,output/"dataset_cache","cpu"); state=_state(initial); del initial
    amp_cfg=load_config(args.amp_config); register_all_modules(); host=collate(samples); baseline=_branch(copy.deepcopy(cfg),state,host,device)
    reports={}
    for label,dtype in (("bf16",torch.bfloat16),("fp16",torch.float16)):
        try:
            trial=_branch(copy.deepcopy(amp_cfg),state,host,device,dtype); gradients=_cmp(baseline["gradients"],trial["gradients"]); updates=_cmp(baseline["updated"],trial["updated"])
            reports[label]={"status":"completed","q_aux_max_abs_diff":float((baseline["q"]-trial["q"]).abs().max()),"pose_max_abs_diff":float((baseline["pose"]-trial["pose"]).abs().max()),"loss_abs_diff":abs(baseline["loss"]-trial["loss"]),"selected_symmetry_exact":bool(torch.equal(baseline["selected"],trial["selected"])),"rotation_deg_abs_diff":abs(baseline["rotation_deg"]-trial["rotation_deg"]),"translation_mm_abs_diff":abs(baseline["translation_mm"]-trial["translation_mm"]),"gradients":gradients,"updated_parameters":updates}
        except RuntimeError as error:
            gc.collect(); torch.cuda.empty_cache()
            reports[label]={"status":"rejected","reason":str(error),"enabled_in_main_config":False}
    payload={"optional_ablation":True,"main_config_remains_fp32":True,"results":reports}
    (output/"mixed_precision_equivalence.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return 0
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--baseline-config",required=True); p.add_argument("--amp-config",required=True); p.add_argument("--manifest",required=True); p.add_argument("--fp32-audit",required=True); p.add_argument("--output-dir",required=True); return run(p.parse_args())
if __name__=="__main__": raise SystemExit(main())
