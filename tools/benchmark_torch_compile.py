#!/usr/bin/env python3
"""Optional torch.compile steady-state benchmark gated by exact fp32 audit."""

from __future__ import annotations

import argparse, copy, gc, json, os, statistics, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT=Path(__file__).resolve().parents[1]; TOOLS=Path(__file__).resolve().parent
for value in (ROOT,TOOLS):
    if str(value) not in sys.path: sys.path.insert(0,str(value))
from multifragment_overfit_common import load_multifragment_context  # noqa: E402
from symm_template_reg.engine.evaluator import move_to_device  # noqa: E402
from symm_template_reg.engine.overfit_trainer import _build_pose_criterion,_loss_values  # noqa: E402
from symm_template_reg.engine.seed import seed_everything  # noqa: E402
from symm_template_reg.models import build_model  # noqa: E402

def _measure(model,batch,criterion,loss_cfg,warmup,steps):
    values=[]; compile_started=time.perf_counter()
    for step in range(warmup+steps):
        model.zero_grad(set_to_none=True); start=torch.cuda.Event(True); end=torch.cuda.Event(True); start.record()
        prediction=model(batch); loss,_=_loss_values(prediction,batch,criterion,loss_cfg); loss.backward(); end.record(); torch.cuda.synchronize()
        if step>=warmup: values.append(start.elapsed_time(end))
    return {"median_ms":statistics.median(values),"p90_ms":sorted(values)[max(0,int(.9*len(values))-1)],"compile_plus_run_wall_sec":time.perf_counter()-compile_started}
def run(args):
    gate=json.loads(Path(args.fp32_audit).read_text());
    if not gate.get("audit_passed"): raise ValueError("torch.compile benchmark requires fp32 audit_passed=true")
    output=Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True,exist_ok=False); device=torch.device("cuda")
    if args.record_observed_failure:
        payload={"optional_ablation":True,"enabled_in_main_config":False,"status":"rejected","selection":"do_not_enable","observed_graph_breaks":["PackedPointBatch.to_padded Tensor.item","runtime SHA256/cache host copy","index_put/scatter disable CUDA graphs"],"backend_failure":"Inductor cannot functionalize the in-place fill_diagonal_ mutation on an as_strided view chain in fine feature diagnostics","steady_state_measurement_available":False}
        (output/"torch_compile_benchmark.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return 0
    cfg,_,_,_,samples,collate,initial=load_multifragment_context(args.config,args.manifest,output/"dataset_cache","cpu"); state=copy.deepcopy(initial.state_dict()); del initial
    batch=move_to_device(collate(samples),device); criterion=_build_pose_criterion(cfg); cfg["loss"]["joint_surface_correspondence_pose_v3"]["_runtime_epoch"]=1
    seed_everything(int(cfg.get("seed",0))); eager=build_model(cfg["model"]).to(device); eager.load_state_dict(state); eager_result=_measure(eager,batch,criterion,cfg["loss"],args.warmup_steps,args.measure_steps); del eager; gc.collect(); torch.cuda.empty_cache()
    torch._dynamo.reset(); seed_everything(int(cfg.get("seed",0))); raw=build_model(cfg["model"]).to(device); raw.load_state_dict(state); compiled=torch.compile(raw,mode=args.mode,dynamic=True)
    try: compiled_result=_measure(compiled,batch,criterion,cfg["loss"],args.warmup_steps,args.measure_steps)
    except Exception as error:
        payload={"optional_ablation":True,"enabled_in_main_config":False,"status":"rejected","selection":"do_not_enable","eager":eager_result,"backend_failure":str(error),"dynamo_counters":{group:{str(k):int(v) for k,v in values.items()} for group,values in torch._dynamo.utils.counters.items()}}
        (output/"torch_compile_benchmark.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return 0
    counters={group:{str(k):int(v) for k,v in values.items()} for group,values in torch._dynamo.utils.counters.items()}
    recompiles=sum(counters.get("stats",{}).get(name,0) for name in ("unique_graphs","calls_captured"))
    payload={"optional_ablation":True,"enabled_in_main_config":False,"mode":args.mode,"eager":eager_result,"compiled":compiled_result,"steady_state_speedup":eager_result["median_ms"]/compiled_result["median_ms"],"dynamo_counters":counters,"recompilation_indicator":recompiles,"selection":"do_not_enable_automatically"}
    (output/"torch_compile_benchmark.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2)); return 0
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True); p.add_argument("--manifest",required=True); p.add_argument("--fp32-audit",required=True); p.add_argument("--warmup-steps",type=int,default=3); p.add_argument("--measure-steps",type=int,default=10); p.add_argument("--mode",default="reduce-overhead"); p.add_argument("--record-observed-failure",action="store_true"); p.add_argument("--output-dir",required=True); return run(p.parse_args())
if __name__=="__main__": raise SystemExit(main())
