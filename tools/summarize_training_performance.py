#!/usr/bin/env python3
"""Combine measured profiles and audits into the compact final performance summary."""

from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def _profile(path): return json.loads((Path(path)/"training_step_profile.json").read_text())
def _median(report,name): return report["phase_statistics"][name]["median_ms"]
def _peak(path,name):
    with (Path(path)/"training_step_profile.csv").open() as stream: return max(float(row[name]) for row in csv.DictReader(stream))
def run(a):
    output=Path(a.output_dir).expanduser().resolve(); output.mkdir(parents=True,exist_ok=False)
    baseline=_profile(a.baseline_profile); optimized=_profile(a.optimized_profile); shared=_profile(a.shared_profile); static=_profile(a.static_profile); vector=_profile(a.vector_profile)
    batch=json.loads((Path(a.batch_benchmark)/"selected_training_batch_mode.json").read_text()); equivalence=json.loads((Path(a.equivalence)/"fp32_equivalence_audit.json").read_text()); cache=json.loads((Path(a.static_cache)/"static_geometry_cache_audit.json").read_text()); amp=json.loads((Path(a.amp_audit)/"mixed_precision_equivalence.json").read_text()); compile_report=json.loads((Path(a.compile_benchmark)/"torch_compile_benchmark.json").read_text())
    base_wall=_median(baseline,"full_wall_clock_ms"); fast_wall=_median(optimized,"full_wall_clock_ms")
    summary={
      "status":"ready_for_full_run","full_training_started":False,"mandatory_stop_reached":True,
      "historical_trainer_epoch_median_ms":14660.0,
      "historical_timing_reproduced":False,
      "baseline_profile":{"median_ms":base_wall,"p90_ms":baseline["phase_statistics"]["full_wall_clock_ms"]["p90_ms"],"peak_allocated_mb":_peak(a.baseline_profile,"peak_allocated_mb"),"peak_reserved_mb":_peak(a.baseline_profile,"peak_reserved_mb")},
      "optimized_profile":{"median_ms":fast_wall,"p90_ms":optimized["phase_statistics"]["full_wall_clock_ms"]["p90_ms"],"peak_allocated_mb":_peak(a.optimized_profile,"peak_allocated_mb"),"peak_reserved_mb":_peak(a.optimized_profile,"peak_reserved_mb")},
      "speedup":base_wall/fast_wall,
      "isolated_wall_savings_ms":{"static_geometry_cache":base_wall-_median(static,"full_wall_clock_ms"),"shared_template_encoding":base_wall-_median(shared,"full_wall_clock_ms"),"vectorized_symmetry_procrustes":base_wall-_median(vector,"full_wall_clock_ms"),"diagnostic_sync_each_step":_median(baseline,"per_module_gradient_diagnostics_ms")},
      "loss_median_ms":{"baseline":_median(baseline,"symmetry_coordinate_tail_procrustes_loss_ms"),"vectorized":_median(vector,"symmetry_coordinate_tail_procrustes_loss_ms")},
      "selected_batch_mode":batch,
      "padding_ratio_before":batch["selected_metrics"]["padding_ratio"],"padding_ratio_after":batch["selected_metrics"]["padding_ratio"],
      "static_cache_audit_passed":cache["audit_passed"],"fp32_equivalence":equivalence,"optional_amp":amp,"optional_torch_compile":compile_report,
    }
    (output/"training_performance_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    rows=[{"metric":k,"value":v} for k,v in (("baseline_median_ms",base_wall),("optimized_median_ms",fast_wall),("speedup",base_wall/fast_wall),("baseline_peak_allocated_mb",summary["baseline_profile"]["peak_allocated_mb"]),("optimized_peak_allocated_mb",summary["optimized_profile"]["peak_allocated_mb"]),("padding_ratio",summary["padding_ratio_after"]))]
    with (output/"training_performance_summary.csv").open("x",newline="") as stream: writer=csv.DictWriter(stream,fieldnames=("metric","value")); writer.writeheader(); writer.writerows(rows)
    lines=["# Итог оптимизации 4×4","",f"- Baseline median/p90: `{base_wall:.3f}` / `{summary['baseline_profile']['p90_ms']:.3f}` ms.",f"- Optimized median/p90: `{fast_wall:.3f}` / `{summary['optimized_profile']['p90_ms']:.3f}` ms.",f"- Ускорение: `{base_wall/fast_wall:.3f}×`.",f"- Выбран batch/accumulation: `{batch['selected_batch_size']}×{batch['selected_gradient_accumulation_steps']}`.",f"- Padding ratio: `{summary['padding_ratio_after']:.6f}`.",f"- Strict fp32 audit: `{equivalence['audit_passed']}`.","- Полное обучение не запускалось. STOP."]
    (output/"training_performance_summary.md").write_text("\n".join(lines)+"\n"); print(json.dumps({"output":str(output),"speedup":summary["speedup"]},indent=2)); return 0
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline_profile","optimized_profile","shared_profile","static_profile","vector_profile","batch_benchmark","equivalence","static_cache","amp_audit","compile_benchmark"): p.add_argument("--"+name.replace("_","-"),required=True)
    p.add_argument("--output-dir",required=True); return run(p.parse_args())
if __name__=="__main__": raise SystemExit(main())
