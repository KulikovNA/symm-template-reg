import csv,json,tempfile,unittest
from pathlib import Path
from symm_template_reg.evaluation.joint_stage import check_joint_stage

class JointGateTest(unittest.TestCase):
    def test_any_bad_metric_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            r=Path(td); (r/"checkpoints").mkdir(); (r/"best_evaluation").mkdir(); audit=r/"audit.json"; audit.write_text('{"target_leakage_detected": false}')
            (r/"resolved_config.json").write_text(json.dumps({"target_leakage_policy":{"audit_path":str(audit)}})); (r/"checkpoints/best_metrics.json").write_text('{"epoch": 1}')
            fields=["sample_id","rotation_error_deg","translation_total_mm","correspondence_p95_mm","visible_alignment_p95_mm","predicted_to_template_surface_p95_mm","procrustes_rank_valid","effective_correspondence_fraction"]
            with (r/"best_evaluation/per_sample_metrics.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(zip(fields,["bad",3,1,1,1,1,True,1])))
            self.assertFalse(check_joint_stage(r)["stage_passed"])
