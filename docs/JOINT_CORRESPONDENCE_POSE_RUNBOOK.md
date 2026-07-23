# Joint correspondence → uniform Procrustes debug runbook

Это controlled overfit на test split одного `scene_000000/fragment_0002`, а не финальная оценка обобщения. Команды никогда автоматически не переходят к следующему числу views.

## Контракт baseline

Сеть получает только visible shell `p_i^C` и template surface. Она предсказывает surface-constrained `q_i^O`, после чего одна pose вычисляется float32 `WeightedProcrustes` с весами `w_i=1/N_valid`. Direct/residual pose heads, ranking, regions и learned confidence в основных configs отсутствуют.

Для каждого общего symmetry element `S` считается

`L(S) = 1.0 Lcorr + 1.0 Lrot + 1.0 Ltrans + 0.5 Lalign + 0.25 Lsurface`,

и весь набор компонентов использует один `argmin_S L(S)`. Масштабы: correspondence 0.002 m, rotation 2°, translation 0.002 m, alignment 0.002 m, surface 0.001 m.

## 1. Общие проверки и manifests

```bash
cd /home/nikita/disser/fragment-template-registration-lab/symm-template-reg

export WD=/home/nikita/disser/fragment-template-registration-lab/work_dirs
export JM=$WD/manifests/joint_correspondence_pose
export TAG=$(date +%Y%m%d_%H%M%S)

python -m compileall -q symm_template_reg tools configs tests
python -m unittest discover -s tests -q

# Выполняйте только если $JM ещё не существует: tool отказывается перезаписывать папку.
python tools/build_joint_view_manifests.py \
  --manifest "$WD/manifests/single_fragment_scene000000_fragment0002_de68591bf9a5.json" \
  --output-dir "$JM"
```

Ожидаются `fragment0002_views02.json`, `views04.json`, `views08.json` с кадрами `(4,8) ⊂ (4,5,2,8) ⊂ (4,5,2,8,0,1,6,9)`.

## 2. Pre-training audits для 2 views

```bash
export LOSS_AUDIT=$WD/joint_loss_contract_audit_$TAG
export LEAK_AUDIT=$WD/joint_target_leakage_audit_$TAG

python tools/audit_joint_loss_contract.py \
  --config configs/debug/joint_correspondence_pose_v2/01_uniform_joint_2views.py \
  --manifest "$JM/fragment0002_views02.json" \
  --device cuda \
  --output-dir "$LOSS_AUDIT"

python tools/audit_target_leakage.py \
  --config configs/debug/joint_correspondence_pose_v2/01_uniform_joint_2views.py \
  --manifest "$JM/fragment0002_views02.json" \
  --device cuda \
  --output-dir "$LEAK_AUDIT"
```

Оба процесса должны завершиться с `audit_passed=true`. Обучение запрещено при другом результате.

## 3. Stage 2 views

```bash
python tools/run_view_ladder.py \
  --config configs/debug/joint_correspondence_pose_v2/01_uniform_joint_2views.py \
  --manifest "$JM/fragment0002_views02.json" \
  --seeds 0 \
  --device cuda \
  --output-dir "$WD/joint_uniform_views02_$TAG" \
  --cfg-options "target_leakage_policy.audit_path=$LEAK_AUDIT/target_leakage_audit.json"
```

Скопируйте точный внутренний `run_dir`, напечатанный командой:

```bash
export RUN02='<printed-2view-run-dir>'
python tools/check_joint_stage_readiness.py --run-dir "$RUN02"
python tools/package_joint_stage_report.py \
  --run-dir "$RUN02" \
  --output "$WD/joint_uniform_views02_report_$TAG.tar.gz"
```

**STOP. Не запускайте 4 views, пока `stage_gate.json` не содержит `stage_passed=true` и архив 2-view не проанализирован.**

## 4. Stage 4 views — только после успешного 2-view gate

Сначала выполните target-leakage audit для 4-view manifest и задайте `LEAK04` на его папку.

```bash
export TAG04=$(date +%Y%m%d_%H%M%S)
export LEAK04=$WD/joint_target_leakage_views04_$TAG04
python tools/audit_target_leakage.py \
  --config configs/debug/joint_correspondence_pose_v2/02_uniform_joint_4views.py \
  --manifest "$JM/fragment0002_views04.json" --device cuda --output-dir "$LEAK04"

python tools/run_view_ladder.py \
  --config configs/debug/joint_correspondence_pose_v2/02_uniform_joint_4views.py \
  --manifest "$JM/fragment0002_views04.json" --seeds 0 --device cuda \
  --output-dir "$WD/joint_uniform_views04_$TAG04" \
  --cfg-options "target_leakage_policy.audit_path=$LEAK04/target_leakage_audit.json"

export RUN04='<printed-4view-run-dir>'
python tools/check_joint_stage_readiness.py --run-dir "$RUN04"
python tools/package_joint_stage_report.py --run-dir "$RUN04" \
  --output "$WD/joint_uniform_views04_report_$TAG04.tar.gz"
```

**STOP. Не запускайте 8 views до успешного 4-view gate и анализа архива.**

## 5. Stage 8 views — только после успешного 4-view gate

```bash
export TAG08=$(date +%Y%m%d_%H%M%S)
export LEAK08=$WD/joint_target_leakage_views08_$TAG08
python tools/audit_target_leakage.py \
  --config configs/debug/joint_correspondence_pose_v2/03_uniform_joint_8views.py \
  --manifest "$JM/fragment0002_views08.json" --device cuda --output-dir "$LEAK08"

python tools/run_view_ladder.py \
  --config configs/debug/joint_correspondence_pose_v2/03_uniform_joint_8views.py \
  --manifest "$JM/fragment0002_views08.json" --seeds 0 --device cuda \
  --output-dir "$WD/joint_uniform_views08_$TAG08" \
  --cfg-options "target_leakage_policy.audit_path=$LEAK08/target_leakage_audit.json"

export RUN08='<printed-8view-run-dir>'
python tools/check_joint_stage_readiness.py --run-dir "$RUN08"
python tools/package_joint_stage_report.py --run-dir "$RUN08" \
  --output "$WD/joint_uniform_views08_report_$TAG08.tar.gz"
```

Future learned-confidence ablation разрешена только после `RUN08/stage_gate.json: stage_passed=true`; она не входит в этот runbook.

## Что ожидать и что передавать на анализ

Каждый sample получает ровно 1500 exposures. Evaluation выполняется каждые 25 эпох, visualizations — на 0, 250, 500, 750, 1000, 1250, 1500. Best выбирается по `eval/physical_normalized_score`; gate требует для каждого sample ≤2°/2 mm pose, correspondence p95 ≤2 mm, alignment p95 ≤2 mm, surface p95 ≤1 mm, valid rank, uniform effective fraction 1 и отсутствие leakage/NaN.

После каждого stage передавайте только архив report. При визуальной проблеме добавьте из одного sample:

- `pred_vs_gt_camera_frame.ply`;
- `correspondences_on_template.ply`;
- `reconstructed_visible_shell.ply`;
- `symmetry_hypotheses_gallery.ply`;
- `visualization_summary.json`.

Ключевые compact-файлы находятся в `best_evaluation/`, `checkpoints/best_metrics.json`, `history/`, `stage_gate.json`, `final_summary.json`. Не нужно присылать все PLY и checkpoint.
