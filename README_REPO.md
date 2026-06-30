# das-addition

Minimal, debuggable pipeline for addition-with-carry causal abstraction alignment.

## Quickstart
1. Install deps:

```bash
python -m pip install -r requirements.txt
```

2. Run the swap plumbing smoke test:

```bash
python scripts/swap_test.py
```

3. Train the tiny model on addition:

```bash
python scripts/train.py --steps 5000 --batch-size 256 --eval-every 200 --target-acc 0.995
```

4. Evaluate accuracy on held-out random samples:

```bash
python scripts/eval.py --batches 20 --batch-size 512
```

Checkpoints are written to `./models` by default.

Important:
- After the objective/labeling fix, old checkpoints are no longer representative.
- Retrain before running GW alignment.

## GW Causal Abstraction Pipeline (starter)
This repo now includes a first-pass GW pipeline for addition-with-carry that does:
1. Abstract carry-variable swaps on base/source pairs.
2. Neural site swaps over candidate `(layer, position)` sites.
3. Effect-signature construction.
4. Neural EGW alignment (semi-dual neural estimator).
5. Hard and soft-handle evaluation on held-out pairs.

Run a small smoke version:

```bash
python scripts/gw_pipeline.py --cpu --train-pairs 4 --calib-pairs 4 --test-pairs 4 --max-sites 8 --egw-outer-steps 10 --egw-inner-steps 4
```

Run a larger pass:

```bash
python scripts/gw_pipeline.py --cpu --train-pairs 24 --calib-pairs 24 --test-pairs 24 --max-sites 40 --egw-outer-steps 120 --egw-inner-steps 30
```

Output JSON is written to `eval/gw_pipeline_result.json` (or `--output` path).

Soft-handle knobs:

```bash
python scripts/gw_pipeline.py --soft-top-k 5 --soft-lambda 1.0
```

Recommended starting point (more stable than top-k=5 on current setup):

```bash
python scripts/gw_pipeline.py --soft-top-k 2 --soft-lambda 0.5
```

Advanced GW tuning knobs:

```bash
python scripts/gw_pipeline.py --neural-component block_output --hard-refine-top-m 12 --soft-tune-topk-grid 1,2,3 --soft-tune-lambda-grid 0.1,0.25,0.5,0.75 --calib-pairs 24 --test-pairs 24
```

High-performance GW hard-handle run (expensive on CPU):

```bash
python scripts/gw_pipeline.py --neural-component block_output --egw-epsilon 0.1 --coupling-temperature 0.7 --soft-top-k 1 --soft-lambda 0.25 --hard-refine-top-m 60 --calib-pairs 24 --test-pairs 24 --output eval/gw_pipeline_result_refine60_block.json
```

Plot results:

```bash
python scripts/plot_gw_results.py --input eval/gw_pipeline_result.json --out-dir eval/plots
```

Run hyperparameter sweep (epsilon, soft top-k, soft lambda):

```bash
python scripts/gw_sweep.py --soft-topk-grid 1,2,3 --soft-lambda-grid 0.25,0.5,1.0 --egw-eps-grid 0.1,0.2,0.5 --coupling-temp-grid 0.7,1.0
```

Plot sweep results:

```bash
python scripts/plot_gw_sweep.py --input eval/gw_sweep_result.json --out-dir eval/plots_sweep
```

Run DAS baseline (gradient-based rotated subspace interventions with site sweep):

```bash
python scripts/das_pipeline.py --checkpoint models/tiny_gpt2_addition.pt --train-pairs 24 --calib-pairs 24 --test-pairs 24 --das-subspace-dim 16 --das-coarse-steps 20 --das-fine-steps 80 --das-shortlist-k 8
```

Run OT-DAS baseline (entropic OT coupling + the same handle/eval stack):

```bash
python scripts/ot_pipeline.py --checkpoint models/tiny_gpt2_addition.pt --train-pairs 24 --calib-pairs 24 --test-pairs 24 --ot-epsilon 0.1 --ot-max-iters 400
```

Compare GW vs DAS:

```bash
python scripts/compare_gw_das.py --gw eval/gw_pipeline_result.json --das eval/das_pipeline_result.json --out-dir eval/plots_compare
```

Comprehensive multi-run comparison report (baseline + tuned GW/DAS):

```bash
python scripts/report_gw_das.py --gw-baseline eval/gw_pipeline_result_strict_baseline.json --gw-optimized eval/gw_pipeline_result_strict_tuned.json --das-baseline eval/das_pipeline_result_strict_baseline.json --das-tuned eval/das_pipeline_result_strict_tuned.json --out-dir eval/plots_report_strict
```

Strict protocol notes:
- `train-pairs`: used for fitting alignments/interventions.
- `calib-pairs`: used for handle/site selection and hyperparameter tuning.
- `test-pairs`: untouched held-out final evaluation only.

Publication-style multi-seed protocol (shared component + shared site budget + CI bars):

```bash
python scripts/pub_protocol.py --seeds 0,1,2 --component mlp_output --site-budget 60 --out-dir eval/pub_protocol_mlp
```

Three-way publication-style protocol (GW-DAS vs OT-DAS vs DAS):

```bash
python scripts/pub_protocol_threeway.py --seeds 0,1,2 --component mlp_output --site-budget 60 --out-dir eval/pub_protocol_threeway_mlp --reuse-gw-das-dir eval/pub_protocol_mlp
```

Protocol outputs:
- Per-seed run JSONs in `eval/pub_protocol_mlp/runs/`
- Aggregate report `eval/pub_protocol_mlp/pub_protocol_report.md`
- Aggregate plot `eval/pub_protocol_mlp/pub_protocol_overall.png`
- Aggregate manifest `eval/pub_protocol_mlp/pub_protocol_results.json`
- Three-way report `eval/pub_protocol_threeway_mlp/threeway_report.md`
- Three-way plots `eval/pub_protocol_threeway_mlp/threeway_baseline_overall.png`, `eval/pub_protocol_threeway_mlp/threeway_tuned_overall.png`, `eval/pub_protocol_threeway_mlp/threeway_tuned_per_variable.png`
- Three-way manifest `eval/pub_protocol_threeway_mlp/threeway_results.json`

## DAS-paper-style Task Comparison (HEQ + MoNLI-style)
This script runs a fair shared protocol on two additional tasks inspired by DAS experimental themes:
- `HEQ` (hierarchical-equality style) with an MLP.
- `MoNLI-style` (synthetic negation + lexical relation) with a tiny transformer.

Both GW and DAS use:
- the same trained model per task/seed,
- the same candidate site set,
- disjoint train/calibration/test pair pools,
- calibration-only model selection and test-only final reporting.

Run:

```bash
python scripts/das_paper_benchmarks.py --cpu --seeds 0,1 --train-pairs 24 --calib-pairs 24 --test-pairs 48 --site-budget 24 --heq-hidden-dim 256 --heq-epochs 30 --heq-train-examples 8000 --heq-val-examples 2000 --heq-pair-pool 2000 --monli-epochs 10 --monli-train-examples 8000 --monli-val-examples 2000 --monli-pair-pool 2200 --gw-outer-steps 20 --gw-inner-steps 5 --gw-refine-top-m 6 --das-coarse-steps 2 --das-fine-steps 6 --das-shortlist-k 6 --out-dir eval/das_paper_tasks
```

Outputs:
- `eval/das_paper_tasks/benchmark_summary.json`
- `eval/das_paper_tasks/benchmark_report.md`
- `eval/das_paper_tasks/task_comparison_overview.png`

## Locked Protocol Orchestration (Official-First)
The repo now includes protocol configs and orchestration scripts for a reproducible, decision-gated workflow:

- `configs/protocols/official_lock_v1.yaml`
- `configs/protocols/threeway_lock_v1.yaml`
- `scripts/run_protocol.py`
- `scripts/aggregate_protocol_results.py`
- `scripts/check_protocol_integrity.py`
- `scripts/harvest_official_anchor.py`
- `scripts/official_delta_report.py`

### Run a protocol

Three-way pilot (3 seeds from config):

```bash
python scripts/run_protocol.py --protocol threeway_lock_v1 --seed-stage pilot --run-id threeway_pilot_v1
```

Three-way final (10 seeds from config):

```bash
python scripts/run_protocol.py --protocol threeway_lock_v1 --seed-stage final --run-id threeway_final_v1
```

Official DAS reproduction (task-filtered example):

```bash
python scripts/run_protocol.py --protocol official_lock_v1 --task arithmetic --run-id official_arithmetic_v1
```

Resume an interrupted run:

```bash
python scripts/run_protocol.py --protocol threeway_lock_v1 --run-id threeway_pilot_v1 --resume
```

### Aggregate and test protocol artifacts

```bash
python scripts/aggregate_protocol_results.py --run-dir eval/protocol_runs/threeway_pilot_v1 --baseline-method das
python scripts/check_protocol_integrity.py --run-dir eval/protocol_runs/threeway_pilot_v1 --allow-unknown-split --allow-missing-site-count
```

### Harvest arithmetic anchor from RedCloud

```bash
python scripts/harvest_official_anchor.py --run-id official_arithmetic_anchor_v1 --wait-for-oracle --key "$HOME/.ssh/redcloud_bot"
```

### Build official-vs-observed delta table

```bash
python scripts/official_delta_report.py --protocol-yaml configs/protocols/official_lock_v1.yaml --run-dir eval/protocol_runs/official_arithmetic_anchor_v1
```

