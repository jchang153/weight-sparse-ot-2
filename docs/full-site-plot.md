# OpenAI Sparse PLOT Full-Site Bundle

This bundle is the corrected share copy for the OpenAI localized-circuit PLOT runs.

Important implementation point:
- Quote uses the full exported 64-site family from `results/quote/string_closing_prune_v2_64/string_closing_circuit_nodes.csv` by default.
- Bracket uses the full exported 133-site family from `results/bracket/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv` by default.
- The effect signature is the canonical raw-delta signature
  \(\phi(y_{\mathrm{swap}}) - \phi(y_{\mathrm{base}})\).
- The PLOT procedure is raw-delta signatures, cosine one-sided UOT / cosine selectors, top-K and strength calibration, then heldout sensitivity/invariance validation.
- The old quote 12-site subset remains only as an explicit legacy/debug option: `--quote-candidate-source interpreted12`.

## Code Entry Points

- `src/experiments/openai_sparse_plot/run_raw_delta_plot_abstraction.py`
  - Quote defaults to full 64 sites via `--quote-candidate-source node_csv`.
  - Bracket defaults to all exported sites when `--bracket-max-sites 0`.
- `src/experiments/openai_sparse_plot/run_bracket_raw_delta_scan.py`
  - Checkpointed full 133-site bracket raw-delta scan.
- `src/experiments/openai_sparse_plot/run_bracket_raw_delta_plot_from_scan.py`
  - Rebuilds bracket PLOT calibration/heldout validation from the completed full-site scan records.

## Included Result Artifacts

Quote full-site run:
- `results/quote/raw_delta_plot_quote_full64_20260629/raw_delta_plot_abstraction.json`
- `results/quote/raw_delta_plot_quote_full64_20260629/raw_delta_plot_abstraction.md`

Quote result:
- Candidate source: node CSV
- Candidate site count: 64
- Raw-delta definition:
  \(\phi(y_{\mathrm{swap}}) - \phi(y_{\mathrm{base}})\)
- Behavior-selected raw cosine-UOT handle: `0.mlp.resid_delta:460`
- Heldout behavior: same = 1.000, flip = 1.000, wrong-preserve = 0.000

Bracket full-site run:
- `results/bracket/bracket_full133_r6_gpu_checkpointed_20260630/bracket_raw_delta_scan.json`
- `results/bracket/bracket_full133_r6_gpu_checkpointed_20260630/singleton_records.jsonl`
- `results/bracket/raw_delta_plot_bracket_full133_r6_gpu_from_scan_20260630/raw_delta_plot_abstraction.json`
- `results/bracket/raw_delta_plot_bracket_full133_r6_gpu_from_scan_20260630/raw_delta_plot_abstraction.md`

Bracket result:
- Candidate site count: 133
- Raw-delta definition:
  \(\phi(y_{\mathrm{swap}}) - \phi(y_{\mathrm{base}})\)
- `max_records_per_relation`: 6
- Clean model behavior on released samples: 32/32
- Behavior-selected raw cosine-UOT handle: `final_resid:1079`, `7.mlp.post_act:4133`, `7.mlp.resid_delta:2041`
- Heldout behavior: same = 1.000, flip = 1.000, wrong-preserve = 0.000

Bracket diagnostic nuance:
- `results/bracket/bracket_refined_causal_diagnostic_r6_lambda4_exact/bracket_refined_causal_diagnostic.md`
- The no-final pair `7.mlp.post_act:4133 + 7.mlp.resid_delta:2041` also validates, while `final_resid:1079` alone fails the wrong-variable control. So `final_resid:1079` should be described as readout-only/optional, not as the whole bracket mechanism.

## Validation Run Before Packaging

From the repo root:

```powershell
.\.venv_sparse_plot\Scripts\python.exe -m py_compile src\experiments\openai_sparse_plot\run_raw_delta_plot_abstraction.py src\experiments\openai_sparse_plot\run_bracket_raw_delta_plot_from_scan.py
.\.venv_sparse_plot\Scripts\python.exe -m unittest discover tests\openai_sparse_plot
```

Result: 55 tests passed.

## Dependencies Not Included

The external OpenAI sparse-circuit dependency and model/cache data are not bundled here:
- `.external/circuit_sparsity`
- local virtual environment
- full generated eval tree outside the selected full-site artifacts
## Strict Full-Site Model-Selection Report

Also included:
- `results/combined/strict_fullsite_model_selection_20260630/strict_fullsite_model_selection.md`
- `results/combined/strict_fullsite_model_selection_20260630/strict_fullsite_model_selection.json`

This report uses only the full exported site families and canonical raw output
deltas. It accepts \(Q_1: X \to U \to Y\) for quote and
\(B_0: X \to R \to Y\) for bracket. It does not use the earlier non-strict
candidate-model sweeps or the earlier 12-site/internal-signature multidepth run
as evidence.
