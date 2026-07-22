# Archived OpenAI Sparse PLOT development notes

> This file preserves the original development log and historical paths. Use
> the repository root `README.md` for the current layout and commands.

This package is the first implementation layer for applying PLOT to OpenAI's
released weight-sparse GPT-style circuits, starting with `csp_yolo1` and
`single_double_quote`.

The package intentionally does not vendor OpenAI's `circuit_sparsity` source.
Install it separately or set `CIRCUIT_SPARSITY_HOME` to a local checkout.

## Current Reproduction Target

The public OpenAI blob layout discovered locally is:

```text
https://openaipublic.blob.core.windows.net/circuit-sparsity/viz/csp_yolo1/single_double_quote/prune_v2/64/viz_data.pt
```

The `64` artifact exports a raw retained circuit with 64 scalar rows and 46
nonzero `pair_data` edges. This is not yet the paper-level 12-node/9-edge
human summary; it is the first faithful local export from the released
visualizer payload.

## First Local Smoke

```powershell
python -m py_compile experiments/openai_sparse_plot/*.py
python -m unittest discover experiments/openai_sparse_plot/tests
python -m experiments.openai_sparse_plot.run_inventory `
  --circuit-home C:\tmp\circuit_sparsity `
  --try-first-candidate `
  --out-dir eval/openai_sparse_plot/string_closing_prune_v2_64
```

To run the current CPU model smoke:

```powershell
python -m experiments.openai_sparse_plot.run_model_smoke `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/model_smoke_csp_yolo1
```

OpenAI's raw `load_model` currently fails on the public `csp_yolo1`
`beeg_config.json` because it includes `bigram_table_rank`, which is absent
from the checked-out `GPTConfig` constructor. `runtime.py` provides a narrow
compatibility loader that filters unsupported training-only config keys and
records the dropped keys in the smoke output.

The inventory command without `--viz-path` and without `--try-first-candidate`
is non-networking. To force a specific artifact, run:

```powershell
python -m experiments.openai_sparse_plot.run_inventory `
  --viz-path <local-or-blob-viz-data-path> `
  --model csp_yolo1 `
  --task single_double_quote `
  --out-dir eval/openai_sparse_plot/string_closing_inventory
```

Expected outputs:

- `inventory.md`
- `inventory.json`
- `string_closing_circuit.json` when a viz artifact is loaded
- `string_closing_circuit_nodes.csv` when a viz artifact is loaded
- `string_closing_circuit_edges.csv` when a viz artifact is loaded

## Current PLOT / Scrub Smoke

After the inventory, model smoke, interpreted subcircuit export, and activation
smoke have passed, run the first local PLOT stage:

```powershell
python -m experiments.openai_sparse_plot.run_effect_signatures `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/effect_signatures_simple_chain_csp_yolo1_pairs8 `
  --max-pairs 8 `
  --min-abs-margin 1.0

python -m experiments.openai_sparse_plot.run_plot_matching `
  --table-json eval/openai_sparse_plot/effect_signatures_simple_chain_csp_yolo1_pairs8/effect_signature_table.json `
  --out-dir eval/openai_sparse_plot/plot_matching_simple_chain_csp_yolo1_pairs8

python -m experiments.openai_sparse_plot.run_scrub_validation `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/scrub_validation_csp_yolo1_pairs8 `
  --max-pairs 8 `
  --min-abs-margin 1.0 `
  --max-same-records-per-site 24 `
  --max-different-records-per-site 24

python -m experiments.openai_sparse_plot.run_group_validation `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/group_validation_simple_chain_csp_yolo1_pairs8 `
  --max-pairs 8 `
  --min-abs-margin 1.0 `
  --max-same-records-per-group 24 `
  --max-different-records-per-group 24

python -m experiments.openai_sparse_plot.run_bootstrap_matching `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/bootstrap_matching_simple_chain_csp_yolo1_b20_pairs8 `
  --max-pairs 8 `
  --sample-pairs 8 `
  --bootstrap-samples 20 `
  --min-abs-margin 1.0 `
  --seed 0

python -m experiments.openai_sparse_plot.run_candidate_model_sweep `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/candidate_model_sweep_csp_yolo1_pairs8 `
  --max-pairs 8 `
  --min-abs-margin 1.0

python -m experiments.openai_sparse_plot.run_calibrated_top_models `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/calibrated_top_models_csp_yolo1 `
  --min-abs-margin 1.0

python -m experiments.openai_sparse_plot.run_position_routing_diagnostic `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/position_routing_diagnostic_csp_yolo1

python -m experiments.openai_sparse_plot.run_nonquote_route_value_diagnostic `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/nonquote_route_value_diagnostic_csp_yolo1

python -m experiments.openai_sparse_plot.run_unmatched_quote_abstraction `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/unmatched_quote_abstraction_csp_yolo1_template `
  --max-pairs 16 `
  --max-records-per-relation 8 `
  --min-abs-margin 1.0

python -m experiments.openai_sparse_plot.run_artifact_sample_audit `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/artifact_sample_audit_csp_yolo1_prune_v2_64

python -m experiments.openai_sparse_plot.run_faithfulness_audit `
  --out-dir eval/openai_sparse_plot/faithfulness_audit_csp_yolo1_prune_v2_64

python -m experiments.openai_sparse_plot.run_restricted_circuit_eval `
  --circuit-home C:\tmp\circuit_sparsity `
  --out-dir eval/openai_sparse_plot/restricted_circuit_eval_csp_yolo1_pairs8 `
  --max-pairs 8
```

Main outputs:

- `eval/openai_sparse_plot/effect_signatures_simple_chain_csp_yolo1_pairs8/effect_signature_table.md`
- `eval/openai_sparse_plot/plot_matching_simple_chain_csp_yolo1_pairs8/plot_matching.md`
- `eval/openai_sparse_plot/scrub_validation_csp_yolo1_pairs8/scrub_validation.md`
- `eval/openai_sparse_plot/group_validation_simple_chain_csp_yolo1_pairs8/group_validation.md`
- `eval/openai_sparse_plot/bootstrap_matching_simple_chain_csp_yolo1_b20_pairs8/bootstrap_matching.md`
- `eval/openai_sparse_plot/candidate_model_sweep_csp_yolo1_pairs8/candidate_model_sweep.md`
- `eval/openai_sparse_plot/calibrated_top_models_csp_yolo1/calibrated_top_models.md`
- `eval/openai_sparse_plot/position_routing_diagnostic_csp_yolo1/position_routing_diagnostic.md`
- `eval/openai_sparse_plot/position_routing_diagnostic_csp_yolo1/p_plot_position_interpretation.md`
- `eval/openai_sparse_plot/nonquote_route_value_diagnostic_csp_yolo1/nonquote_route_value_diagnostic.md`
- `eval/openai_sparse_plot/unmatched_quote_abstraction_csp_yolo1_template/unmatched_quote_abstraction.md`
- `eval/openai_sparse_plot/artifact_sample_audit_csp_yolo1_prune_v2_64/artifact_sample_audit.md`
- `eval/openai_sparse_plot/faithfulness_audit_csp_yolo1_prune_v2_64/faithfulness_audit.md`
- `eval/openai_sparse_plot/restricted_circuit_eval_csp_yolo1_pairs8/restricted_circuit_eval.md`
- `eval/openai_sparse_plot/reproduction_report.md`

Current interpretation: the primary task-level SCM for `single_double_quote`
should be the single-variable abstraction `X -> U -> Y`, where `U` is the
unmatched opening quote type and `Y` is the matching closing quote. PLOT/UOT is
used only as a sparse selector for candidate neural handles implementing `U`.
On a template-heldout run, calibration templates `assign, print` select the
expected early quote-type handles; heldout templates `handler_arg, paren_assign`
show same-`U` preservation `1.000`, opposite-`U` flip `1.000`, and wrong-variable
preservation `0.000` for the selected top handles. The position-routing and
non-quote diagnostics explain why we do not use `P` or generic token copying as
the main causal abstraction.

The strongest causal-scrubbing results are the opening detector pair
`0.mlp.post_act:863/2790` and storage channel `0.mlp.resid_delta:460`:
same-quote resampling preserves output sign and different-quote resampling
flips it in the current capped sample. The grouped `full_quote_type_path` also
flips under different-quote resampling. Downstream quote-copy/output sites move
the binary margin in the right direction, and the `output_preference` group
flips often but not always.

The candidate-model sweep compares eight causal abstractions. The smallest
accurate full-coverage model is `m7_internal_path_supernode_3`: opening quote
detectors -> internal quote-type path supernode -> observed output. It has 100%
internal strict IIA and 100% coverage of the canonical released circuit nodes.
Smaller behavior-only models also reach 100% internal strict IIA, but cover
only 57.1% of the circuit because they omit the downstream copy/write path.

The calibrated top-model sweep reruns the three best full-coverage models
under a template-heldout split. Calibration uses `assign` and `print` prompt
pairs; heldout uses `paren_assign` and `handler_arg` prompt pairs. It sweeps
192 stage-aware PLOT hyperparameter configurations per model. The selected
configuration for all three top models is centered cosine cost, epsilon 0.05,
neural beta 0.1, and stage penalty 1.0. `m5` and `m7` tie on heldout matching
and internal IIA, but `m7` remains the cleaner paper-facing abstraction because
the final residual readout is treated as observed output rather than included
inside the internal path.

The position-routing diagnostic is the first step toward a more faithful causal
model with `P = opener_position(X)`, `Q = quote_type(X_P)`, and
`Y = closing_quote(Q)`. It reconstructs layer-10 head-82 attention from Q/K
activations and checks whether the full attention row from the final position
selects the true opener. Current result: full head-82 attention is a plausible
`P` object on simple, one-distractor, and opposite-distractor prompts; the
single QK channel 657 alone is not sufficient under distractors.

The artifact task-sample audit evaluates the loaded full model on OpenAI's
released 32 `task_samples` from the same `viz_data.pt` artifact. Rows must be
right-trimmed because token id `0` is used as padding even though it decodes to
a printable tinypython token. With that handling, the model matches the paired
quote label on 31/32 samples; the only miss has near-zero binary quote margin.

The released `viz_data` artifact reports pruned-circuit losses, but this local
implementation has not yet instantiated an executable pruned model. A naive
activation-channel mask over retained nodes was tested and is not faithful
enough, so strict original-vs-pruned faithfulness remains open.
