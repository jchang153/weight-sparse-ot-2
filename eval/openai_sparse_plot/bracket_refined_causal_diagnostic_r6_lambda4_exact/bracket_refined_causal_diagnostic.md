# Bracket Refined Causal Diagnostic

Question: does raw-delta PLOT identify an internal depth variable, or mostly a late readout/output-margin variable?

Refined model under test:

```text
X -> D_mid -> R_late -> Y
```

- `D_mid`: internal parsed bracket-depth state.
- `R_late`: late residual/readout expression of the depth decision.
- released samples: `32`
- clean accuracy: `1.000`
- max records per relation: `6`

## Hard Internal-Depth Handles

| handle | calibration same | calibration flip | calibration wrong-preserve | heldout same | heldout flip | heldout wrong-preserve | heldout shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| `depth_path_1249` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 11.602 |
| `late_depth_readout_7_mlp_post` | 1.000 | 0.500 | 0.500 | 1.000 | 0.500 | 0.500 | 7.599 |
| `late_depth_signal_core` | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 12.092 |
| `late_depth_state_7_mlp_input` | 1.000 | 1.000 | 0.125 | 1.000 | 1.000 | 0.000 | 9.978 |
| `layer1_control_1643` | 1.000 | 0.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.549 |

## Raw PLOT Handle Ablations

| variant | selected strength | calibration score | calibration same | calibration flip | calibration wrong-preserve | heldout score | heldout same | heldout flip | heldout wrong-preserve | heldout shift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `raw_plot_full_top3` | 4.000 | 0.969 | 1.000 | 1.000 | 0.000 | 0.975 | 1.000 | 1.000 | 0.000 | 10.895 |
| `raw_plot_no_final_top2` | 4.000 | 0.976 | 1.000 | 1.000 | 0.000 | 0.981 | 1.000 | 1.000 | 0.000 | 12.479 |
| `raw_plot_final_only` | 4.000 | 0.700 | 1.000 | 0.500 | 0.500 | 0.705 | 1.000 | 0.500 | 0.500 | 7.573 |

## Soft Variant Weights

- `raw_plot_full_top3`: `final_resid:1079`=0.341, `7.mlp.post_act:4133`=0.330, `7.mlp.resid_delta:2041`=0.330
- `raw_plot_no_final_top2`: `7.mlp.post_act:4133`=0.500, `7.mlp.resid_delta:2041`=0.500
- `raw_plot_final_only`: `final_resid:1079`=1.000
