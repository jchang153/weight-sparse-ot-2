# Strict Full-Site Causal Model Selection

This report uses only the corrected protocol: full exported site family and canonical raw output deltas `phi(y_swap) - phi(y_base)`.

## Protocol

- quote candidate sites: `64` from `node_csv`
- bracket candidate sites: `133`
- effect signature: `phi(y_swap) - phi(y_base)`
- excluded from this strict report: stage-aware penalties, handpicked site-family narrowing, internal/progressive signatures

## Quote

Accepted model: `Q1: X -> U -> Y`, where `U` is unmatched quote type.

- accepted raw cosine-UOT handle: `0.mlp.resid_delta:460`
- heldout same: `1.000`
- heldout flip: `1.000`
- heldout wrong-preserve: `0.000`

Full-site raw cosine-UOT top ranks:

| rank | site | weight | cost |
|---:|---|---:|---:|
| 1 | `0.mlp.resid_delta:460` | 0.101937 | 0.002166 |
| 2 | `final_resid:83` | 0.101241 | 0.003263 |
| 3 | `0.mlp.post_act:2790` | 0.096796 | 0.010445 |
| 4 | `final_resid:728` | 0.090352 | 0.021470 |
| 5 | `10.attn.v:663` | 0.089916 | 0.022242 |
| 6 | `0.mlp.post_act:863` | 0.088821 | 0.024203 |
| 7 | `10.attn.act_in:460` | 0.084275 | 0.032609 |
| 8 | `0.mlp.act_in:205` | 0.082679 | 0.035669 |
| 9 | `2.mlp.post_act:1076` | 0.081348 | 0.038265 |
| 10 | `10.attn.resid_delta:83` | 0.075062 | 0.051133 |
| 11 | `0.mlp.act_in:207` | 0.070178 | 0.061897 |
| 12 | `7.attn.resid_delta:787` | 0.003182 | 0.556849 |

Interpretation: the richer pointer/copy model `Q2` is unsupported under the strict protocol. Route/readout sites appear in the full-site ranking, but the calibrated heldout handle is the simpler singleton `U` handle, and canonical output deltas do not provide a separate position variable `P` signal.

## Bracket

Accepted model: `B0: X -> R -> Y`, where `R` is the late saturated close-count readout.

- accepted raw cosine-UOT handle: `final_resid:1079, 7.mlp.post_act:4133, 7.mlp.resid_delta:2041`
- heldout same: `1.000`
- heldout flip: `1.000`
- heldout wrong-preserve: `0.000`

Full-site raw cosine-UOT top ranks:

| rank | site | weight | cost |
|---:|---|---:|---:|
| 1 | `final_resid:1079` | 0.048984 | 0.003284 |
| 2 | `7.mlp.post_act:4133` | 0.047430 | 0.008444 |
| 3 | `7.mlp.resid_delta:2041` | 0.047400 | 0.008544 |
| 4 | `final_resid:2041` | 0.047348 | 0.008718 |
| 5 | `7.mlp.post_act:6561` | 0.047266 | 0.008998 |
| 6 | `final_resid:607` | 0.046971 | 0.010000 |
| 7 | `4.attn.resid_delta:1079` | 0.046731 | 0.010819 |
| 8 | `7.mlp.act_in:1079` | 0.046589 | 0.011304 |
| 9 | `7.mlp.resid_delta:607` | 0.045850 | 0.013863 |
| 10 | `4.attn.act_in:1249` | 0.044217 | 0.019666 |
| 11 | `7.mlp.post_act:2511` | 0.041038 | 0.031605 |
| 12 | `7.mlp.post_act:3082` | 0.040915 | 0.032085 |

Interpretation: `B1`/`B2` are not validated under the strict protocol. Multi-depth same-`R` different-`D` cases do not create an output-delta signal for `D`; finding `D` would require an explicitly approved internal/progressive signature experiment.

Historical context: the earlier multidepth run selected `B0`, but it was not strict because it used a 12-site `D` candidate list and an internal depth/R-probe signature. I am not using it as evidence for this strict report.

