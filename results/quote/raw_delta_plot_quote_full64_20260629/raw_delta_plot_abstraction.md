# Raw-Delta PLOT Abstraction Runs

This version uses raw effect signatures. Each coordinate is exactly `phi(y_swap) - phi(y_base)` for one base/source intervention.

`phi` for the neural model is the binary output margin. `phi` for the abstract model is the signed class output, so the abstract raw delta is `source_sign - base_sign`.

## Hard-Handle Raw Replay

### Quote

- source JSON: `eval\openai_sparse_plot\unmatched_quote_abstraction_csp_yolo1_template\unmatched_quote_abstraction.json`

| method | rank | handle | weight | cost | cosine sim | heldout same | heldout flip | heldout wrong-preserve |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `raw_squared_uot` | 1 | `attention_value_channel` | 0.083 | 6.615 | n/a | 1.000 | 0.000 | 1.000 |
| `raw_squared_uot` | 2 | `attention_value_write` | 0.083 | 114.225 | n/a | 1.000 | 0.125 | 0.667 |
| `raw_squared_uot` | 3 | `query_only_control` | 0.083 | 124.749 | n/a | 1.000 | 0.000 | 1.000 |
| `raw_squared_uot` | 4 | `detector_routing_control` | 0.083 | 124.774 | n/a | 1.000 | 0.000 | 1.000 |
| `raw_squared_uot` | 5 | `quote_detector_mass_control` | 0.083 | 126.275 | n/a | 1.000 | 0.000 | 1.000 |
| `raw_cosine_uot` | 1 | `opening_quote_detectors` | 0.131 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_uot` | 2 | `stored_and_attention_read` | 0.131 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_uot` | 3 | `stored_quote_type` | 0.131 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_uot` | 4 | `full_quote_type_path` | 0.131 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_uot` | 5 | `full_interpreted_12` | 0.131 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_similarity` | 1 | `opening_quote_detectors` | 0.116 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_similarity` | 2 | `stored_and_attention_read` | 0.116 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_similarity` | 3 | `stored_quote_type` | 0.116 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_similarity` | 4 | `full_quote_type_path` | 0.115 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |
| `raw_cosine_similarity` | 5 | `full_interpreted_12` | 0.115 | 0.002 | 0.998 | 1.000 | 1.000 | 0.000 |

## Singleton Soft-Handle Raw Runs

### Quote

- candidate singleton sites: `64`
- kept quote pairs: `16`

Top singleton sites by raw-delta coupling:

| method | rank | site | weight | cost | cosine sim |
|---|---:|---|---:|---:|---:|
| `raw_squared_uot` | 1 | `10.attn.v:663` | 0.826 | 4.614 | n/a |
| `raw_squared_uot` | 2 | `0.mlp.act_in:205` | 0.003 | 6.929 | n/a |
| `raw_squared_uot` | 3 | `2.mlp.post_act:1076` | 0.003 | 7.294 | n/a |
| `raw_squared_uot` | 4 | `0.mlp.act_in:207` | 0.003 | 11.845 | n/a |
| `raw_squared_uot` | 5 | `10.attn.resid_delta:83` | 0.003 | 20.184 | n/a |
| `raw_squared_uot` | 6 | `final_resid:728` | 0.003 | 63.001 | n/a |
| `raw_squared_uot` | 7 | `0.mlp.act_in:180` | 0.003 | 85.545 | n/a |
| `raw_squared_uot` | 8 | `10.attn.q:657` | 0.003 | 93.682 | n/a |
| `raw_cosine_uot` | 1 | `0.mlp.resid_delta:460` | 0.102 | 0.002 | 0.998 |
| `raw_cosine_uot` | 2 | `final_resid:83` | 0.101 | 0.003 | 0.997 |
| `raw_cosine_uot` | 3 | `0.mlp.post_act:2790` | 0.097 | 0.010 | 0.990 |
| `raw_cosine_uot` | 4 | `final_resid:728` | 0.090 | 0.021 | 0.979 |
| `raw_cosine_uot` | 5 | `10.attn.v:663` | 0.090 | 0.022 | 0.978 |
| `raw_cosine_uot` | 6 | `0.mlp.post_act:863` | 0.089 | 0.024 | 0.976 |
| `raw_cosine_uot` | 7 | `10.attn.act_in:460` | 0.084 | 0.033 | 0.967 |
| `raw_cosine_uot` | 8 | `0.mlp.act_in:205` | 0.083 | 0.036 | 0.964 |
| `raw_cosine_similarity` | 1 | `0.mlp.resid_delta:460` | 0.059 | 0.002 | 0.998 |
| `raw_cosine_similarity` | 2 | `final_resid:83` | 0.059 | 0.003 | 0.997 |
| `raw_cosine_similarity` | 3 | `0.mlp.post_act:2790` | 0.058 | 0.010 | 0.990 |
| `raw_cosine_similarity` | 4 | `final_resid:728` | 0.058 | 0.021 | 0.979 |
| `raw_cosine_similarity` | 5 | `10.attn.v:663` | 0.058 | 0.022 | 0.978 |
| `raw_cosine_similarity` | 6 | `0.mlp.post_act:863` | 0.058 | 0.024 | 0.976 |
| `raw_cosine_similarity` | 7 | `10.attn.act_in:460` | 0.057 | 0.033 | 0.967 |
| `raw_cosine_similarity` | 8 | `0.mlp.act_in:205` | 0.057 | 0.036 | 0.964 |

Calibrated soft handles:

| method | calibration | K | strength | calibration raw cost | behavior score | heldout raw cosine cost | heldout same | heldout flip | heldout wrong-preserve | sites |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `raw_squared_uot` | `raw_cost` | 8 | 1.000 | 4.348 | n/a | 0.012 | 1.000 | 0.000 | 1.000 | `10.attn.v:663, 0.mlp.act_in:205, 2.mlp.post_act:1076, 0.mlp.act_in:207, 10.attn.resid_delta:83, final_resid:728, 0.mlp.act_in:180, 10.attn.q:657` |
| `raw_cosine_uot` | `raw_cost` | 1 | 1.000 | 0.002 | n/a | 0.010 | 1.000 | 1.000 | 0.000 | `0.mlp.resid_delta:460` |
| `raw_cosine_similarity` | `raw_cost` | 1 | 1.000 | 0.002 | n/a | 0.010 | 1.000 | 1.000 | 0.000 | `0.mlp.resid_delta:460` |
| `raw_squared_uot` | `behavior` | 1 | 4.000 | 1401.181 | 0.969 | 0.003 | 1.000 | 1.000 | 0.000 | `10.attn.v:663` |
| `raw_cosine_uot` | `behavior` | 1 | 4.000 | 0.004 | 0.991 | 0.002 | 1.000 | 1.000 | 0.000 | `0.mlp.resid_delta:460` |
| `raw_cosine_similarity` | `behavior` | 1 | 4.000 | 0.004 | 0.991 | 0.002 | 1.000 | 1.000 | 0.000 | `0.mlp.resid_delta:460` |
