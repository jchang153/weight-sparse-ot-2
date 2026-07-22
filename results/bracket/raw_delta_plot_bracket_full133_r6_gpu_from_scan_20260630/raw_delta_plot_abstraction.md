# Raw-Delta PLOT Abstraction Runs

This version uses raw effect signatures. Each coordinate is exactly `phi(y_swap) - phi(y_base)` for one base/source intervention.

`phi` for the neural model is the binary output margin. `phi` for the abstract model is the signed class output, so the abstract raw delta is `source_sign - base_sign`.

## Singleton Soft-Handle Raw Runs

### Bracket

- candidate singleton sites: `133`
- clean accuracy: `1.000`

Top singleton sites by raw-delta coupling:

| method | rank | site | weight | cost | cosine sim |
|---|---:|---|---:|---:|---:|
| `raw_squared_uot` | 1 | `final_resid:1079` | 0.953 | 1.101 | n/a |
| `raw_squared_uot` | 2 | `7.mlp.post_act:6561` | 0.047 | 1.582 | n/a |
| `raw_squared_uot` | 3 | `7.mlp.post_act:4133` | 0.000 | 3.016 | n/a |
| `raw_squared_uot` | 4 | `7.mlp.resid_delta:1200` | 0.000 | 6.006 | n/a |
| `raw_squared_uot` | 5 | `final_resid:1200` | 0.000 | 6.066 | n/a |
| `raw_squared_uot` | 6 | `7.mlp.post_act:2511` | 0.000 | 18.193 | n/a |
| `raw_squared_uot` | 7 | `final_resid:607` | 0.000 | 22.070 | n/a |
| `raw_squared_uot` | 8 | `final_resid:431` | 0.000 | 26.298 | n/a |
| `raw_cosine_uot` | 1 | `final_resid:1079` | 0.049 | 0.003 | 0.997 |
| `raw_cosine_uot` | 2 | `7.mlp.post_act:4133` | 0.047 | 0.008 | 0.992 |
| `raw_cosine_uot` | 3 | `7.mlp.resid_delta:2041` | 0.047 | 0.009 | 0.991 |
| `raw_cosine_uot` | 4 | `final_resid:2041` | 0.047 | 0.009 | 0.991 |
| `raw_cosine_uot` | 5 | `7.mlp.post_act:6561` | 0.047 | 0.009 | 0.991 |
| `raw_cosine_uot` | 6 | `final_resid:607` | 0.047 | 0.010 | 0.990 |
| `raw_cosine_uot` | 7 | `4.attn.resid_delta:1079` | 0.047 | 0.011 | 0.989 |
| `raw_cosine_uot` | 8 | `7.mlp.act_in:1079` | 0.047 | 0.011 | 0.989 |
| `raw_cosine_similarity` | 1 | `final_resid:1079` | 0.030 | 0.003 | 0.997 |
| `raw_cosine_similarity` | 2 | `7.mlp.post_act:4133` | 0.030 | 0.008 | 0.992 |
| `raw_cosine_similarity` | 3 | `7.mlp.resid_delta:2041` | 0.030 | 0.009 | 0.991 |
| `raw_cosine_similarity` | 4 | `final_resid:2041` | 0.030 | 0.009 | 0.991 |
| `raw_cosine_similarity` | 5 | `7.mlp.post_act:6561` | 0.030 | 0.009 | 0.991 |
| `raw_cosine_similarity` | 6 | `final_resid:607` | 0.030 | 0.010 | 0.990 |
| `raw_cosine_similarity` | 7 | `4.attn.resid_delta:1079` | 0.030 | 0.011 | 0.989 |
| `raw_cosine_similarity` | 8 | `7.mlp.act_in:1079` | 0.030 | 0.011 | 0.989 |

Calibrated soft handles:

| method | calibration | K | strength | calibration raw cost | behavior score | heldout raw cosine cost | heldout same | heldout flip | heldout wrong-preserve | sites |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `raw_squared_uot` | `raw_cost` | 2 | 1.000 | 0.984 | n/a | 0.002 | 1.000 | 0.000 | 1.000 | `final_resid:1079, 7.mlp.post_act:6561` |
| `raw_cosine_uot` | `raw_cost` | 1 | 0.500 | 0.003 | n/a | 0.002 | 1.000 | 0.000 | 1.000 | `final_resid:1079` |
| `raw_cosine_similarity` | `raw_cost` | 1 | 0.500 | 0.003 | n/a | 0.002 | 1.000 | 0.000 | 1.000 | `final_resid:1079` |
| `raw_squared_uot` | `behavior` | 2 | 4.000 | 435.636 | 0.701 | 0.002 | 1.000 | 0.500 | 0.500 | `final_resid:1079, 7.mlp.post_act:6561` |
| `raw_cosine_uot` | `behavior` | 3 | 4.000 | 0.005 | 0.969 | 0.003 | 1.000 | 1.000 | 0.000 | `final_resid:1079, 7.mlp.post_act:4133, 7.mlp.resid_delta:2041` |
| `raw_cosine_similarity` | `behavior` | 3 | 4.000 | 0.005 | 0.969 | 0.003 | 1.000 | 1.000 | 0.000 | `final_resid:1079, 7.mlp.post_act:4133, 7.mlp.resid_delta:2041` |
