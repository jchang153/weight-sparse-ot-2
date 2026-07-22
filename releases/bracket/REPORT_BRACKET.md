# Binary Bracket Readout and Staged Mediation in an OpenAI Sparse Circuit

## Result At A Glance

We study OpenAI's public `csp_yolo2` sparse transformer on a code-completion task. The model emits either one closing bracket or two closing brackets. It does not emit an arbitrary number of closes on this task distribution.

The strict full-133-site PLOT run certifies a binary variable `R`:

```text
X -> R -> Y
```

A refined experiment finds a perfect intermediate handle `R_mid = 4.attn.resid_delta:1079` and strong downstream mediation evidence. However, no tested `R_late` handle passes every invariance and blocking requirement. Consequently, the complete chain `X -> R_mid -> R_late -> Y` remains strong but not certified.

## Task And Variables

| Symbol | Meaning |
|---|---|
| `X` | Entire bracket-completion prompt. |
| `D` | Number of currently active unmatched opening brackets. This describes the prompt but is not assumed to be represented as a separate causal variable. |
| `R` | Saturated binary readout: one-close when `D=1`, two-close when `D>=2`. |
| `R_mid` | Intermediate-layer version of the binary `R` decision. |
| `R_late` | Proposed later version of the same binary decision, closer to output. |
| `Y` | Output class: `]` or `]]`. |
| `T2` | Diagnostic bit `1[D>=2]`, used by the frozen readout to measure the internal binary state. |

Equations on natural prompts:

```text
R = T2 = 1[D >= 2]
Y = ]  if R=0
Y = ]] if R=1
```

Examples:

```text
one active opener:       D=1, R=0, Y=]
three nested openers:    D=3, R=1, Y=]]
```

`R` is therefore not an arbitrary counter. Depths 2, 3, and 4 all have the same natural output class.

## Causal Models

### B0: Coarse Binary Readout

**Status: certified.**

```text
X -> R -> Y
```

Equations:

```text
R = 1[active_depth(X) >= 2]
Y = close_class(R)
```

Intervention: patch a candidate neural handle from a source prompt into a base prompt. A valid `R` handle changes output when source and base are on opposite sides of the depth threshold and preserves output when they have the same `R`.

Strict all-133-site handle:

```text
final_resid:1079
7.mlp.post_act:4133
7.mlp.resid_delta:2041
```

Heldout same/flip/wrong-preserve is `1.000/1.000/0.000`.

### B1: Pure Staged Chain

**Status: strong evidence, not certified.**

```text
X -> R_mid -> R_late -> Y
```

Natural-run equations are:

```text
R_mid = 1[D>=2]
R_late = R_mid
Y = close_class(R_late)
```

`R_mid` and `R_late` have the same value naturally, but they are different causal variables because interventions can set one while holding the other fixed. PLOT finds:

```text
R_mid = 4.attn.resid_delta:1079
```

This handle passes every balanced heldout sensitivity and invariance gate at `1.000`.

The best tested downstream handle is:

```text
7.mlp.act_in:1079 + 4.attn.q:1292
```

It sets output to the source class in both directions, but it preserves the base `T2` state only `0.880` for one-to-two interventions. A clean `R_late` variable should change output while leaving the upstream `T2` measurement unchanged. That requirement fails.

The chain blocking test patches `R_mid` from source while restoring `R_late` to base. Under a pure chain, this must restore base output. It succeeds at `1.000` for one-to-two but only `0.529` for two-to-one. The pure chain is therefore not certified with this downstream handle.

### B2: Staged Chain With A Bypass

**Status: diagnostic evidence, not certified.**

```text
X -> R_mid -> R_late -> Y
       \--------------> Y
```

This model adds a path by which `R_mid` can affect `Y` without passing through the tested `R_late` variable. Its equations can be written as:

```text
R_late = f(R_mid, X)
Y = g(R_late, R_mid, X)
```

The controlled residual fraction after fixing `R_late` is `0.207` for one-to-two and `0.249` for two-to-one. Every one of 24 heldout base-content clusters has a positive residual in both directions.

That is evidence for missing downstream computation or an incomplete `R_late` handle. It is not enough to certify the bypass graph: a valid independent `R_late` variable must first be isolated.

## PLOT Protocols

### Coarse B0

Every one of the 133 exported circuit sites is a singleton candidate. For each source/base pair:

```text
abstract signature = R(source) - R(base)

neural signature(site)
  = close_margin(output after patching site from source into base)
    - close_margin(clean base output)

close_margin = logit(]]) - logit(])
```

Matching uses cosine-cost one-sided UOT with `epsilon=0.08`, `beta=0.08`, followed by top-K and strength calibration.

### Joint R_mid/R_late Search

The refined experiment still uses all 133 singleton candidates and does not search handpicked pairs or triples. It fits one `2 x 133` coupling. For each record:

```text
R_mid abstract row  = [delta T2, delta P(]), delta P(]])]
R_late abstract row = [0,        delta P(]), delta P(]])]
```

The neural row for a site contains the corresponding changes after patching that site. `R_mid` is allowed to move the internal `T2` state and output. `R_late` is expected to move output without moving the upstream `T2` state.

The declared sweep is:

```text
epsilon = 0.02, 0.08, 0.32
beta    = 0.02, 0.08, 0.32
K       = 1, 2, 3, 5, 8
lambda  = 0.5, 1, 2, 4
```

Dfit constructs signatures and fits the frozen `T2` readout. Dcal chooses the transport setting, top-K support, and strength. A separate balanced Dcal rechecks direct and blocking gates. Dte is disjoint and is used only after all choices are frozen.

## Final Certification Table

| Claim | Status | Decisive evidence |
|---|---|---|
| `X -> R -> Y` | Certified | Full-133 PLOT handle; heldout `1.000/1.000/0.000`. |
| `X -> R_mid -> Y` | Certified | Every balanced direct gate is `1.000`. |
| `X -> R_mid -> R_late -> Y` | Not certified | `R_late` upstream-state preservation is `0.880`; reverse blocking restores base output only `0.529`. |
| Chain plus `R_mid -> Y` bypass | Not certified | Residual is robust, but `R_late` is not independently validated. |

The correct conclusion is that the circuit has a validated intermediate binary readout and substantial downstream mediation compatible with a staged chain. The presently tested downstream handle is incomplete.

## Reproduction

The ZIP does not contain model weights. It uses OpenAI's public implementation at [github.com/openai/circuit_sparsity](https://github.com/openai/circuit_sparsity), pinned to commit `dbf1fe0d27b76c19e10d2a715f28c2e5da535e08`.

The public model files are:

```text
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo2/beeg_config.json
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo2/final_model.pt
```

The coarse rerun also uses OpenAI's public task input:

```text
https://openaipublic.blob.core.windows.net/circuit-sparsity/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt
```

`MODEL_ARTIFACTS.json` records the byte count and SHA256 checksum of all three files. The setup command downloads and verifies them automatically.

After setup:

```text
python -m bracket_repro audit
python -m bracket_repro run --experiment coarse --cuda
python -m bracket_repro run --experiment chain --cuda --resume
python -m bracket_repro report
```

`audit` recomputes the compact evidence without loading the transformer. The chain run is checkpointed because it is much more expensive than the audit.
