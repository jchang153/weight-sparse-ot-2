# Quote-Type Causal Abstraction in an OpenAI Sparse Circuit

## Result At A Glance

We study OpenAI's public `csp_yolo1` sparse transformer on prompts that contain an unmatched single or double quote. The model must predict the matching closing quote.

The strict PLOT experiment uses every one of the 64 sites in the exported localized circuit. It certifies the model

```text
X -> U -> Y
```

where `U` is the unmatched quote type. PLOT selects the singleton neural handle `0.mlp.resid_delta:460`. Heldout same-variable preservation, different-variable flipping, and wrong-variable preservation are `1.000`, `1.000`, and `0.000`, respectively. The final number is supposed to be zero: when quote type changes, preserving unrelated prompt properties must not prevent the output from changing.

A richer position-and-copy account is not certified. The attention diagnostic often finds the opener, but it is imperfect and does not implement a general token-copy operation.

## Task And Variables

| Symbol | Meaning |
|---|---|
| `X` | Entire tokenized prompt. |
| `Y` | Next closing-quote token predicted by the model. |
| `U` | Type of the currently unmatched quote: single or double. |
| `P` | Token position of the currently unmatched opening quote. |
| `Q` | Quote type read from `X` at position `P`: `Q = X_P`, restricted to single/double quote. |

Running example:

```text
X = handler(prefix, ("hello
P = position 9
Q = double quote
U = double quote
Y = "
```

`P` is a token index, not a character index. The exact number depends on the released TinyPython tokenizer.

## Causal Models

### Q0: Direct Baseline

**Status: not a causal explanation.**

```text
X -> Y
```

The only equation is `Y = model(X)`. This predicts behavior but has no internal variable on which to intervene. It therefore cannot explain why replacing one small internal state transfers quote type between prompts.

### Q1: Unmatched Quote Type

**Status: certified.**

```text
X -> U -> Y
```

Equations:

```text
U = unmatched_quote_type(X)
Y = matching_close(U)
```

For `print("hello`, `U=double` and `Y="`. For `print('hello`, `U=single` and `Y='`.

Intervention: take a base prompt and a source prompt, then replace the candidate neural site's base activation with a soft source-minus-base update. If source and base have opposite `U`, a valid handle must change the base output to the source quote. If they have the same `U`, it must preserve the output.

### Q2: Position Then Type Extraction

**Status: not certified; diagnostic hypothesis only.**

```text
X -> P
(X, P) -> Q
Q -> Y
```

Equations:

```text
P = position of the currently unmatched opening quote
Q = X_P, restricted to quote type
Y = matching_close(Q)
```

This model is stronger than Q1. It says the circuit first chooses a position and then reads the value stored there. We test `P` by inspecting whether the proposed attention head ranks the active opener first. We test the copy interpretation by forcing the same route to carry values from quote and nonquote positions.

The diagnostic components are predeclared rather than discovered by the strict Q1 PLOT run: layer-10 attention head 82, query/key channel `657`, and value channel `663`. The position test compares the full head with its channel-657 proxy; the copy test replaces head 82's final-position output with a selected source-position value, either for the full head or channel 663 alone. These choices test the richer Q2 hypothesis and are not a filtered candidate set for Q1.

The full head ranks the target opener first on `14/18 = 0.778` diagnostic prompts. It is perfect on the simple single-opener family but fails on some same-quote and multiple-distractor cases. When forced to route a code token or a content token, the expected nonquote token appears in the model's top five outputs at rate `0.000`. Thus these experiments do not validate a general `P` followed by token-copy `Q=X_P` abstraction.

## Strict PLOT Protocol For Q1

Candidate neural sites are all 64 rows of `data/quote_circuit_nodes.csv`; no site family is removed. For each source/base pair:

```text
abstract signature = quote_code(source) - quote_code(base)
quote_code(single) = -1
quote_code(double) = +1

neural signature(site)
  = quote_margin(output after patching site from source into base)
    - quote_margin(clean base output)

quote_margin = logit(double quote) - logit(single quote)
```

The main matcher is cosine-cost one-sided unbalanced optimal transport with `epsilon=0.08` and `beta=0.08`. Calibration tests `K in {1,2,3,5,8}` and strength `lambda in {0.5,1,2,4}`. Dfit constructs signatures, Dcal chooses `K` and `lambda`, and Dte is used only after selection.

Selected configuration:

```text
site     = 0.mlp.resid_delta:460
K        = 1
lambda   = 4
```

## Certification

| Heldout requirement | Result | Required |
|---|---:|---:|
| Same `U` preserves base output | 1.000 | >= 0.90 |
| Opposite `U` changes output to source | 1.000 | >= 0.90 |
| Same nuisance but opposite `U` preserves base | 0.000 | <= 0.10 |

The last controls keep position, content, or content length fixed while changing quote type. A valid `U` intervention should still change output, so base-output preservation should be near zero.

## Reproduction

The ZIP does not contain model weights. It uses OpenAI's public implementation at [github.com/openai/circuit_sparsity](https://github.com/openai/circuit_sparsity), pinned to commit `dbf1fe0d27b76c19e10d2a715f28c2e5da535e08`.

The public model files are:

```text
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo1/beeg_config.json
https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo1/final_model.pt
```

After setup:

```text
python -m quote_repro audit
python -m quote_repro run --experiment certified --cuda
python -m quote_repro run --experiment pointer-copy --cuda
python -m quote_repro report
```

`audit` uses the included compact evidence and does not load the transformer. `run` downloads or reuses the public checkpoint and recomputes model interventions.

## Interpretation And Limits

The evidence cleanly supports an internal unmatched-quote-type variable. It does not establish that the localized circuit implements a general pointer machine or a general token-copy mechanism. The pointer/copy experiments are explicitly declared mechanistic diagnostics; they are not part of the strict all-64-site Q1 discovery.
