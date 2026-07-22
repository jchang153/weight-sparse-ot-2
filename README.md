# openai-plot

`openai-plot` studies causal abstractions in OpenAI's public weight-sparse
transformers using PLOT-style effect-signature matching and activation
interventions. The repository currently contains experiments for quote closing,
binary bracket closing, causal-site ablation, and progressive discovery of an
upstream bracket-depth representation.

The model checkpoints and OpenAI's `circuit_sparsity` implementation are not
stored in this repository. The standalone reproduction bundles record their
download URLs, byte counts, source revision, and SHA-256 checksums.

## Main results

### Quote closing

The full 64-site quote experiment supports the abstraction
\(X \to U \to Y\), where \(U\) is the unmatched opening-quote type. The
certified neural handle is `0.mlp.resid_delta:460`. Richer pointer-routing and
generic copy hypotheses are retained as diagnostics rather than certified
models.

Start with [the quote report](releases/quote/REPORT_QUOTE.md), or inspect the
[frozen quote results](results/quote/).

### Bracket closing

The full 133-site bracket experiment first supports the binary abstraction
\(X \to R \to Y\), where \(R\) distinguishes one-close from two-close
behavior. The refined binary handle is `4.attn.resid_delta:1079`; the earlier
multi-site and staged-chain results are retained with their stated caveats.

Start with [the bracket report](releases/bracket/REPORT_BRACKET.md), or inspect
the [frozen bracket results](results/bracket/).

### Ablation and progressive discovery

Ablating the certified quote and bracket sites substantially reduces task
accuracy. Searching the remaining localized sites does not recover a certified
replacement under continuous mean clamping. Progressive PLOT identifies a
graded active-depth state and supports the refined model
\(X \to D \to R \to Y\), with upstream depth writes at
`2.attn.resid_delta:1249` and `3.attn.resid_delta:1249`, a depth read at
`4.attn.act_in:1249`, and the binary threshold at
`4.attn.resid_delta:1079`.

See the [ablation and progressive report](releases/ablation-progressive/REPORT.md).

## Repository layout

| Path | Purpose |
|---|---|
| `src/experiments/openai_sparse_plot/` | Canonical experiment implementation. The historical Python import path `experiments.openai_sparse_plot` is preserved. |
| `tests/openai_sparse_plot/` | Offline unit tests for the canonical implementation. |
| `results/quote/` | Frozen quote inventories, scans, diagnostics, and selected results. |
| `results/bracket/` | Frozen bracket inventories, scans, diagnostics, and selected results. |
| `results/combined/` | Reports that compare or jointly select across tasks. |
| `releases/quote/` | Self-contained quote reproduction snapshot with pinned evidence. |
| `releases/bracket/` | Self-contained bracket reproduction snapshot with pinned evidence. |
| `releases/ablation-progressive/` | Self-contained ablation and progressive-discovery snapshot. |
| `docs/` | Repository-level methods and result documentation. |

The directories under `releases/` are frozen standalone snapshots. They
intentionally contain source and data needed to reproduce their associated
reports, even when that material overlaps with the canonical package. New
development should normally happen under `src/`, with new frozen outputs placed
under the appropriate task in `results/`.

## Local setup

The canonical package supports Python 3.11 and 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

OpenAI's external implementation must be available separately for model-backed
runs:

```bash
git clone https://github.com/openai/circuit_sparsity.git .external/circuit_sparsity
git -C .external/circuit_sparsity checkout --detach dbf1fe0d27b76c19e10d2a715f28c2e5da535e08
export CIRCUIT_SPARSITY_HOME="$PWD/.external/circuit_sparsity"
```

Model acquisition instructions and verified checksums live in each release's
`MODEL_ARTIFACTS.json` and README. The two public checkpoints are large, so the
offline audits below are the recommended first step.

## Offline verification

Run the canonical unit suite:

```bash
python -m unittest discover tests/openai_sparse_plot
```

Audit each frozen release without loading a transformer:

```bash
PYTHONPATH=releases/quote/src python -m quote_repro audit
PYTHONPATH=releases/bracket/src python -m bracket_repro audit
PYTHONPATH=releases/ablation-progressive/src python -m sparse_circuit_repro audit
```

The release-specific READMEs document complete model-backed reruns:

- [Quote reproduction](releases/quote/README.md)
- [Bracket reproduction](releases/bracket/README.md)
- [Ablation and progressive reproduction](releases/ablation-progressive/README.md)

## Reproducibility conventions

- Candidate universes contain all 64 exported quote sites or all 133 exported
  bracket sites unless a report explicitly documents a staged exclusion.
- Effect signatures use the raw output difference
  \(\phi(y_{\mathrm{swap}})-\phi(y_{\mathrm{base}})\).
- Fit, calibration, and held-out splits have distinct roles; held-out data is
  not used for selector calibration.
- Failed signatures, unsuccessful redundancy searches, and diagnostic-only
  models remain available alongside accepted results.
- Frozen release manifests verify the exact files used by their offline audits.

For the original full-site result summary, see
[docs/full-site-plot.md](docs/full-site-plot.md). Historical documentation from
the repository's earlier scaffold is retained under `docs/archive/` and is not
part of the current `openai-plot` workflow.
