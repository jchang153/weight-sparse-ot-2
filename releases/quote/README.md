# OpenAI Localized Quote-Circuit PLOT Reproduction

This release reproduces the certified `X -> U -> Y` quote-type abstraction and the unsupported richer position/type-extraction hypothesis. Start with [REPORT_QUOTE.md](REPORT_QUOTE.md).

## Quick Audit

The audit uses included records and does not load the transformer:

```text
python -m pip install -e .
python -m quote_repro audit
python -m unittest discover tests
```

## Full Model Run

Windows PowerShell:

```text
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m quote_repro run --experiment certified --cuda
.\.venv\Scripts\python.exe -m quote_repro run --experiment pointer-copy --cuda
```

Linux:

```text
bash scripts/setup.sh
.venv/bin/python -m quote_repro run --experiment certified --cuda
.venv/bin/python -m quote_repro run --experiment pointer-copy --cuda
```

The setup downloads `csp_yolo1` from OpenAI. The checkpoint is intentionally not stored in this ZIP.
