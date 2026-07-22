# OpenAI Localized Bracket-Circuit PLOT Reproduction

This release reproduces the certified binary bracket readout and the balanced audit of the proposed `R_mid -> R_late` chain. Start with [REPORT_BRACKET.md](REPORT_BRACKET.md).

## Quick Audit

```text
python -m pip install -e .
python -m bracket_repro audit
python -m unittest discover tests
```

## Full Model Run

Windows PowerShell:

```text
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m bracket_repro run --experiment coarse --cuda
.\.venv\Scripts\python.exe -m bracket_repro run --experiment chain --cuda --resume
```

Linux:

```text
bash scripts/setup.sh
.venv/bin/python -m bracket_repro run --experiment coarse --cuda
.venv/bin/python -m bracket_repro run --experiment chain --cuda --resume
```

The setup downloads `csp_yolo2` and the public bracket-task `viz_data.pt` from OpenAI, verifies every checksum in `MODEL_ARTIFACTS.json`, and runs a clean-inference smoke test. These public binary assets are intentionally not stored in this ZIP. The chain run is checkpointed and substantially more expensive than the offline audit.
