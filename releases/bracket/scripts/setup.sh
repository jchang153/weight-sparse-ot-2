#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TORCH_VARIANT="${TORCH_VARIANT:-cu121}"
python3 -m venv .venv
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install torch==2.5.1 --index-url "https://download.pytorch.org/whl/$TORCH_VARIANT"
"$PYTHON" -m pip install -r requirements-lock.txt
"$PYTHON" -m pip install -e .
test -d .external/circuit_sparsity || git clone https://github.com/openai/circuit_sparsity.git .external/circuit_sparsity
git -C .external/circuit_sparsity checkout --detach dbf1fe0d27b76c19e10d2a715f28c2e5da535e08
"$PYTHON" -m bracket_repro download-model
CUDA_FLAG="--cuda"; test "$TORCH_VARIANT" = cpu && CUDA_FLAG=""
"$PYTHON" -m bracket_repro run --experiment chain --smoke $CUDA_FLAG
