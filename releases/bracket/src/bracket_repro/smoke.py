from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from experiments.openai_sparse_plot.runtime import load_sparse_gpt_model, make_tinypython_encoding

from .common import release_root


def run(*, cuda: bool) -> dict[str, object]:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    model, info = load_sparse_gpt_model(
        model_name="csp_yolo2",
        circuit_home=circuit,
        cuda=cuda,
        flash=True,
        grad_checkpointing=False,
    )
    enc = make_tinypython_encoding(circuit)
    prompt = "values = [[1, 9, 3, 11"
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
    if cuda:
        ids = ids.cuda()
    with torch.no_grad():
        logits, _, _ = model(ids)
    one = int(enc.encode("]\n")[0])
    two = int(enc.encode("]]\n")[0])
    payload = {
        "model": info["model_name"],
        "prompt": prompt,
        "tokens": int(ids.shape[1]),
        "one_close_logit": float(logits[0, -1, one]),
        "two_close_logit": float(logits[0, -1, two]),
        "finite": bool(torch.isfinite(logits[0, -1, [one, two]]).all()),
    }
    out = root / "outputs" / "model_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["finite"]:
        raise RuntimeError("bracket smoke produced non-finite logits")
    return payload
