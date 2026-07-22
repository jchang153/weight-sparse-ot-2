from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from experiments.openai_sparse_plot.runtime import (
    load_sparse_gpt_model,
    make_tinypython_encoding,
    quote_token_ids,
)

from .common import release_root


def run(*, cuda: bool) -> dict[str, object]:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    model, info = load_sparse_gpt_model(
        model_name="csp_yolo1",
        circuit_home=circuit,
        cuda=cuda,
        flash=False,
        grad_checkpointing=False,
    )
    enc = make_tinypython_encoding(circuit)
    quote_ids = quote_token_ids(enc)
    examples = (
        ("x = \"abc", "double"),
        ("x = 'abc", "single"),
    )
    records: list[dict[str, object]] = []
    for prompt, expected in examples:
        ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
        if cuda:
            ids = ids.cuda()
        with torch.no_grad():
            logits, _, _ = model(ids)
        single_logit = float(logits[0, -1, quote_ids["single"]])
        double_logit = float(logits[0, -1, quote_ids["double"]])
        predicted = "double" if double_logit > single_logit else "single"
        records.append(
            {
                "prompt": prompt,
                "expected_quote_type": expected,
                "predicted_quote_type": predicted,
                "single_quote_logit": single_logit,
                "double_quote_logit": double_logit,
                "finite": bool(torch.isfinite(logits[0, -1, [quote_ids["single"], quote_ids["double"]]]).all()),
                "correct": predicted == expected,
            }
        )
    payload = {
        "model": info["model_name"],
        "records": records,
        "all_finite": all(bool(record["finite"]) for record in records),
        "all_correct": all(bool(record["correct"]) for record in records),
    }
    out = root / "outputs" / "model_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_finite"] or not payload["all_correct"]:
        raise RuntimeError("quote smoke failed clean quote-type inference")
    return payload
