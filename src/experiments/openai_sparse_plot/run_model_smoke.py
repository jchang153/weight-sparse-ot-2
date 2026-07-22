from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


DEFAULT_PROMPTS = (
    'x = "hello',
    "x = 'hello",
    'print("abc',
    "print('abc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small csp_yolo1 string-closing smoke test.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/model_smoke"))
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = make_tinypython_encoding(args.circuit_home)
    token_ids = quote_token_ids(enc)
    model, info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )

    rows = []
    with torch.no_grad():
        for prompt in DEFAULT_PROMPTS:
            ids = torch.tensor(enc.encode(prompt), dtype=torch.long).unsqueeze(0)
            if args.cuda:
                ids = ids.cuda()
            logits, _, _ = model(ids)
            last = logits[0, -1]
            single_logit = float(last[token_ids["single"]])
            double_logit = float(last[token_ids["double"]])
            margin = double_logit - single_logit
            rows.append(
                {
                    "prompt": prompt,
                    "decoded": enc.decode(ids[0].detach().cpu().tolist()),
                    "n_tokens": int(ids.shape[1]),
                    "single_logit": single_logit,
                    "double_logit": double_logit,
                    "double_minus_single": margin,
                    "predicted_closing_quote": "double" if margin > 0 else "single",
                }
            )

    payload = {
        "model_info": info,
        "quote_token_ids": token_ids,
        "rows": rows,
    }
    (args.out_dir / "model_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# csp_yolo1 Model Smoke", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- dropped config keys: `{', '.join(info['dropped_config_keys'])}`")
    lines.append(f"- missing state keys: `{', '.join(info['missing_keys'])}`")
    lines.append(f"- unexpected state keys: `{', '.join(info['unexpected_keys'])}`")
    lines.append(f"- quote token ids: `{token_ids}`")
    lines.extend(["", "## Prompt Results", ""])
    for row in rows:
        lines.append(
            "- `{prompt}` -> `{pred}` "
            "(double_minus_single={margin:.4f})".format(
                prompt=row["prompt"],
                pred=row["predicted_closing_quote"],
                margin=row["double_minus_single"],
            )
        )
    (args.out_dir / "model_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
