from __future__ import annotations

import argparse
from pathlib import Path

from .effect_signatures import (
    build_effect_prompt_pairs,
    build_effect_signature_table,
    write_effect_signature_table,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build source/ablation effect signatures for OpenAI sparse PLOT.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/effect_signatures"))
    parser.add_argument("--max-pairs", type=int, default=2)
    parser.add_argument("--min-abs-margin", type=float, default=0.0)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    token_ids = quote_token_ids(enc)
    model, _ = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)
    pairs = build_effect_prompt_pairs(max_pairs=args.max_pairs)
    table, diagnostics = build_effect_signature_table(
        model,
        enc,
        pairs=pairs,
        single_token_id=token_ids["single"],
        double_token_id=token_ids["double"],
        device=device,
        min_abs_margin=args.min_abs_margin,
    )
    write_effect_signature_table(table, diagnostics, out_dir=args.out_dir)
    print(
        f"wrote effect signature table to {args.out_dir} "
        f"abstract={len(table.abstract_variable_ids)} neural={len(table.neural_site_ids)} "
        f"features={len(table.feature_names)} kept_pairs={table.metadata['kept_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
