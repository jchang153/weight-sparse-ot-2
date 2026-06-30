from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import DEFAULT_MODEL, DEFAULT_TASK, candidate_viz_paths, load_viz_data
from .effect_signatures import build_effect_prompt_pairs
from .interpreted_circuit import PAPER_BACKED_NODE_SPECS
from .restricted_circuit import (
    evaluate_restricted_margin_records,
    retained_masks_from_node_ids,
    retained_masks_from_viz_data,
    summarize_margin_records,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate activation-restricted circuits from OpenAI retained nodes.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--sweep", default="prune_v2")
    parser.add_argument("--k", default="64")
    parser.add_argument("--viz-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/restricted_circuit_eval"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _prompts_from_pairs(max_pairs: int) -> tuple[str, ...]:
    prompts = []
    for left, right in build_effect_prompt_pairs(max_pairs=max_pairs):
        prompts.extend([left.prompt, right.prompt])
    return tuple(prompts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    viz_path = args.viz_path
    if viz_path is None:
        viz_path = candidate_viz_paths(model=args.model, task=args.task, sweeps=(args.sweep,), ks=(args.k,))[0]
    print("loading artifact/model", flush=True)
    viz_data = load_viz_data(viz_path)
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    prompts = _prompts_from_pairs(args.max_pairs)

    full_retained_masks = retained_masks_from_viz_data(viz_data)
    interpreted_masks = retained_masks_from_node_ids(tuple(spec.node_id for spec in PAPER_BACKED_NODE_SPECS))

    full_records = evaluate_restricted_margin_records(
        model,
        enc,
        prompts,
        masks=full_retained_masks,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    interpreted_records = evaluate_restricted_margin_records(
        model,
        enc,
        prompts,
        masks=interpreted_masks,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    payload = {
        "model_info": model_info,
        "viz_path": viz_path,
        "prompts": len(prompts),
        "full_retained_mask_count": len(full_retained_masks),
        "full_retained_channel_count": sum(len(mask.retained_channels) for mask in full_retained_masks),
        "interpreted_mask_count": len(interpreted_masks),
        "interpreted_channel_count": sum(len(mask.retained_channels) for mask in interpreted_masks),
        "full_retained_summary": summarize_margin_records(full_records),
        "interpreted_summary": summarize_margin_records(interpreted_records),
        "full_retained_records": full_records,
        "interpreted_records": interpreted_records,
        "limitation": (
            "This is an activation-restricted circuit: hook activations outside retained channels are zeroed. "
            "It is not proven identical to OpenAI's reported weight-pruned/intervened circuit loss."
        ),
    }
    (args.out_dir / "restricted_circuit_eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Retained Activation Circuit Evaluation", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- viz artifact: `{viz_path}`")
    lines.append(f"- prompts: `{len(prompts)}`")
    lines.append(f"- full retained hooks: `{len(full_retained_masks)}`")
    lines.append(f"- full retained channels: `{payload['full_retained_channel_count']}`")
    lines.append(f"- interpreted hooks: `{len(interpreted_masks)}`")
    lines.append(f"- interpreted channels: `{payload['interpreted_channel_count']}`")
    lines.extend(["", "## Summary", ""])
    for label, summary in (
        ("full_retained", payload["full_retained_summary"]),
        ("interpreted_12_site", payload["interpreted_summary"]),
    ):
        lines.append(
            f"- `{label}`: matches clean `{summary['restricted_matches_clean_rate']:.3f}`, "
            f"matches expected `{summary['restricted_matches_expected_rate']:.3f}`, "
            f"mean |margin delta| `{summary['mean_abs_margin_delta']:.3f}`"
        )
    lines.extend(["", "## Limitation", ""])
    lines.append(payload["limitation"])
    (args.out_dir / "restricted_circuit_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote restricted circuit eval to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
