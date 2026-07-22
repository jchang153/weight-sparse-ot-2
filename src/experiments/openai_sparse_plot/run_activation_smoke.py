from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .activation import (
    ChannelSite,
    binary_quote_margin,
    encode_prompt,
    extract_site_values,
    quote_positions_for_prompt,
    record_activations,
    run_with_patch,
)
from .interpreted_circuit import PAPER_BACKED_NODE_SPECS
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


PROMPT_PAIRS = (
    ('x = "hello', "x = 'hello"),
    ('print("abc', "print('abc"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and patch interpreted string-closing sites.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/activation_smoke"))
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def site_position(site: ChannelSite, positions: dict[str, int]) -> int:
    if site.hook_key in {"10.attn.q", "10.attn.resid_delta", "final_resid"}:
        return positions["final_position"]
    if site.hook_key == "10.attn.act_in" and site.channel == 1013:
        return positions["final_position"]
    return positions["opening_quote_position"]


def prompt_kind(prompt: str) -> str:
    if '"' in prompt:
        return "double"
    if "'" in prompt:
        return "single"
    raise ValueError("prompt has no quote")


def margin_target_direction(source_prompt: str) -> int:
    return 1 if prompt_kind(source_prompt) == "double" else -1


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    sites = [
        ChannelSite.from_node_id(spec.node_id, label=spec.label)
        for spec in PAPER_BACKED_NODE_SPECS
        if not spec.node_id.startswith("0.mlp.post_act")
    ]
    prompt_records = []
    caches = {}
    logits_by_prompt = {}
    token_ids_by_prompt = {}
    positions_by_prompt = {}

    for pair in PROMPT_PAIRS:
        for prompt in pair:
            ids = encode_prompt(enc, prompt, device=device)
            positions = quote_positions_for_prompt(enc, prompt)
            logits, cache = record_activations(model, ids, sites)
            margin = binary_quote_margin(
                logits,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
            interesting_positions = sorted(
                {positions["opening_quote_position"], positions["final_position"]}
            )
            prompt_records.append(
                {
                    "prompt": prompt,
                    "quote_type": prompt_kind(prompt),
                    "token_ids": [int(x) for x in ids[0].detach().cpu().tolist()],
                    "decoded_tokens": [enc.decode([int(x)]) for x in ids[0].detach().cpu().tolist()],
                    "positions": positions,
                    "double_minus_single_margin": margin,
                    "predicted_closing_quote": "double" if margin > 0 else "single",
                    "site_values": extract_site_values(cache, sites, positions=interesting_positions),
                    "site_value_positions": interesting_positions,
                }
            )
            caches[prompt] = cache
            logits_by_prompt[prompt] = logits.detach().cpu()
            token_ids_by_prompt[prompt] = ids
            positions_by_prompt[prompt] = positions

    patch_records = []
    for double_prompt, single_prompt in PROMPT_PAIRS:
        for source_prompt, base_prompt in ((double_prompt, single_prompt), (single_prompt, double_prompt)):
            source_positions = positions_by_prompt[source_prompt]
            base_positions = positions_by_prompt[base_prompt]
            base_margin = binary_quote_margin(
                logits_by_prompt[base_prompt],
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
            source_margin = binary_quote_margin(
                logits_by_prompt[source_prompt],
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
            expected_direction = margin_target_direction(source_prompt)
            for site in sites:
                base_pos = site_position(site, base_positions)
                source_pos = site_position(site, source_positions)
                patched_logits = run_with_patch(
                    model,
                    token_ids_by_prompt[base_prompt],
                    site=site,
                    source_cache=caches[source_prompt],
                    positions=[base_pos],
                    source_positions=[source_pos],
                )
                patched_margin = binary_quote_margin(
                    patched_logits.detach().cpu(),
                    single_token_id=tokens["single"],
                    double_token_id=tokens["double"],
                )
                movement = patched_margin - base_margin
                patch_records.append(
                    {
                        "source_prompt": source_prompt,
                        "base_prompt": base_prompt,
                        "site_id": site.site_id,
                        "site_label": site.label,
                        "source_position": source_pos,
                        "base_position": base_pos,
                        "source_margin": source_margin,
                        "base_margin": base_margin,
                        "patched_margin": patched_margin,
                        "movement": movement,
                        "expected_direction": expected_direction,
                        "moves_toward_source_sign": movement * expected_direction > 0,
                        "closer_to_source_margin": abs(patched_margin - source_margin)
                        < abs(base_margin - source_margin),
                    }
                )

    payload = {
        "model_info": model_info,
        "quote_token_ids": tokens,
        "sites": [site.__dict__ for site in sites],
        "prompt_records": prompt_records,
        "patch_records": patch_records,
    }
    (args.out_dir / "activation_smoke.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Activation And Patch Smoke", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- prompts: `{len(prompt_records)}`")
    lines.append(f"- patched single-site interventions: `{len(patch_records)}`")
    lines.append("")
    lines.append("## Clean Margins")
    lines.append("")
    for record in prompt_records:
        lines.append(
            f"- `{record['prompt']}` -> `{record['predicted_closing_quote']}` "
            f"margin `{record['double_minus_single_margin']:.4f}` positions `{record['positions']}`"
        )
    lines.append("")
    lines.append("## Patch Records Moving Toward Source")
    lines.append("")
    for record in patch_records:
        if record["moves_toward_source_sign"]:
            lines.append(
                f"- `{record['site_id']}` {record['base_prompt']!r} <- {record['source_prompt']!r}: "
                f"{record['base_margin']:.4f} -> {record['patched_margin']:.4f}"
            )
    lines.append("")
    lines.append("## Patch Records Not Moving Toward Source")
    lines.append("")
    for record in patch_records:
        if not record["moves_toward_source_sign"]:
            lines.append(
                f"- `{record['site_id']}` {record['base_prompt']!r} <- {record['source_prompt']!r}: "
                f"{record['base_margin']:.4f} -> {record['patched_margin']:.4f}"
            )
    (args.out_dir / "activation_smoke.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote activation smoke to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
