from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, encode_prompt, record_activations
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


HEAD_INDEX = 82
VALUE_CHANNEL = 663


@dataclass(frozen=True)
class PromptSpec:
    family: str
    prompt: str
    note: str


PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec("single_opener", 'x = " hello', "double opener with separable content token"),
    PromptSpec("single_opener", "x = ' hello", "single opener with separable content token"),
    PromptSpec("single_opener", 'print(" abc', "double opener after function call"),
    PromptSpec("single_opener", "print(' abc", "single opener after function call"),
    PromptSpec("single_opener", 'value = (" data', "double opener after parenthesis"),
    PromptSpec("single_opener", "value = (' data", "single opener after parenthesis"),
    PromptSpec("completed_distractor", 'a = "x"; b = \' hello', "completed double distractor, target single"),
    PromptSpec("completed_distractor", 'a = \'x\'; b = " hello', "completed single distractor, target double"),
    PromptSpec("same_quote_distractor", 'a = "x"; b = " hello', "completed same-quote distractor, target double"),
    PromptSpec("same_quote_distractor", "a = 'x'; b = ' hello", "completed same-quote distractor, target single"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether head-82 value routing can support non-quote outputs."
    )
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/nonquote_route_value_diagnostic"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _token_spans(enc: Any, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    offset = 0
    rows = []
    for idx, tok in enumerate(token_ids):
        text = enc.decode([int(tok)])
        rows.append(
            {
                "idx": idx,
                "token_id": int(tok),
                "text": text,
                "start": offset,
                "end": offset + len(text),
            }
        )
        offset += len(text)
    return rows


def _char_to_token(spans: Sequence[Mapping[str, Any]], char_idx: int) -> int:
    for row in spans:
        if int(row["start"]) <= int(char_idx) < int(row["end"]):
            return int(row["idx"])
    raise ValueError(f"character index {char_idx} is outside token spans")


def _last_quote_char(prompt: str) -> dict[str, Any]:
    for idx in range(len(prompt) - 1, -1, -1):
        char = prompt[idx]
        if char in {"'", '"'}:
            return {
                "char_idx": idx,
                "quote": char,
                "quote_type": "double" if char == '"' else "single",
            }
    raise ValueError(f"prompt has no quote: {prompt!r}")


def _first_content_after_quote(spans: Sequence[Mapping[str, Any]], opener_pos: int) -> int:
    for row in spans:
        pos = int(row["idx"])
        text = str(row["text"])
        if pos > int(opener_pos) and "'" not in text and '"' not in text and any(ch.isalnum() for ch in text):
            return pos
    raise ValueError("could not find a non-quote content token after the opener")


def _first_code_token_before_quote(spans: Sequence[Mapping[str, Any]], opener_pos: int) -> int:
    for row in spans:
        pos = int(row["idx"])
        text = str(row["text"])
        if pos < int(opener_pos) and "'" not in text and '"' not in text and any(ch.isalpha() for ch in text):
            return pos
    raise ValueError("could not find a non-quote code token before the opener")


def _top_vocab(logits: torch.Tensor, enc: Any, *, top_k: int) -> list[dict[str, Any]]:
    vals, idx = torch.topk(logits[0, -1], k=int(top_k))
    return [
        {"rank": rank + 1, "token_id": int(tok), "token": enc.decode([int(tok)]), "logit": float(logit)}
        for rank, (logit, tok) in enumerate(zip(vals.tolist(), idx.tolist()))
    ]


def _token_rank(logits: torch.Tensor, token_id: int) -> int:
    target = logits[0, -1, int(token_id)]
    return int((logits[0, -1] > target).sum().item()) + 1


def _token_score(logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    probs = torch.softmax(logits[0, -1], dim=-1)
    return {
        "rank": _token_rank(logits, int(token_id)),
        "logit": float(logits[0, -1, int(token_id)]),
        "prob": float(probs[int(token_id)]),
    }


def _quote_scores(logits: torch.Tensor, quote_ids: Mapping[str, int]) -> dict[str, Any]:
    probs = torch.softmax(logits[0, -1], dim=-1)
    single = int(quote_ids["single"])
    double = int(quote_ids["double"])
    return {
        "single_logit": float(logits[0, -1, single]),
        "double_logit": float(logits[0, -1, double]),
        "single_prob": float(probs[single]),
        "double_prob": float(probs[double]),
        "quote_prob_mass": float(probs[single] + probs[double]),
        "margin_double_minus_single": float(logits[0, -1, double] - logits[0, -1, single]),
    }


def _run_with_y_patch(
    model: Any,
    token_ids: torch.Tensor,
    *,
    cache: Mapping[str, torch.Tensor],
    final_position: int,
    source_position: int,
    d_head: int,
    patch_kind: str,
) -> torch.Tensor:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    head_start = HEAD_INDEX * int(d_head)
    head_end = head_start + int(d_head)
    source_v = cache["10.attn.v"][0, int(source_position)].detach()

    def _patch_y(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        if patch_kind == "full_head82_y_to_source_v":
            patched[0, int(final_position), head_start:head_end] = source_v[head_start:head_end].to(
                device=patched.device,
                dtype=patched.dtype,
            )
        elif patch_kind == "channel663_y_to_source_v":
            patched[0, int(final_position), VALUE_CHANNEL] = source_v[VALUE_CHANNEL].to(
                device=patched.device,
                dtype=patched.dtype,
            )
        elif patch_kind == "zero_head82_y":
            patched[0, int(final_position), head_start:head_end] = 0.0
        else:
            raise ValueError(f"unknown patch kind: {patch_kind}")
        return patched

    with torch.no_grad():
        with hook_recorder(regex="^$", interventions={"10.attn.y": _patch_y}):
            logits, _, _ = model(token_ids)
    return logits.detach().cpu()


def _copy_expected_id(
    *,
    source_kind: str,
    source_token_id: int,
    quote_type: str,
    quote_ids: Mapping[str, int],
) -> int:
    if source_kind == "opener_quote":
        return int(quote_ids["double"] if quote_type == "double" else quote_ids["single"])
    return int(source_token_id)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["source_kind"]), str(row["patch_kind"])), []).append(row)
    out: dict[str, Any] = {}
    for (source_kind, patch_kind), members in sorted(grouped.items()):
        key = f"{source_kind}::{patch_kind}"
        out[key] = {
            "n": len(members),
            "expected_top1": sum(1 for row in members if int(row["expected_after"]["rank"]) == 1) / len(members),
            "expected_top5": sum(1 for row in members if int(row["expected_after"]["rank"]) <= 5) / len(members),
            "mean_expected_rank": sum(float(row["expected_after"]["rank"]) for row in members) / len(members),
            "mean_expected_logit_delta": sum(float(row["expected_logit_delta"]) for row in members) / len(members),
            "mean_expected_prob_after": sum(float(row["expected_after"]["prob"]) for row in members) / len(members),
            "mean_quote_prob_mass_after": sum(float(row["quote_scores_after"]["quote_prob_mass"]) for row in members)
            / len(members),
            "quote_top1_rate": sum(
                1
                for row in members
                if row["top_vocab_after"][0]["token"] in {"'", '"'}
            )
            / len(members),
        }
    return out


def _format_summary(summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Source kind | Patch | n | expected top1 | expected top5 | mean expected rank | mean logit delta | mean prob | mean quote mass | quote top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary.items():
        source_kind, patch_kind = key.split("::", 1)
        lines.append(
            f"| `{source_kind}` | `{patch_kind}` | {row['n']} | {row['expected_top1']:.3f} | "
            f"{row['expected_top5']:.3f} | {row['mean_expected_rank']:.1f} | "
            f"{row['mean_expected_logit_delta']:.3f} | {row['mean_expected_prob_after']:.5f} | "
            f"{row['mean_quote_prob_mass_after']:.3f} | {row['quote_top1_rate']:.3f} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    enc = make_tinypython_encoding(args.circuit_home)
    quote_ids = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    d_head = int(model.config.d_head)
    sites = (
        ChannelSite.from_node_id("10.attn.v:663"),
        ChannelSite.from_node_id("10.attn.y:663"),
    )

    rows = []
    for spec in PROMPTS:
        token_ids_raw = enc.encode(spec.prompt)
        spans = _token_spans(enc, token_ids_raw)
        opener = _last_quote_char(spec.prompt)
        opener_pos = _char_to_token(spans, int(opener["char_idx"]))
        content_pos = _first_content_after_quote(spans, opener_pos)
        code_pos = _first_code_token_before_quote(spans, opener_pos)
        final_position = len(token_ids_raw) - 1
        ids = encode_prompt(enc, spec.prompt, device=device)
        natural_logits, cache = record_activations(model, ids, sites)
        natural_logits = natural_logits.detach().cpu()

        source_specs = (
            ("opener_quote", opener_pos),
            ("content_after_opener", content_pos),
            ("code_before_opener", code_pos),
        )
        patch_kinds = (
            "full_head82_y_to_source_v",
            "channel663_y_to_source_v",
            "zero_head82_y",
        )
        for source_kind, source_position in source_specs:
            source_token_id = int(spans[int(source_position)]["token_id"])
            expected_id = _copy_expected_id(
                source_kind=source_kind,
                source_token_id=source_token_id,
                quote_type=str(opener["quote_type"]),
                quote_ids=quote_ids,
            )
            before = _token_score(natural_logits, expected_id)
            for patch_kind in patch_kinds:
                patched_logits = _run_with_y_patch(
                    model,
                    ids,
                    cache=cache,
                    final_position=final_position,
                    source_position=source_position,
                    d_head=d_head,
                    patch_kind=patch_kind,
                )
                after = _token_score(patched_logits, expected_id)
                rows.append(
                    {
                        "family": spec.family,
                        "prompt": spec.prompt,
                        "note": spec.note,
                        "token_spans": spans,
                        "opener_quote": opener,
                        "opener_position": opener_pos,
                        "content_position": content_pos,
                        "code_position": code_pos,
                        "final_position": final_position,
                        "source_kind": source_kind,
                        "source_position": int(source_position),
                        "source_token_id": source_token_id,
                        "source_token": str(spans[int(source_position)]["text"]),
                        "expected_token_id": int(expected_id),
                        "expected_token": enc.decode([int(expected_id)]),
                        "patch_kind": patch_kind,
                        "expected_before": before,
                        "expected_after": after,
                        "expected_logit_delta": float(after["logit"] - before["logit"]),
                        "quote_scores_before": _quote_scores(natural_logits, quote_ids),
                        "quote_scores_after": _quote_scores(patched_logits, quote_ids),
                        "binary_quote_margin_before": binary_quote_margin(
                            natural_logits,
                            single_token_id=quote_ids["single"],
                            double_token_id=quote_ids["double"],
                        ),
                        "binary_quote_margin_after": binary_quote_margin(
                            patched_logits,
                            single_token_id=quote_ids["single"],
                            double_token_id=quote_ids["double"],
                        ),
                        "top_vocab_before": _top_vocab(natural_logits, enc, top_k=args.top_k),
                        "top_vocab_after": _top_vocab(patched_logits, enc, top_k=args.top_k),
                    }
                )

    summary = _aggregate(rows)
    payload = {
        "model_info": model_info,
        "quote_token_ids": quote_ids,
        "head_index": HEAD_INDEX,
        "value_channel": VALUE_CHANNEL,
        "d_head": d_head,
        "summary": summary,
        "rows": rows,
    }
    json_path = args.out_dir / "nonquote_route_value_diagnostic.json"
    md_path = args.out_dir / "nonquote_route_value_diagnostic.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Non-Quote Route Value Diagnostic",
        "",
        f"- model: `{args.model}`",
        f"- head index: `{HEAD_INDEX}`",
        f"- value channel: `{VALUE_CHANNEL}`",
        f"- prompts: `{len(PROMPTS)}`",
        "",
        "Question: if the route is forced to a non-quote token, does the head/value path make the model output that token?",
        "",
        "Patch definitions:",
        "",
        "- `full_head82_y_to_source_v`: replace head-82's final-position attention output with the source token's full head-82 value vector.",
        "- `channel663_y_to_source_v`: replace only the localized value scalar `663` at the final-position attention output.",
        "- `zero_head82_y`: zero head-82's final-position attention output as a control.",
        "",
        "For `opener_quote`, the expected token is the matching quote token. For the two non-quote source kinds, the expected token is the source token itself.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(_format_summary(summary))
    lines.extend(["", "## Prompt Details", ""])
    for row in rows:
        if row["patch_kind"] != "full_head82_y_to_source_v":
            continue
        lines.append(
            f"- `{row['prompt']}` source `{row['source_kind']}` token `{row['source_token']}` "
            f"expected `{row['expected_token']}` rank `{row['expected_after']['rank']}` "
            f"logit delta `{row['expected_logit_delta']:.3f}` quote mass `{row['quote_scores_after']['quote_prob_mass']:.3f}`"
        )
        top = ", ".join(f"{x['rank']}:{x['token']!r}" for x in row["top_vocab_after"][:5])
        lines.append(f"  - top after patch: {top}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
