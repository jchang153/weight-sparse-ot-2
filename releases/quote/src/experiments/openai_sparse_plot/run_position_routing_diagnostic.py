from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, encode_prompt, record_activations
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


HEAD_INDEX = 82
QK_CHANNEL = 657
QUERY_SOURCE_CHANNEL = 1013
KEY_SOURCE_CHANNEL = 985
EXTRA_KEY_SOURCE_CHANNEL = 898


@dataclass(frozen=True)
class RoutingPrompt:
    family: str
    prompt: str
    note: str


PROMPTS: tuple[RoutingPrompt, ...] = (
    RoutingPrompt("single_opener", 'x = "hello', "one double opener, short prefix"),
    RoutingPrompt("single_opener", "x = 'hello", "one single opener, short prefix"),
    RoutingPrompt("single_opener", 'print("abc', "one double opener after function call"),
    RoutingPrompt("single_opener", "print('abc", "one single opener after function call"),
    RoutingPrompt("single_opener", 'value = ("hello', "one double opener after paren"),
    RoutingPrompt("single_opener", "value = ('hello", "one single opener after paren"),
    RoutingPrompt("single_opener", 'handler(prefix, ("abc', "one double opener late in prompt"),
    RoutingPrompt("single_opener", "handler(prefix, ('abc", "one single opener late in prompt"),
    RoutingPrompt("completed_distractor", 'a = "x"; b = \'hello', "earlier completed double string, target single"),
    RoutingPrompt("completed_distractor", 'a = \'x\'; b = "hello', "earlier completed single string, target double"),
    RoutingPrompt("completed_distractor", 'prefix = "old"; print(\'abc', "earlier completed double string before print single"),
    RoutingPrompt("completed_distractor", 'prefix = \'old\'; print("abc', "earlier completed single string before print double"),
    RoutingPrompt("same_quote_distractor", 'a = "x"; b = "hello', "earlier completed double string, target double"),
    RoutingPrompt("same_quote_distractor", "a = 'x'; b = 'hello", "earlier completed single string, target single"),
    RoutingPrompt(
        "opposite_quote_distractors",
        'a = \'x\'; b = \'y\'; c = "hello',
        "two completed single strings, target double",
    ),
    RoutingPrompt(
        "opposite_quote_distractors",
        'a = "x"; b = "y"; c = \'hello',
        "two completed double strings, target single",
    ),
    RoutingPrompt("two_completed_distractors", 'a = "x"; b = \'y\'; c = "hello', "two completed distractors, target double"),
    RoutingPrompt("two_completed_distractors", 'a = \'x\'; b = "y"; c = \'hello', "two completed distractors, target single"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether head-82 attention selects the opener position P.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/position_routing_diagnostic"))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def _token_spans(enc: Any, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    offset = 0
    rows = []
    for idx, tok in enumerate(token_ids):
        text = enc.decode([int(tok)])
        start = offset
        end = start + len(text)
        rows.append({"idx": idx, "token_id": int(tok), "text": text, "start": start, "end": end})
        offset = end
    return rows


def _char_to_token(spans: Sequence[Mapping[str, Any]], char_idx: int) -> int:
    for row in spans:
        if int(row["start"]) <= int(char_idx) < int(row["end"]):
            return int(row["idx"])
    raise ValueError(f"character index {char_idx} is outside token spans")


def _quote_chars(prompt: str) -> list[dict[str, Any]]:
    rows = []
    for idx, char in enumerate(prompt):
        if char == '"' or char == "'":
            rows.append({"char_idx": idx, "quote": char, "quote_type": "double" if char == '"' else "single"})
    if not rows:
        raise ValueError(f"prompt has no quote: {prompt!r}")
    return rows


def _top_positions(probs: torch.Tensor, spans: Sequence[Mapping[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    k = min(int(top_k), int(probs.numel()))
    vals, idx = torch.topk(probs, k=k)
    return [
        {
            "rank": rank + 1,
            "position": int(pos),
            "weight": float(weight),
            "token": str(spans[int(pos)]["text"]),
            "token_id": int(spans[int(pos)]["token_id"]),
        }
        for rank, (weight, pos) in enumerate(zip(vals.tolist(), idx.tolist()))
    ]


def _rank_of_position(probs: torch.Tensor, position: int) -> int:
    order = torch.argsort(probs, descending=True)
    matches = (order == int(position)).nonzero(as_tuple=True)[0]
    return int(matches[0].item()) + 1


def _top_quote_position(
    probs: torch.Tensor,
    quote_token_positions: Sequence[int],
    spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not quote_token_positions:
        return {}
    best = max(quote_token_positions, key=lambda pos: float(probs[int(pos)]))
    quote_text = str(spans[int(best)]["text"])
    return {
        "position": int(best),
        "weight": float(probs[int(best)]),
        "token": quote_text,
        "quote_type": "double" if '"' in quote_text else "single" if "'" in quote_text else "unknown",
    }


def _quote_position_rank(probs: torch.Tensor, position: int, quote_token_positions: Sequence[int]) -> int | None:
    if not quote_token_positions:
        return None
    ranked = sorted((int(pos) for pos in quote_token_positions), key=lambda pos: float(probs[pos]), reverse=True)
    return ranked.index(int(position)) + 1 if int(position) in ranked else None


def _top_vocab(logits: torch.Tensor, enc: Any, *, top_k: int) -> list[dict[str, Any]]:
    vals, idx = torch.topk(logits[0, -1], k=int(top_k))
    return [
        {"rank": rank + 1, "token_id": int(tok), "token": enc.decode([int(tok)]), "logit": float(logit)}
        for rank, (logit, tok) in enumerate(zip(vals.tolist(), idx.tolist()))
    ]


def _attention_rows(
    cache: Mapping[str, torch.Tensor],
    *,
    model: Any,
    final_position: int,
    d_head: int,
) -> dict[str, torch.Tensor]:
    q = cache["10.attn.q"][0].to(torch.float32)
    k = cache["10.attn.k"][0].to(torch.float32)
    act_in = cache["10.attn.act_in"][0].to(torch.float32)
    head_start = HEAD_INDEX * d_head
    head_end = head_start + d_head
    allowed = slice(0, int(final_position) + 1)
    full_scores = (q[int(final_position), head_start:head_end] @ k[allowed, head_start:head_end].t()) / math.sqrt(d_head)
    channel_scores = q[int(final_position), QK_CHANNEL] * k[allowed, QK_CHANNEL] / math.sqrt(d_head)
    q_out = QK_CHANNEL
    k_out = int(model.config.n_head) * d_head + QK_CHANNEL
    weight = model.transformer.h[10].attn.c_attn.weight.detach().cpu().to(torch.float32)
    q_local = act_in[int(final_position), QUERY_SOURCE_CHANNEL] * weight[q_out, QUERY_SOURCE_CHANNEL]
    k_local = act_in[allowed, KEY_SOURCE_CHANNEL] * weight[k_out, KEY_SOURCE_CHANNEL]
    k_local_plus_extra = k_local + act_in[allowed, EXTRA_KEY_SOURCE_CHANNEL] * weight[k_out, EXTRA_KEY_SOURCE_CHANNEL]
    localized_scores = q_local * k_local / math.sqrt(d_head)
    localized_plus_extra_scores = q_local * k_local_plus_extra / math.sqrt(d_head)
    return {
        "full_head_scores": full_scores,
        "full_head_probs": torch.softmax(full_scores, dim=-1),
        "channel_scores": channel_scores,
        "channel_probs": torch.softmax(channel_scores, dim=-1),
        "localized_qk_scores": localized_scores,
        "localized_qk_probs": torch.softmax(localized_scores, dim=-1),
        "localized_qk_plus_extra_scores": localized_plus_extra_scores,
        "localized_qk_plus_extra_probs": torch.softmax(localized_plus_extra_scores, dim=-1),
        "q_local": q_local.reshape(()),
        "localized_weight_q_1013": weight[q_out, QUERY_SOURCE_CHANNEL].reshape(()),
        "localized_weight_k_985": weight[k_out, KEY_SOURCE_CHANNEL].reshape(()),
        "extra_weight_k_898": weight[k_out, EXTRA_KEY_SOURCE_CHANNEL].reshape(()),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    out = {}
    for family, members in sorted(by_family.items()):
        out[family] = {
            "n": len(members),
            "quote_accuracy": sum(1 for row in members if row["predicted_quote_type"] == row["target_quote_type"]) / len(members),
            "target_top1_all": sum(1 for row in members if row[key]["target_rank_all"] == 1) / len(members),
            "target_top3_all": sum(1 for row in members if row[key]["target_rank_all"] <= 3) / len(members),
            "target_top1_quote": sum(1 for row in members if row[key]["target_rank_quote"] == 1) / len(members),
            "mean_target_weight": sum(float(row[key]["target_weight"]) for row in members) / len(members),
            "mean_target_rank_all": sum(float(row[key]["target_rank_all"]) for row in members) / len(members),
        }
    all_rows = list(rows)
    out["ALL"] = {
        "n": len(all_rows),
        "quote_accuracy": sum(1 for row in all_rows if row["predicted_quote_type"] == row["target_quote_type"]) / len(all_rows),
        "target_top1_all": sum(1 for row in all_rows if row[key]["target_rank_all"] == 1) / len(all_rows),
        "target_top3_all": sum(1 for row in all_rows if row[key]["target_rank_all"] <= 3) / len(all_rows),
        "target_top1_quote": sum(1 for row in all_rows if row[key]["target_rank_quote"] == 1) / len(all_rows),
        "mean_target_weight": sum(float(row[key]["target_weight"]) for row in all_rows) / len(all_rows),
        "mean_target_rank_all": sum(float(row[key]["target_rank_all"]) for row in all_rows) / len(all_rows),
    }
    return out


def _format_summary_table(summary: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Family | n | quote acc | target top1 all | target top3 all | target top1 among quotes | mean target weight | mean target rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in summary.items():
        lines.append(
            f"| `{family}` | {row['n']} | {row['quote_accuracy']:.3f} | {row['target_top1_all']:.3f} | "
            f"{row['target_top3_all']:.3f} | {row['target_top1_quote']:.3f} | "
            f"{row['mean_target_weight']:.3f} | {row['mean_target_rank_all']:.2f} |"
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
        ChannelSite.from_node_id("10.attn.act_in:1013"),
        ChannelSite.from_node_id("10.attn.act_in:985"),
        ChannelSite.from_node_id("10.attn.act_in:898"),
        ChannelSite.from_node_id("10.attn.q:657"),
        ChannelSite.from_node_id("10.attn.k:657"),
    )
    rows = []
    for item in PROMPTS:
        token_ids = enc.encode(item.prompt)
        spans = _token_spans(enc, token_ids)
        quote_chars = _quote_chars(item.prompt)
        for quote in quote_chars:
            quote["token_position"] = _char_to_token(spans, int(quote["char_idx"]))
        target = quote_chars[-1]
        target_pos = int(target["token_position"])
        final_position = len(token_ids) - 1
        ids = encode_prompt(enc, item.prompt, device=device)
        logits, cache = record_activations(model, ids, sites)
        margin = binary_quote_margin(logits.detach().cpu(), single_token_id=quote_ids["single"], double_token_id=quote_ids["double"])
        predicted = "double" if margin > 0 else "single"
        attn = _attention_rows(cache, model=model, final_position=final_position, d_head=d_head)
        quote_positions = sorted({int(q["token_position"]) for q in quote_chars})

        def pack_attention(prefix: str, probs: torch.Tensor, scores: torch.Tensor) -> dict[str, Any]:
            return {
                "target_weight": float(probs[target_pos]),
                "target_score": float(scores[target_pos]),
                "target_rank_all": _rank_of_position(probs, target_pos),
                "target_rank_quote": _quote_position_rank(probs, target_pos, quote_positions),
                "top_positions": _top_positions(probs, spans, top_k=args.top_k),
                "top_quote_position": _top_quote_position(probs, quote_positions, spans),
                "kind": prefix,
            }

        rows.append(
            {
                "family": item.family,
                "prompt": item.prompt,
                "note": item.note,
                "decoded": enc.decode(token_ids),
                "n_tokens": len(token_ids),
                "token_spans": spans,
                "quote_chars": quote_chars,
                "target_quote": target,
                "target_quote_type": target["quote_type"],
                "target_position": target_pos,
                "final_position": final_position,
                "binary_quote_margin": margin,
                "predicted_quote_type": predicted,
                "quote_correct": predicted == target["quote_type"],
                "top_vocab": _top_vocab(logits.detach().cpu(), enc, top_k=args.top_k),
                "full_head82": pack_attention(
                    "full_head82",
                    attn["full_head_probs"].detach().cpu(),
                    attn["full_head_scores"].detach().cpu(),
                ),
                "channel657": pack_attention(
                    "channel657",
                    attn["channel_probs"].detach().cpu(),
                    attn["channel_scores"].detach().cpu(),
                ),
                "localized_qk": pack_attention(
                    "localized_qk",
                    attn["localized_qk_probs"].detach().cpu(),
                    attn["localized_qk_scores"].detach().cpu(),
                ),
                "localized_qk_plus_extra": pack_attention(
                    "localized_qk_plus_extra",
                    attn["localized_qk_plus_extra_probs"].detach().cpu(),
                    attn["localized_qk_plus_extra_scores"].detach().cpu(),
                ),
                "localized_values": {
                    "q_local": float(attn["q_local"]),
                    "weight_q_1013": float(attn["localized_weight_q_1013"]),
                    "weight_k_985": float(attn["localized_weight_k_985"]),
                    "weight_k_898": float(attn["extra_weight_k_898"]),
                },
            }
        )

    payload = {
        "model_info": model_info,
        "quote_token_ids": quote_ids,
        "head_index": HEAD_INDEX,
        "qk_channel": QK_CHANNEL,
        "d_head": d_head,
        "summary": {
            "full_head82": _summarize(rows, "full_head82"),
            "channel657": _summarize(rows, "channel657"),
            "localized_qk": _summarize(rows, "localized_qk"),
            "localized_qk_plus_extra": _summarize(rows, "localized_qk_plus_extra"),
        },
        "rows": rows,
    }
    (args.out_dir / "position_routing_diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Position Routing Diagnostic", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- head index: `{HEAD_INDEX}`")
    lines.append(f"- QK channel: `{QK_CHANNEL}`")
    lines.append(f"- prompts: `{len(rows)}`")
    lines.append("")
    lines.append("`P` is defined here as the token containing the last quote character in the unfinished prompt.")
    lines.append("")
    lines.append("## Full Head-82 Attention")
    lines.extend(_format_summary_table(payload["summary"]["full_head82"]))
    lines.append("")
    lines.append("## Channel-657 Only Attention Proxy")
    lines.extend(_format_summary_table(payload["summary"]["channel657"]))
    lines.append("")
    lines.append("## Localized QK Subcircuit")
    lines.append("")
    lines.append(
        "`localized_qk` uses only `10.attn.act_in:1013 -> 10.attn.q:657` and "
        "`10.attn.act_in:985 -> 10.attn.k:657`."
    )
    lines.extend(_format_summary_table(payload["summary"]["localized_qk"]))
    lines.append("")
    lines.append("## Localized QK Plus Extra Raw Key Edge")
    lines.append("")
    lines.append(
        "`localized_qk_plus_extra` additionally includes the raw edge "
        "`10.attn.act_in:898 -> 10.attn.k:657`, which appears in the released artifact "
        "but was not part of our canonical 12-site interpretation."
    )
    lines.extend(_format_summary_table(payload["summary"]["localized_qk_plus_extra"]))
    lines.append("")
    lines.append("## Prompt Details")
    lines.append("")
    for row in rows:
        lines.append(
            f"- `{row['prompt']}` family `{row['family']}` target `{row['target_quote_type']}` "
            f"pred `{row['predicted_quote_type']}` margin `{row['binary_quote_margin']:.3f}`"
        )
        full = row["full_head82"]
        chan = row["channel657"]
        lines.append(
            f"  - full head: target rank all `{full['target_rank_all']}`, rank among quotes `{full['target_rank_quote']}`, "
            f"weight `{full['target_weight']:.3f}`, top quote pos `{full['top_quote_position'].get('position')}`"
        )
        lines.append(
            f"  - channel 657: target rank all `{chan['target_rank_all']}`, rank among quotes `{chan['target_rank_quote']}`, "
            f"weight `{chan['target_weight']:.3f}`, top quote pos `{chan['top_quote_position'].get('position')}`"
        )
        local = row["localized_qk"]
        local_extra = row["localized_qk_plus_extra"]
        lines.append(
            f"  - localized QK: target rank all `{local['target_rank_all']}`, rank among quotes `{local['target_rank_quote']}`, "
            f"weight `{local['target_weight']:.3f}`, top quote pos `{local['top_quote_position'].get('position')}`"
        )
        lines.append(
            f"  - localized QK + 898: target rank all `{local_extra['target_rank_all']}`, "
            f"rank among quotes `{local_extra['target_rank_quote']}`, weight `{local_extra['target_weight']:.3f}`, "
            f"top quote pos `{local_extra['top_quote_position'].get('position')}`"
        )
    (args.out_dir / "position_routing_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
