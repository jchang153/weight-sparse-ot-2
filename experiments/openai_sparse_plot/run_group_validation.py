from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .activation import ChannelSite, binary_quote_margin, run_with_group_patch
from .effect_signatures import (
    build_effect_prompt_pairs,
    collect_clean_runs,
    filter_correct_pairs,
    interpreted_channel_sites,
    site_patch_position,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


@dataclass(frozen=True)
class PatchGroup:
    group_id: str
    label: str
    node_ids: tuple[str, ...]

    def sites(self) -> tuple[ChannelSite, ...]:
        return tuple(ChannelSite.from_node_id(node_id) for node_id in self.node_ids)


DEFAULT_GROUPS: tuple[PatchGroup, ...] = (
    PatchGroup(
        group_id="opening_quote_detectors",
        label="layer-0 opening quote detector neurons",
        node_ids=("0.mlp.post_act:863", "0.mlp.post_act:2790"),
    ),
    PatchGroup(
        group_id="stored_quote_type",
        label="layer-0 stored quote-type channel",
        node_ids=("0.mlp.resid_delta:460",),
    ),
    PatchGroup(
        group_id="attention_value_copy",
        label="attention quote-type read/value/write path",
        node_ids=("10.attn.act_in:460", "10.attn.v:663", "10.attn.resid_delta:83"),
    ),
    PatchGroup(
        group_id="output_preference",
        label="attention output plus final residual preference",
        node_ids=("10.attn.resid_delta:83", "final_resid:83"),
    ),
    PatchGroup(
        group_id="full_quote_type_path",
        label="stored type plus downstream copy/output path",
        node_ids=(
            "0.mlp.resid_delta:460",
            "10.attn.act_in:460",
            "10.attn.v:663",
            "10.attn.resid_delta:83",
            "final_resid:83",
        ),
    ),
    PatchGroup(
        group_id="detector_routing_control",
        label="quote detector and attention-routing control path",
        node_ids=("0.mlp.resid_delta:985", "10.attn.act_in:985", "10.attn.k:657", "10.attn.q:657"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped source-resampling tests for string-closing sites.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/group_validation"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--max-same-records-per-group", type=int, default=24)
    parser.add_argument("--max-different-records-per-group", type=int, default=24)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _quote_sign_from_margin(margin: float) -> int:
    return 1 if float(margin) > 0 else -1


def _mean(rows: Iterable[float]) -> float:
    vals = [float(x) for x in rows]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["group_id"]].append(record)
    summary = {}
    for group_id, rows in sorted(by_group.items()):
        same = [row for row in rows if row["relation"] == "same_quote_type"]
        different = [row for row in rows if row["relation"] == "different_quote_type"]
        summary[group_id] = {
            "label": rows[0]["group_label"],
            "node_ids": rows[0]["node_ids"],
            "same_records": len(same),
            "different_records": len(different),
            "same_preserve_sign_rate": _mean(row["patched_preserves_base_sign"] for row in same),
            "same_mean_abs_margin_delta": _mean(abs(row["patched_margin"] - row["base_margin"]) for row in same),
            "different_flip_to_source_rate": _mean(row["patched_matches_source_sign"] for row in different),
            "different_moves_toward_source_rate": _mean(row["moves_toward_source_sign"] for row in different),
            "different_mean_source_signed_shift": _mean(
                (row["patched_margin"] - row["base_margin"]) * row["source_sign"] for row in different
            ),
        }
    return summary


def _all_examples(pairs: Iterable[tuple[Any, Any]]) -> list[Any]:
    out = []
    for left, right in pairs:
        out.extend([left, right])
    return out


def _positions_for_group(group: PatchGroup, positions: dict[str, int | str]) -> dict[str, list[int]]:
    out = {}
    for site in group.sites():
        out[site.site_id] = [site_patch_position(site, positions)]
    return out


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

    record_sites = interpreted_channel_sites(include_post_act=True)
    pairs = build_effect_prompt_pairs(max_pairs=args.max_pairs)
    runs = collect_clean_runs(
        model,
        enc,
        pairs,
        sites=record_sites,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    kept_pairs = filter_correct_pairs(pairs, runs, min_abs_margin=args.min_abs_margin)
    if not kept_pairs:
        raise ValueError("no pairs survived clean prediction and margin filtering")
    examples = _all_examples(kept_pairs)

    records: list[dict[str, Any]] = []
    for group in DEFAULT_GROUPS:
        same_count = 0
        different_count = 0
        sites = group.sites()
        for base_ex in examples:
            base = runs[base_ex.example_id]
            base_sign = base_ex.sign()
            for source_ex in examples:
                if source_ex.example_id == base_ex.example_id:
                    continue
                relation = "same_quote_type" if source_ex.sign() == base_sign else "different_quote_type"
                if relation == "same_quote_type":
                    if same_count >= args.max_same_records_per_group:
                        continue
                    same_count += 1
                else:
                    if different_count >= args.max_different_records_per_group:
                        continue
                    different_count += 1
                source = runs[source_ex.example_id]
                patched_logits = run_with_group_patch(
                    model,
                    base.token_ids,
                    sites=sites,
                    source_cache=source.cache,
                    positions_by_site=_positions_for_group(group, base.positions),
                    source_positions_by_site=_positions_for_group(group, source.positions),
                )
                patched_margin = binary_quote_margin(
                    patched_logits.detach().cpu(),
                    single_token_id=tokens["single"],
                    double_token_id=tokens["double"],
                )
                source_sign = source_ex.sign()
                patched_sign = _quote_sign_from_margin(patched_margin)
                records.append(
                    {
                        "group_id": group.group_id,
                        "group_label": group.label,
                        "node_ids": group.node_ids,
                        "relation": relation,
                        "base_example_id": base_ex.example_id,
                        "source_example_id": source_ex.example_id,
                        "base_prompt": base_ex.prompt,
                        "source_prompt": source_ex.prompt,
                        "base_sign": base_sign,
                        "source_sign": source_sign,
                        "base_margin": base.margin,
                        "source_margin": source.margin,
                        "patched_margin": patched_margin,
                        "patched_sign": patched_sign,
                        "patched_preserves_base_sign": patched_sign == base_sign,
                        "patched_matches_source_sign": patched_sign == source_sign,
                        "moves_toward_source_sign": (patched_margin - base.margin) * source_sign > 0,
                    }
                )

    summary = _summarize(records)
    payload = {
        "model_info": model_info,
        "quote_token_ids": tokens,
        "input_pairs": len(pairs),
        "kept_pairs": len(kept_pairs),
        "min_abs_margin": args.min_abs_margin,
        "groups": [group.__dict__ for group in DEFAULT_GROUPS],
        "summary": summary,
        "records": records,
    }
    (args.out_dir / "group_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# String-Closing Group Validation", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- kept prompt pairs: `{len(kept_pairs)}` / `{len(pairs)}`")
    lines.append(f"- records: `{len(records)}`")
    lines.extend(["", "## Group Summary", ""])
    for group_id, row in summary.items():
        lines.append(
            f"- `{group_id}`: same preserve `{row['same_preserve_sign_rate']:.3f}`, "
            f"same mean |delta| `{row['same_mean_abs_margin_delta']:.3f}`, "
            f"different flip `{row['different_flip_to_source_rate']:.3f}`, "
            f"different move `{row['different_moves_toward_source_rate']:.3f}`, "
            f"different signed shift `{row['different_mean_source_signed_shift']:.3f}`"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "A faithful quote-type group should preserve same-quote behavior and flip or strongly move the output under different-quote resampling."
    )
    lines.append(
        "The detector/routing group is a control: it should affect routing more than quote identity and therefore is not expected to flip the closing quote by itself."
    )
    (args.out_dir / "group_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote group validation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
