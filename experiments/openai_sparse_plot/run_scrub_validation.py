from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .activation import ChannelSite, binary_quote_margin, run_with_patch
from .effect_signatures import (
    build_effect_prompt_pairs,
    collect_clean_runs,
    filter_correct_pairs,
    interpreted_channel_sites,
    site_patch_position,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


DEFAULT_SCRUB_SITES: tuple[tuple[str, str], ...] = (
    ("0.mlp.resid_delta:460", "stored quote-type residual channel"),
    ("10.attn.act_in:460", "attention input quote-type read"),
    ("10.attn.v:663", "attention value quote-type channel"),
    ("10.attn.resid_delta:83", "attention output quote-preference write"),
    ("final_resid:83", "final residual quote-preference channel"),
    ("0.mlp.resid_delta:985", "quote detector control site"),
    ("10.attn.q:657", "attention query control site"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run same/different-value resampling tests for string closing sites.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/scrub_validation"))
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--max-same-records-per-site", type=int, default=24)
    parser.add_argument("--max-different-records-per-site", type=int, default=24)
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
    by_site: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_site[record["site_id"]].append(record)

    summary: dict[str, Any] = {}
    for site_id, rows in sorted(by_site.items()):
        same = [row for row in rows if row["relation"] == "same_quote_type"]
        different = [row for row in rows if row["relation"] == "different_quote_type"]
        summary[site_id] = {
            "label": rows[0].get("site_label"),
            "same_records": len(same),
            "different_records": len(different),
            "same_preserve_sign_rate": _mean(row["patched_preserves_base_sign"] for row in same),
            "same_closer_to_base_than_source_rate": _mean(row["closer_to_base_than_source"] for row in same),
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

    sites = tuple(ChannelSite.from_node_id(site_id, label=label) for site_id, label in DEFAULT_SCRUB_SITES)
    record_sites_by_id = {site.site_id: site for site in interpreted_channel_sites(include_post_act=True)}
    for site in sites:
        record_sites_by_id.setdefault(site.site_id, site)
    record_sites = tuple(record_sites_by_id.values())
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
    for site in sites:
        same_count = 0
        different_count = 0
        for base_ex in examples:
            base = runs[base_ex.example_id]
            base_pos = site_patch_position(site, base.positions)
            base_sign = base_ex.sign()
            base_margin = base.margin
            for source_ex in examples:
                if source_ex.example_id == base_ex.example_id:
                    continue
                relation = "same_quote_type" if source_ex.sign() == base_sign else "different_quote_type"
                if relation == "same_quote_type":
                    if same_count >= args.max_same_records_per_site:
                        continue
                    same_count += 1
                else:
                    if different_count >= args.max_different_records_per_site:
                        continue
                    different_count += 1
                source = runs[source_ex.example_id]
                source_pos = site_patch_position(site, source.positions)
                patched_logits = run_with_patch(
                    model,
                    base.token_ids,
                    site=site,
                    source_cache=source.cache,
                    positions=[base_pos],
                    source_positions=[source_pos],
                )
                patched_margin = binary_quote_margin(
                    patched_logits.detach().cpu(),
                    single_token_id=tokens["single"],
                    double_token_id=tokens["double"],
                )
                source_margin = source.margin
                source_sign = source_ex.sign()
                patched_sign = _quote_sign_from_margin(patched_margin)
                records.append(
                    {
                        "site_id": site.site_id,
                        "site_label": site.label,
                        "relation": relation,
                        "base_example_id": base_ex.example_id,
                        "source_example_id": source_ex.example_id,
                        "base_prompt": base_ex.prompt,
                        "source_prompt": source_ex.prompt,
                        "base_sign": base_sign,
                        "source_sign": source_sign,
                        "base_margin": base_margin,
                        "source_margin": source_margin,
                        "patched_margin": patched_margin,
                        "patched_sign": patched_sign,
                        "patched_preserves_base_sign": patched_sign == base_sign,
                        "patched_matches_source_sign": patched_sign == source_sign,
                        "moves_toward_source_sign": (patched_margin - base_margin) * source_sign > 0,
                        "closer_to_base_than_source": abs(patched_margin - base_margin)
                        < abs(source_margin - base_margin),
                        "base_position": base_pos,
                        "source_position": source_pos,
                    }
                )

    summary = _summarize(records)
    payload = {
        "model_info": model_info,
        "quote_token_ids": tokens,
        "input_pairs": len(pairs),
        "kept_pairs": len(kept_pairs),
        "min_abs_margin": args.min_abs_margin,
        "sites": [site.__dict__ for site in sites],
        "summary": summary,
        "records": records,
        "interpretation_note": (
            "Same-quote resampling is a preservation test. Different-quote resampling is a flip/movement test. "
            "This is causal-scrubbing-style evidence for candidate variables, not a full proof of abstraction faithfulness."
        ),
    }
    (args.out_dir / "scrub_validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# String-Closing Scrub Validation", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- kept prompt pairs: `{len(kept_pairs)}` / `{len(pairs)}`")
    lines.append(f"- records: `{len(records)}`")
    lines.append("")
    lines.append("## Site Summary")
    lines.append("")
    for site_id, row in summary.items():
        lines.append(
            f"- `{site_id}` ({row['label']}): same preserve `{row['same_preserve_sign_rate']:.3f}`, "
            f"same mean |delta| `{row['same_mean_abs_margin_delta']:.3f}`, "
            f"different flip `{row['different_flip_to_source_rate']:.3f}`, "
            f"different move `{row['different_moves_toward_source_rate']:.3f}`, "
            f"different signed shift `{row['different_mean_source_signed_shift']:.3f}`"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "A quote-type carrier should preserve output sign under same-quote resampling and flip or strongly move the binary margin under different-quote resampling."
    )
    lines.append(
        "Detector/query control sites may be necessary for routing without encoding quote type, so they are not expected to pass the different-quote flip criterion."
    )
    (args.out_dir / "scrub_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote scrub validation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
