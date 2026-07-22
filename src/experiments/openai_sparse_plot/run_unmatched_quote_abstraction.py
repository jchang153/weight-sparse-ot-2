from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, run_with_group_patch
from .effect_signatures import (
    build_effect_prompt_pairs,
    collect_clean_runs,
    filter_correct_pairs,
    site_patch_position,
)
from .interpreted_circuit import PAPER_BACKED_NODE_SPECS
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids
from .schema import StringClosingExample


@dataclass(frozen=True)
class CandidateHandle:
    handle_id: str
    label: str
    node_ids: tuple[str, ...]
    kind: str

    def sites(self) -> tuple[ChannelSite, ...]:
        return tuple(ChannelSite.from_node_id(node_id, label=self.label) for node_id in self.node_ids)


@dataclass(frozen=True)
class ResamplingSpec:
    relation: str
    base_id: str
    source_id: str
    wrong_variable: str | None = None


CORE_12_NODES: tuple[str, ...] = tuple(spec.node_id for spec in PAPER_BACKED_NODE_SPECS)


DEFAULT_HANDLES: tuple[CandidateHandle, ...] = (
    CandidateHandle(
        "opening_quote_detectors",
        "layer-0 quote detector pair",
        ("0.mlp.post_act:863", "0.mlp.post_act:2790"),
        "hand_openai_group",
    ),
    CandidateHandle(
        "stored_quote_type",
        "layer-0 stored quote-type channel",
        ("0.mlp.resid_delta:460",),
        "hand_openai_group",
    ),
    CandidateHandle(
        "stored_and_attention_read",
        "stored quote type plus layer-10 read input",
        ("0.mlp.resid_delta:460", "10.attn.act_in:460"),
        "hand_openai_group",
    ),
    CandidateHandle(
        "attention_value_channel",
        "head-82 value quote-type channel",
        ("10.attn.v:663",),
        "hand_openai_group",
    ),
    CandidateHandle(
        "attention_value_write",
        "head-82 value and residual write",
        ("10.attn.v:663", "10.attn.resid_delta:83"),
        "hand_openai_group",
    ),
    CandidateHandle(
        "output_preference",
        "attention output plus final residual preference",
        ("10.attn.resid_delta:83", "final_resid:83"),
        "hand_openai_group",
    ),
    CandidateHandle(
        "full_quote_type_path",
        "stored type plus downstream copy/output path",
        (
            "0.mlp.resid_delta:460",
            "10.attn.act_in:460",
            "10.attn.v:663",
            "10.attn.resid_delta:83",
            "final_resid:83",
        ),
        "hand_openai_group",
    ),
    CandidateHandle(
        "detector_routing_control",
        "quote detector and attention-routing control path",
        ("0.mlp.resid_delta:985", "10.attn.act_in:985", "10.attn.k:657", "10.attn.q:657"),
        "control_openai_group",
    ),
    CandidateHandle(
        "query_only_control",
        "final-position query channel only",
        ("10.attn.q:657",),
        "control_openai_group",
    ),
    CandidateHandle(
        "quote_detector_mass_control",
        "quote detector residual mass channel",
        ("0.mlp.resid_delta:985",),
        "control_openai_group",
    ),
    CandidateHandle(
        "full_interpreted_12",
        "all interpreted OpenAI string-closing nodes",
        CORE_12_NODES,
        "full_openai_circuit",
    ),
    CandidateHandle(
        "random_channels_seed0",
        "fixed random-channel control",
        ("0.mlp.resid_delta:1", "10.attn.v:1", "final_resid:1"),
        "random_control",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate U = unmatched quote type as a single-variable causal abstraction."
    )
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/unmatched_quote_abstraction"))
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--max-records-per-relation", type=int, default=28)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _all_examples(pairs: Iterable[tuple[StringClosingExample, StringClosingExample]]) -> list[StringClosingExample]:
    out: list[StringClosingExample] = []
    for left, right in pairs:
        out.extend([left, right])
    return out


def _mean(values: Iterable[float | bool]) -> float:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _quote_sign_from_margin(margin: float) -> int:
    return 1 if float(margin) > 0 else -1


def _positions_for_handle(handle: CandidateHandle, positions: Mapping[str, int | str]) -> dict[str, list[int]]:
    return {site.site_id: [site_patch_position(site, positions)] for site in handle.sites()}


def _example_lookup(pairs: Sequence[tuple[StringClosingExample, StringClosingExample]]) -> dict[str, StringClosingExample]:
    return {example.example_id: example for example in _all_examples(pairs)}


def _same_opener_position(base: StringClosingExample, source: StringClosingExample, runs: Mapping[str, Any]) -> bool:
    return int(runs[base.example_id].positions["opening_quote_position"]) == int(
        runs[source.example_id].positions["opening_quote_position"]
    )


def _same_content_length(base: StringClosingExample, source: StringClosingExample) -> bool:
    return len(base.content) == len(source.content)


def _candidate_specs_for_relation(
    *,
    relation: str,
    examples: Sequence[StringClosingExample],
    runs: Mapping[str, Any],
) -> list[ResamplingSpec]:
    specs: list[ResamplingSpec] = []
    for base in examples:
        for source in examples:
            if source.example_id == base.example_id:
                continue
            same_u = source.sign() == base.sign()
            if relation == "same_u":
                if same_u and source.pair_id != base.pair_id:
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id))
            elif relation == "opposite_u":
                if not same_u and source.pair_id == base.pair_id:
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id))
            elif relation == "wrong_same_position":
                if not same_u and _same_opener_position(base, source, runs):
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id, "opening_position"))
            elif relation == "wrong_same_content":
                if not same_u and source.content == base.content:
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id, "content"))
            elif relation == "wrong_same_content_length":
                if not same_u and _same_content_length(base, source):
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id, "content_length"))
            else:
                raise ValueError(f"unknown relation: {relation}")
    return specs


def _build_resampling_specs(
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    runs: Mapping[str, Any],
    *,
    max_records_per_relation: int,
) -> tuple[ResamplingSpec, ...]:
    examples = sorted(_all_examples(pairs), key=lambda x: x.example_id)
    relations = (
        "same_u",
        "opposite_u",
        "wrong_same_position",
        "wrong_same_content",
        "wrong_same_content_length",
    )
    out: list[ResamplingSpec] = []
    for relation in relations:
        specs = _candidate_specs_for_relation(relation=relation, examples=examples, runs=runs)
        out.extend(specs[: int(max_records_per_relation)])
    return tuple(out)


def _run_records_for_handle(
    *,
    model: Any,
    handle: CandidateHandle,
    specs: Sequence[ResamplingSpec],
    examples: Mapping[str, StringClosingExample],
    runs: Mapping[str, Any],
    single_token_id: int,
    double_token_id: int,
) -> list[dict[str, Any]]:
    records = []
    sites = handle.sites()
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = runs[spec.base_id]
        source = runs[spec.source_id]
        patched_logits = run_with_group_patch(
            model,
            base.token_ids,
            sites=sites,
            source_cache=source.cache,
            positions_by_site=_positions_for_handle(handle, base.positions),
            source_positions_by_site=_positions_for_handle(handle, source.positions),
        )
        patched_margin = binary_quote_margin(
            patched_logits.detach().cpu(),
            single_token_id=single_token_id,
            double_token_id=double_token_id,
        )
        base_sign = base_ex.sign()
        source_sign = source_ex.sign()
        patched_sign = _quote_sign_from_margin(patched_margin)
        records.append(
            {
                "handle_id": handle.handle_id,
                "handle_label": handle.label,
                "handle_kind": handle.kind,
                "node_ids": handle.node_ids,
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_example_id": base_ex.example_id,
                "source_example_id": source_ex.example_id,
                "base_prompt": base_ex.prompt,
                "source_prompt": source_ex.prompt,
                "base_template_id": base_ex.template_id,
                "source_template_id": source_ex.template_id,
                "base_content": base_ex.content,
                "source_content": source_ex.content,
                "base_sign": base_sign,
                "source_sign": source_sign,
                "base_margin": base.margin,
                "source_margin": source.margin,
                "patched_margin": patched_margin,
                "patched_sign": patched_sign,
                "patched_preserves_base_sign": patched_sign == base_sign,
                "patched_matches_source_sign": patched_sign == source_sign,
                "moves_toward_source_sign": (patched_margin - base.margin) * source_sign > 0,
                "source_signed_shift": (patched_margin - base.margin) * source_sign,
                "abs_margin_delta": abs(patched_margin - base.margin),
                "base_opening_position": int(base.positions["opening_quote_position"]),
                "source_opening_position": int(source.positions["opening_quote_position"]),
            }
        )
    return records


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_handle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_handle[str(row["handle_id"])].append(row)

    summary: dict[str, Any] = {}
    for handle_id, rows in sorted(by_handle.items()):
        relation_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            relation_rows[str(row["relation"])].append(row)
        same = relation_rows["same_u"]
        opposite = relation_rows["opposite_u"]
        wrong_position = relation_rows["wrong_same_position"]
        wrong_content = relation_rows["wrong_same_content"]
        wrong_length = relation_rows["wrong_same_content_length"]
        summary[handle_id] = {
            "handle_label": rows[0]["handle_label"],
            "handle_kind": rows[0]["handle_kind"],
            "node_ids": rows[0]["node_ids"],
            "records": len(rows),
            "same_u_records": len(same),
            "same_u_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in same),
            "same_u_mean_abs_margin_delta": _mean(row["abs_margin_delta"] for row in same),
            "opposite_u_records": len(opposite),
            "opposite_u_flip_rate": _mean(row["patched_matches_source_sign"] for row in opposite),
            "opposite_u_move_rate": _mean(row["moves_toward_source_sign"] for row in opposite),
            "opposite_u_mean_source_signed_shift": _mean(row["source_signed_shift"] for row in opposite),
            "wrong_position_records": len(wrong_position),
            "wrong_position_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in wrong_position),
            "wrong_content_records": len(wrong_content),
            "wrong_content_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in wrong_content),
            "wrong_length_records": len(wrong_length),
            "wrong_length_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in wrong_length),
        }
    return summary


def _metric(row: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    value = float(value)
    if math.isnan(value):
        return default
    return value


def _handle_signature(row: Mapping[str, Any]) -> tuple[float, ...]:
    same_delta = _metric(row, "same_u_mean_abs_margin_delta", default=100.0)
    signed_shift = _metric(row, "opposite_u_mean_source_signed_shift", default=0.0)
    return (
        _metric(row, "same_u_preserve_rate"),
        _metric(row, "opposite_u_flip_rate"),
        1.0 - _metric(row, "wrong_position_preserve_rate", default=1.0),
        1.0 - _metric(row, "wrong_content_preserve_rate", default=1.0),
        1.0 - _metric(row, "wrong_length_preserve_rate", default=1.0),
        1.0 / (1.0 + max(0.0, same_delta)),
        (math.tanh(signed_shift / 10.0) + 1.0) / 2.0,
    )


def _causal_score(row: Mapping[str, Any]) -> float:
    sig = _handle_signature(row)
    return float(sum(sig) / len(sig))


def _selector_payload(
    calibration_summary: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float,
    beta_neural: float,
) -> dict[str, Any]:
    handle_ids = tuple(calibration_summary)
    signatures = tuple(_handle_signature(calibration_summary[handle_id]) for handle_id in handle_ids)
    desired = torch.ones((1, len(signatures[0])), dtype=torch.float32)
    neural = torch.tensor(signatures, dtype=torch.float32)
    cost = cost_matrix(desired, neural, mode="squared")
    coupling = sinkhorn_one_sided_uot(cost, epsilon=float(epsilon), beta_neural=float(beta_neural), n_iter=300)
    weights = coupling[0]
    ranked = sorted(
        (
            {
                "handle_id": handle_id,
                "weight": float(weights[i]),
                "cost": float(cost[0, i]),
                "causal_score": _causal_score(calibration_summary[handle_id]),
                "signature": signatures[i],
            }
            for i, handle_id in enumerate(handle_ids)
        ),
        key=lambda row: (-float(row["weight"]), float(row["cost"])),
    )
    return {
        "feature_names": (
            "same_u_preserve",
            "opposite_u_flip",
            "wrong_position_failure",
            "wrong_content_failure",
            "wrong_length_failure",
            "same_u_low_delta",
            "opposite_u_shift_score",
        ),
        "desired_signature": tuple(float(x) for x in desired[0].tolist()),
        "handle_ids": handle_ids,
        "signatures": signatures,
        "cost": cost.tolist(),
        "coupling": coupling.tolist(),
        "ranked_handles": ranked,
        "note": (
            "This is a one-row PLOT/UOT selector for U = unmatched quote type. "
            "It is not interpreted as a rich variable-to-variable coupling."
        ),
    }


def _effect_ranked_payload(calibration_summary: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        (
            {
                "handle_id": handle_id,
                "opposite_u_flip_rate": _metric(row, "opposite_u_flip_rate"),
                "opposite_u_mean_source_signed_shift": _metric(row, "opposite_u_mean_source_signed_shift"),
                "same_u_preserve_rate": _metric(row, "same_u_preserve_rate"),
                "same_u_mean_abs_margin_delta": _metric(row, "same_u_mean_abs_margin_delta"),
            }
            for handle_id, row in calibration_summary.items()
        ),
        key=lambda row: (
            -float(row["opposite_u_mean_source_signed_shift"]),
            -float(row["opposite_u_flip_rate"]),
            float(row["same_u_mean_abs_margin_delta"]),
        ),
    )
    return {
        "ranked_handles": ranked,
        "note": (
            "Effect-ranked baseline: rank handles only by calibration opposite-U source-signed margin shift, "
            "without wrong-variable controls or PLOT/UOT costs."
        ),
    }


def _split_pairs(
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
) -> tuple[tuple[tuple[StringClosingExample, StringClosingExample], ...], tuple[tuple[StringClosingExample, StringClosingExample], ...]]:
    calibration_templates = {"assign", "print"}
    calibration = []
    heldout = []
    for pair in pairs:
        template_id = pair[0].template_id
        if template_id in calibration_templates:
            calibration.append(pair)
        else:
            heldout.append(pair)
    if calibration and heldout:
        return tuple(calibration), tuple(heldout)

    calibration = []
    heldout = []
    for idx, pair in enumerate(pairs):
        if idx % 2 == 0:
            calibration.append(pair)
        else:
            heldout.append(pair)
    return tuple(calibration), tuple(heldout)


def _write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    kept_pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    calibration_pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    heldout_pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    calibration_summary: Mapping[str, Mapping[str, Any]],
    heldout_summary: Mapping[str, Mapping[str, Any]],
    selector: Mapping[str, Any],
    effect_ranked: Mapping[str, Any],
) -> None:
    calibration_templates = sorted({pair[0].template_id for pair in calibration_pairs})
    heldout_templates = sorted({pair[0].template_id for pair in heldout_pairs})
    lines = [
        "# Unmatched Quote-Type Abstraction",
        "",
        "High-level causal model tested here:",
        "",
        "```text",
        "X -> U -> Y",
        "U = unmatched opening quote type in {single, double}",
        "Y = matching next closing quote",
        "```",
        "",
        f"- model: `{args.model}`",
        f"- kept prompt pairs: `{len(kept_pairs)}`",
        f"- calibration pairs: `{len(calibration_pairs)}`",
        f"- heldout pairs: `{len(heldout_pairs)}`",
        f"- calibration templates: `{', '.join(calibration_templates)}`",
        f"- heldout templates: `{', '.join(heldout_templates)}`",
        f"- max records per relation: `{args.max_records_per_relation}`",
        "",
        "A good `U` handle should preserve output under same-`U` resampling, flip under opposite-`U` resampling, and fail wrong-variable preservation tests.",
        "",
        "## PLOT/UOT Selector On Calibration",
        "",
        "| rank | handle | kind | weight | cost | causal score |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(selector["ranked_handles"], start=1):
        handle_id = str(row["handle_id"])
        summary = calibration_summary[handle_id]
        lines.append(
            f"| {rank} | `{handle_id}` | `{summary['handle_kind']}` | {row['weight']:.3f} | "
            f"{row['cost']:.3f} | {row['causal_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Effect-Ranked Baseline On Calibration",
            "",
            "| rank | handle | opposite flip | signed shift | same preserve | same abs delta |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(effect_ranked["ranked_handles"], start=1):
        lines.append(
            f"| {rank} | `{row['handle_id']}` | {row['opposite_u_flip_rate']:.3f} | "
            f"{row['opposite_u_mean_source_signed_shift']:.3f} | {row['same_u_preserve_rate']:.3f} | "
            f"{row['same_u_mean_abs_margin_delta']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Heldout Validation",
            "",
            "| handle | kind | same preserve | opposite flip | wrong position preserve | wrong content preserve | wrong length preserve | same abs delta | opposite signed shift |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ranked_ids = [str(row["handle_id"]) for row in selector["ranked_handles"]]
    remaining_ids = [handle_id for handle_id in heldout_summary if handle_id not in ranked_ids]
    for handle_id in [*ranked_ids, *remaining_ids]:
        row = heldout_summary[handle_id]
        lines.append(
            f"| `{handle_id}` | `{row['handle_kind']}` | "
            f"{_metric(row, 'same_u_preserve_rate'):.3f} | "
            f"{_metric(row, 'opposite_u_flip_rate'):.3f} | "
            f"{_metric(row, 'wrong_position_preserve_rate'):.3f} | "
            f"{_metric(row, 'wrong_content_preserve_rate'):.3f} | "
            f"{_metric(row, 'wrong_length_preserve_rate'):.3f} | "
            f"{_metric(row, 'same_u_mean_abs_margin_delta'):.3f} | "
            f"{_metric(row, 'opposite_u_mean_source_signed_shift'):.3f} |"
        )
    top_id = ranked_ids[0]
    top = heldout_summary[top_id]
    wrong_failure = 1.0 - (
        _metric(top, "wrong_position_preserve_rate")
        + _metric(top, "wrong_content_preserve_rate")
        + _metric(top, "wrong_length_preserve_rate")
    ) / 3.0
    lines.extend(
        [
            "",
            "## Current Interpretation",
            "",
            f"- Top calibration-selected handle: `{top_id}`.",
            f"- Heldout same-`U` preserve: `{_metric(top, 'same_u_preserve_rate'):.3f}`.",
            f"- Heldout opposite-`U` flip: `{_metric(top, 'opposite_u_flip_rate'):.3f}`.",
            f"- Heldout average wrong-variable failure: `{wrong_failure:.3f}`.",
            "",
            "The selector is intentionally treated as a sparse handle selector, not as a nondegenerate multi-variable PLOT coupling.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    pairs = build_effect_prompt_pairs(max_pairs=args.max_pairs)
    record_sites_by_id: dict[str, ChannelSite] = {}
    for handle in DEFAULT_HANDLES:
        for site in handle.sites():
            record_sites_by_id.setdefault(site.site_id, site)
    record_sites = tuple(record_sites_by_id.values())
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
    if len(kept_pairs) < 4:
        raise ValueError("too few pairs survived clean prediction and margin filtering")
    calibration_pairs, heldout_pairs = _split_pairs(kept_pairs)
    lookup = _example_lookup(kept_pairs)

    split_payloads = {}
    for split_name, split_pairs in (("calibration", calibration_pairs), ("heldout", heldout_pairs)):
        specs = _build_resampling_specs(
            split_pairs,
            runs,
            max_records_per_relation=args.max_records_per_relation,
        )
        records: list[dict[str, Any]] = []
        for handle in DEFAULT_HANDLES:
            print(f"{split_name}: patching {handle.handle_id}", flush=True)
            records.extend(
                _run_records_for_handle(
                    model=model,
                    handle=handle,
                    specs=specs,
                    examples=lookup,
                    runs=runs,
                    single_token_id=tokens["single"],
                    double_token_id=tokens["double"],
                )
            )
        split_payloads[split_name] = {
            "pairs": [pair[0].pair_id for pair in split_pairs],
            "specs": [spec.__dict__ for spec in specs],
            "records": records,
            "summary": _summarize_records(records),
        }

    selector = _selector_payload(
        split_payloads["calibration"]["summary"],
        epsilon=args.selector_epsilon,
        beta_neural=args.selector_beta,
    )
    effect_ranked = _effect_ranked_payload(split_payloads["calibration"]["summary"])
    payload = {
        "model_info": model_info,
        "quote_token_ids": tokens,
        "causal_model": {
            "variables": ["X", "U", "Y"],
            "edges": [["X", "U"], ["U", "Y"]],
            "U": "unmatched opening quote type in {single, double}",
            "Y": "matching next closing quote",
        },
        "input_pairs": len(pairs),
        "kept_pairs": len(kept_pairs),
        "min_abs_margin": args.min_abs_margin,
        "handles": [handle.__dict__ for handle in DEFAULT_HANDLES],
        "selector": selector,
        "effect_ranked_baseline": effect_ranked,
        "splits": split_payloads,
    }
    (args.out_dir / "unmatched_quote_abstraction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        args.out_dir / "unmatched_quote_abstraction.md",
        args=args,
        kept_pairs=kept_pairs,
        calibration_pairs=calibration_pairs,
        heldout_pairs=heldout_pairs,
        calibration_summary=split_payloads["calibration"]["summary"],
        heldout_summary=split_payloads["heldout"]["summary"],
        selector=selector,
        effect_ranked=effect_ranked,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "top_selector": selector["ranked_handles"][:5]}, indent=2))


if __name__ == "__main__":
    main()
