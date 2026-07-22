from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, run_with_group_patch
from .candidate_causal_models import CANDIDATE_MODELS, CandidateCausalModel, CandidateVariable
from .effect_signatures import (
    build_effect_prompt_pairs,
    collect_clean_runs,
    filter_correct_pairs,
    interpreted_channel_sites,
    site_patch_position,
)
from .plot_matching import MatchingResult, cost_matrix, fit_matching, sinkhorn_one_sided_uot
from .run_plot_matching import load_payload, table_from_payload
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids
from .schema import EffectSignatureTable


CANONICAL_CIRCUIT_NODES: frozenset[str] = frozenset(
    {
        "0.mlp.post_act:863",
        "0.mlp.post_act:2790",
        "0.mlp.resid_delta:460",
        "10.attn.act_in:460",
        "10.attn.v:663",
        "10.attn.resid_delta:83",
        "final_resid:83",
    }
)

SITE_STAGE: dict[str, str] = {
    "0.mlp.post_act:863": "opening",
    "0.mlp.post_act:2790": "opening",
    "0.mlp.resid_delta:460": "storage",
    "0.mlp.resid_delta:985": "routing",
    "10.attn.act_in:460": "storage",
    "10.attn.act_in:985": "routing",
    "10.attn.act_in:1013": "routing",
    "10.attn.q:657": "routing",
    "10.attn.k:657": "routing",
    "10.attn.v:663": "copy",
    "10.attn.resid_delta:83": "logit",
    "final_resid:83": "output",
}

STAGE_ORDER: dict[str, float] = {
    "opening": 0.0,
    "storage": 1.0,
    "copy": 2.0,
    "logit": 3.0,
    "output": 4.0,
    "path": 2.0,
    "routing": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep candidate causal models for string-closing PLOT/IIA.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument(
        "--base-table-json",
        type=Path,
        default=Path("results/quote/effect_signatures_simple_chain_csp_yolo1_pairs8/effect_signature_table.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/candidate_model_sweep"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _mean_rows(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("need at least one row")
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tuple(float(x) for x in tensor.mean(dim=0))


def _profile_rows(base_table: EffectSignatureTable) -> dict[str, tuple[float, ...]]:
    by_name = {
        name: tuple(row)
        for name, row in zip(base_table.abstract_variable_ids, base_table.abstract_signatures)
    }
    return {
        **by_name,
        "CopiedQuoteType": by_name["CopiedQuoteTypeAtFinalPosition"],
        "OutputPreference": by_name["ClosingQuoteLogitPreference"],
        "AttentionReadValueCopy": by_name["CopiedQuoteTypeAtFinalPosition"],
        "FullQuotePath": _mean_rows(
            [
                by_name["StoredQuoteType"],
                by_name["CopiedQuoteTypeAtFinalPosition"],
                by_name["ClosingQuoteLogitPreference"],
                by_name["Output"],
            ]
        ),
        "StoredAndCopiedQuoteType": _mean_rows(
            [by_name["StoredQuoteType"], by_name["CopiedQuoteTypeAtFinalPosition"]]
        ),
        "StoredAndOutputPreference": _mean_rows(
            [by_name["StoredQuoteType"], by_name["ClosingQuoteLogitPreference"]]
        ),
    }


def candidate_effect_table(model: CandidateCausalModel, base_table: EffectSignatureTable) -> EffectSignatureTable:
    profiles = _profile_rows(base_table)
    rows = []
    for variable in model.variables:
        if variable.profile_id not in profiles:
            raise KeyError(f"missing signature profile {variable.profile_id!r}")
        rows.append(profiles[variable.profile_id])
    return EffectSignatureTable.from_sequences(
        abstract_variable_ids=model.variable_ids(),
        neural_site_ids=base_table.neural_site_ids,
        abstract_signatures=rows,
        neural_signatures=base_table.neural_signatures,
        feature_names=base_table.feature_names,
        metadata={
            **base_table.metadata,
            "candidate_model_id": model.model_id,
            "candidate_model_label": model.label,
            "candidate_edges": model.edges,
        },
    )


def expected_rank_audit_for_model(result: MatchingResult, model: CandidateCausalModel) -> dict[str, dict[str, Any]]:
    expected = {var.variable_id: set(var.node_ids) for var in model.variables}
    audit = {}
    for row_idx, row_label in enumerate(result.row_labels):
        vals, idx = torch.sort(result.coupling[row_idx], descending=True)
        ranked_labels = [result.col_labels[int(i)] for i in idx]
        family = expected.get(row_label, set())
        best_rank = None
        best_site = None
        best_weight = None
        family_mass = 0.0
        for site in family:
            if site not in result.col_labels:
                continue
            col = result.col_labels.index(site)
            rank = ranked_labels.index(site) + 1
            weight = float(result.coupling[row_idx, col])
            family_mass += weight
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_site = site
                best_weight = weight
        audit[row_label] = {
            "expected_sites": sorted(family),
            "top_site": ranked_labels[0],
            "top_weight": float(vals[0]),
            "best_expected_rank": best_rank,
            "best_expected_site": best_site,
            "best_expected_weight": best_weight,
            "expected_family_mass": family_mass,
            "expected_top1": ranked_labels[0] in family,
        }
    return audit


def _stage_penalty(table: EffectSignatureTable, model: CandidateCausalModel, *, penalty: float = 0.6) -> torch.Tensor:
    variables = {var.variable_id: var for var in model.variables}
    rows = []
    for variable_id in table.abstract_variable_ids:
        variable = variables[variable_id]
        row = []
        var_stage_value = STAGE_ORDER.get(variable.stage, 2.0)
        for site_id in table.neural_site_ids:
            site_stage = SITE_STAGE.get(site_id, "routing")
            site_stage_value = STAGE_ORDER.get(site_stage, 2.0)
            if site_id in variable.node_ids:
                row.append(0.0)
            else:
                row.append(min(float(penalty), 0.18 * abs(var_stage_value - site_stage_value) + 0.12))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def stage_aware_match_for_model(
    table: EffectSignatureTable,
    model: CandidateCausalModel,
    *,
    cost_mode: str = "centered_cosine",
    epsilon: float = 0.25,
    beta_neural: float = 0.25,
    stage_penalty: float = 0.6,
) -> MatchingResult:
    base_cost = cost_matrix(
        torch.tensor(table.abstract_signatures, dtype=torch.float32),
        torch.tensor(table.neural_signatures, dtype=torch.float32),
        mode=cost_mode,  # type: ignore[arg-type]
    )
    cost = base_cost + _stage_penalty(table, model, penalty=stage_penalty)
    coupling = sinkhorn_one_sided_uot(cost, epsilon=epsilon, beta_neural=beta_neural, n_iter=300)
    return MatchingResult(
        cost=cost.detach().cpu(),
        coupling=coupling.detach().cpu(),
        method="stage_aware_uot",
        row_labels=table.abstract_variable_ids,
        col_labels=table.neural_site_ids,
    )


def _quote_sign_from_margin(margin: float) -> int:
    return 1 if float(margin) > 0 else -1


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(x) for x in values) / len(values))


def _group_key(node_ids: Sequence[str]) -> str:
    return " + ".join(node_ids)


def _positions_for_sites(sites: Sequence[ChannelSite], positions: Mapping[str, int | str]) -> dict[str, list[int]]:
    return {site.site_id: [site_patch_position(site, positions)] for site in sites}


def evaluate_group_iia(
    *,
    model_obj: Any,
    runs: Mapping[str, Any],
    examples: Sequence[Any],
    variable: CandidateVariable,
    single_token_id: int,
    double_token_id: int,
) -> dict[str, Any]:
    sites = tuple(ChannelSite.from_node_id(node_id) for node_id in variable.node_ids)
    records = []
    for base_ex in examples:
        base = runs[base_ex.example_id]
        base_sign = base_ex.sign()
        for source_ex in examples:
            if source_ex.example_id == base_ex.example_id:
                continue
            source = runs[source_ex.example_id]
            source_sign = source_ex.sign()
            relation = "same_quote_type" if source_sign == base_sign else "different_quote_type"
            patched_logits = run_with_group_patch(
                model_obj,
                base.token_ids,
                sites=sites,
                source_cache=source.cache,
                positions_by_site=_positions_for_sites(sites, base.positions),
                source_positions_by_site=_positions_for_sites(sites, source.positions),
            )
            patched_margin = binary_quote_margin(
                patched_logits.detach().cpu(),
                single_token_id=single_token_id,
                double_token_id=double_token_id,
            )
            patched_sign = _quote_sign_from_margin(patched_margin)
            records.append(
                {
                    "relation": relation,
                    "base_example_id": base_ex.example_id,
                    "source_example_id": source_ex.example_id,
                    "base_sign": base_sign,
                    "source_sign": source_sign,
                    "base_margin": base.margin,
                    "source_margin": source.margin,
                    "patched_margin": patched_margin,
                    "patched_sign": patched_sign,
                    "same_preserve": patched_sign == base_sign,
                    "different_flip": patched_sign == source_sign,
                    "moves_toward_source": (patched_margin - base.margin) * source_sign > 0,
                }
            )
    same = [row for row in records if row["relation"] == "same_quote_type"]
    different = [row for row in records if row["relation"] == "different_quote_type"]
    same_correct = sum(bool(row["same_preserve"]) for row in same)
    different_flip = sum(bool(row["different_flip"]) for row in different)
    different_move = sum(bool(row["moves_toward_source"]) for row in different)
    strict_correct = same_correct + different_flip
    move_correct = same_correct + different_move
    return {
        "variable_id": variable.variable_id,
        "label": variable.label,
        "node_ids": variable.node_ids,
        "role": variable.role,
        "same_records": len(same),
        "different_records": len(different),
        "same_preserve_count": same_correct,
        "different_flip_count": different_flip,
        "different_move_count": different_move,
        "same_preserve_accuracy": same_correct / max(1, len(same)),
        "different_flip_accuracy": different_flip / max(1, len(different)),
        "different_move_accuracy": different_move / max(1, len(different)),
        "strict_iia_accuracy": strict_correct / max(1, len(records)),
        "move_or_preserve_accuracy": move_correct / max(1, len(records)),
        "mean_same_abs_margin_delta": _mean([abs(row["patched_margin"] - row["base_margin"]) for row in same]),
        "mean_different_source_signed_shift": _mean(
            [(row["patched_margin"] - row["base_margin"]) * row["source_sign"] for row in different]
        ),
        "records": records,
    }


def _all_examples(pairs: Sequence[tuple[Any, Any]]) -> tuple[Any, ...]:
    out = []
    for left, right in pairs:
        out.extend([left, right])
    return tuple(out)


def model_accuracy(variable_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    internal = [row for row in variable_summaries if row.get("role") != "observed_output"]

    def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total_same = sum(int(row["same_records"]) for row in rows)
        total_diff = sum(int(row["different_records"]) for row in rows)
        total = total_same + total_diff
        strict = sum(
            int(row["same_preserve_count"]) + int(row["different_flip_count"])
            for row in rows
        )
        move = sum(
            int(row["same_preserve_count"]) + int(row["different_move_count"])
            for row in rows
        )
        return {
            "same_records": total_same,
            "different_records": total_diff,
            "total_records": total,
            "strict_iia_accuracy": strict / max(1, total),
            "move_or_preserve_accuracy": move / max(1, total),
            "mean_variable_strict_iia_accuracy": _mean([float(row["strict_iia_accuracy"]) for row in rows]),
            "mean_variable_move_or_preserve_accuracy": _mean(
                [float(row["move_or_preserve_accuracy"]) for row in rows]
            ),
        }

    total_same = sum(int(row["same_records"]) for row in variable_summaries)
    total_diff = sum(int(row["different_records"]) for row in variable_summaries)
    total = total_same + total_diff
    strict = sum(
        int(row["same_preserve_count"]) + int(row["different_flip_count"])
        for row in variable_summaries
    )
    move = sum(
        int(row["same_preserve_count"]) + int(row["different_move_count"])
        for row in variable_summaries
    )
    full = {
        "same_records": total_same,
        "different_records": total_diff,
        "total_records": total,
        "strict_iia_accuracy": strict / max(1, total),
        "move_or_preserve_accuracy": move / max(1, total),
        "mean_variable_strict_iia_accuracy": _mean([float(row["strict_iia_accuracy"]) for row in variable_summaries]),
        "mean_variable_move_or_preserve_accuracy": _mean(
            [float(row["move_or_preserve_accuracy"]) for row in variable_summaries]
        ),
    }
    internal_metrics = _aggregate(internal)
    return {
        **full,
        "internal_strict_iia_accuracy": internal_metrics["strict_iia_accuracy"],
        "internal_move_or_preserve_accuracy": internal_metrics["move_or_preserve_accuracy"],
        "internal_mean_variable_strict_iia_accuracy": internal_metrics["mean_variable_strict_iia_accuracy"],
        "internal_variable_count": len(internal),
    }


def canonical_coverage(variables: Sequence[CandidateVariable]) -> dict[str, Any]:
    covered = {node_id for variable in variables for node_id in variable.node_ids if node_id in CANONICAL_CIRCUIT_NODES}
    return {
        "covered_nodes": sorted(covered),
        "missing_nodes": sorted(CANONICAL_CIRCUIT_NODES - covered),
        "canonical_coverage": len(covered) / len(CANONICAL_CIRCUIT_NODES),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"

    base_payload = load_payload(args.base_table_json)
    base_table = table_from_payload(base_payload)
    base_table.validate()

    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    model_obj, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    pairs = build_effect_prompt_pairs(max_pairs=args.max_pairs)
    record_sites = interpreted_channel_sites(include_post_act=True)
    runs = collect_clean_runs(
        model_obj,
        enc,
        pairs,
        sites=record_sites,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    kept_pairs = filter_correct_pairs(pairs, runs, min_abs_margin=args.min_abs_margin)
    if not kept_pairs:
        raise ValueError("no prompt pairs survived clean prediction and margin filtering")
    examples = _all_examples(kept_pairs)

    group_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    results = []
    for candidate in CANDIDATE_MODELS:
        print(f"evaluating {candidate.model_id}", flush=True)
        table = candidate_effect_table(candidate, base_table)
        unconstrained = fit_matching(
            table,
            method="uot",
            cost_mode="centered_cosine",
            epsilon=0.25,
            beta_neural=0.25,
            n_iter=300,
        )
        stage_aware = stage_aware_match_for_model(table, candidate)

        variable_iia = []
        for variable in candidate.variables:
            key = tuple(variable.node_ids)
            if key not in group_cache:
                group_cache[key] = evaluate_group_iia(
                    model_obj=model_obj,
                    runs=runs,
                    examples=examples,
                    variable=variable,
                    single_token_id=tokens["single"],
                    double_token_id=tokens["double"],
                )
            summary = dict(group_cache[key])
            summary["variable_id"] = variable.variable_id
            summary["label"] = variable.label
            summary["role"] = variable.role
            variable_iia.append(summary)

        results.append(
            {
                "model_id": candidate.model_id,
                "label": candidate.label,
                "notes": candidate.notes,
                "edges": candidate.edges,
                "variable_count": candidate.variable_count,
                "native_node_count": candidate.native_node_count,
                "canonical_coverage": canonical_coverage(candidate.variables),
                "variables": [
                    {
                        "variable_id": var.variable_id,
                        "label": var.label,
                        "node_ids": var.node_ids,
                        "profile_id": var.profile_id,
                        "stage": var.stage,
                        "role": var.role,
                    }
                    for var in candidate.variables
                ],
                "plot_unconstrained": {
                    "top_matches": unconstrained.top_matches(top_k=4),
                    "expected_rank_audit": expected_rank_audit_for_model(unconstrained, candidate),
                },
                "plot_stage_aware": {
                    "top_matches": stage_aware.top_matches(top_k=4),
                    "expected_rank_audit": expected_rank_audit_for_model(stage_aware, candidate),
                },
                "iia_by_variable": variable_iia,
                "model_accuracy": model_accuracy(variable_iia),
            }
        )

    ranked = sorted(
        results,
        key=lambda row: (
            -float(row["model_accuracy"]["internal_strict_iia_accuracy"]),
            -float(row["canonical_coverage"]["canonical_coverage"]),
            int(row["variable_count"]),
            int(row["native_node_count"]),
        ),
    )
    payload = {
        "model_info": model_info,
        "base_table_json": str(args.base_table_json),
        "input_pairs": len(pairs),
        "kept_pairs": len(kept_pairs),
        "examples": len(examples),
        "min_abs_margin": float(args.min_abs_margin),
        "quote_token_ids": tokens,
        "results": results,
        "ranked_model_ids": [row["model_id"] for row in ranked],
    }
    (args.out_dir / "candidate_model_sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Candidate Causal Model Sweep", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- kept prompt pairs: `{len(kept_pairs)}` / `{len(pairs)}`")
    lines.append(f"- examples: `{len(examples)}`")
    lines.append("- IIA uses all ordered base/source pairs among retained examples.")
    lines.extend(["", "## Model Summary", ""])
    for row in ranked:
        acc = row["model_accuracy"]
        cov = row["canonical_coverage"]
        lines.append(
            f"- `{row['model_id']}` ({row['variable_count']} vars, {row['native_node_count']} native nodes): "
            f"strict IIA `{acc['strict_iia_accuracy']:.3f}`, "
            f"internal strict `{acc['internal_strict_iia_accuracy']:.3f}`, "
            f"coverage `{cov['canonical_coverage']:.3f}`, "
            f"move/preserve `{acc['move_or_preserve_accuracy']:.3f}`"
        )
    lines.extend(["", "## Per-Model Details", ""])
    for row in results:
        lines.append(f"### {row['model_id']}: {row['label']}")
        lines.append(
            f"- size: `{row['variable_count']}` variables, `{row['native_node_count']}` native nodes"
        )
        lines.append(
            f"- strict IIA: `{row['model_accuracy']['strict_iia_accuracy']:.3f}`; "
            f"internal strict IIA: `{row['model_accuracy']['internal_strict_iia_accuracy']:.3f}`; "
            f"coverage: `{row['canonical_coverage']['canonical_coverage']:.3f}`; "
            f"move/preserve: `{row['model_accuracy']['move_or_preserve_accuracy']:.3f}`"
        )
        lines.append("- stage-aware PLOT top matches:")
        for var_id, matches in row["plot_stage_aware"]["top_matches"].items():
            top = matches[0]
            lines.append(f"  - `{var_id}` -> `{top[0]}` ({top[1]:.3f})")
        lines.append("- IIA by variable:")
        for var in row["iia_by_variable"]:
            lines.append(
                f"  - `{var['variable_id']}` {list(var['node_ids'])}: "
                f"same `{var['same_preserve_accuracy']:.3f}`, "
                f"diff flip `{var['different_flip_accuracy']:.3f}`, "
                f"diff move `{var['different_move_accuracy']:.3f}`, "
                f"strict `{var['strict_iia_accuracy']:.3f}`"
            )
        lines.append("")
    (args.out_dir / "candidate_model_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote candidate model sweep to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
