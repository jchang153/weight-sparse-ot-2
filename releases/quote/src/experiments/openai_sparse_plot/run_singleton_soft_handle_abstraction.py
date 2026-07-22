from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, run_with_weighted_group_patch
from .artifacts import load_viz_data
from .effect_signatures import (
    build_effect_prompt_pairs,
    collect_clean_runs as collect_quote_runs,
    filter_correct_pairs,
    site_patch_position,
)
from .interpreted_circuit import PAPER_BACKED_NODE_SPECS
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .run_bracket_counting_abstraction import (
    DEFAULT_VIZ_PATH,
    BracketExample,
    CandidateHandle as BracketHandle,
    _build_resampling_specs as build_bracket_specs,
    _clean_summary as bracket_clean_summary,
    _collect_runs as collect_bracket_runs,
    _handle_signature as bracket_signature,
    _load_released_examples,
    _metric as bracket_metric,
    _record_sites as bracket_record_sites,
    _run_records_for_handle as run_bracket_records_for_handle,
    _sign_from_margin as bracket_sign_from_margin,
    _summarize_records as summarize_bracket_records,
    bracket_margin,
)
from .run_unmatched_quote_abstraction import (
    CandidateHandle as QuoteHandle,
    _all_examples as all_quote_examples,
    _build_resampling_specs as build_quote_specs,
    _example_lookup as quote_example_lookup,
    _handle_signature as quote_signature,
    _metric as quote_metric,
    _quote_sign_from_margin,
    _run_records_for_handle as run_quote_records_for_handle,
    _split_pairs as split_quote_pairs,
    _summarize_records as summarize_quote_records,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids
from .schema import StringClosingExample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLOT soft top-K handles over singleton OpenAI circuit sites.")
    parser.add_argument("--task", choices=("quote", "bracket", "both"), default="both")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/singleton_soft_handles"))
    parser.add_argument("--quote-model", default="csp_yolo1")
    parser.add_argument("--quote-max-pairs", type=int, default=16)
    parser.add_argument("--quote-min-abs-margin", type=float, default=1.0)
    parser.add_argument("--bracket-model", default="csp_yolo2")
    parser.add_argument("--bracket-viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument(
        "--bracket-node-csv",
        type=Path,
        default=Path("eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"),
    )
    parser.add_argument("--bracket-site-id", action="append", default=None, help="Restrict bracket singleton menu to selected node IDs.")
    parser.add_argument("--bracket-max-sites", type=int, default=0, help="0 means all exported singleton sites.")
    parser.add_argument("--max-records-per-relation", type=int, default=6)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _parse_int_grid(text: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("integer grid cannot be empty")
    return vals


def _parse_float_grid(text: str) -> tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not vals:
        raise ValueError("float grid cannot be empty")
    return vals


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _selector_weights(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    signature_fn,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    site_ids = tuple(summary)
    signatures = tuple(signature_fn(summary[site_id]) for site_id in site_ids)
    desired = torch.ones((1, len(signatures[0])), dtype=torch.float32)
    neural = torch.tensor(signatures, dtype=torch.float32)

    squared_cost = cost_matrix(desired, neural, mode="squared")
    squared_uot = sinkhorn_one_sided_uot(squared_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]

    cosine_cost = cost_matrix(desired, neural, mode="cosine")
    cosine_uot = sinkhorn_one_sided_uot(cosine_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]
    cosine_similarity = 1.0 - cosine_cost[0]
    cosine_direct = cosine_similarity.clamp_min(0.0)
    if float(cosine_direct.sum()) <= 0.0:
        cosine_direct = torch.softmax(cosine_similarity, dim=0)
    else:
        cosine_direct = cosine_direct / cosine_direct.sum().clamp_min(1e-12)

    selectors = {
        "squared_uot": {
            "cost_mode": "squared",
            "coupling_rule": "one_sided_uot",
            "weights": squared_uot,
            "cost": squared_cost[0],
            "similarity": None,
        },
        "cosine_uot": {
            "cost_mode": "cosine",
            "coupling_rule": "one_sided_uot",
            "weights": cosine_uot,
            "cost": cosine_cost[0],
            "similarity": cosine_similarity,
        },
        "cosine_similarity": {
            "cost_mode": "cosine",
            "coupling_rule": "row_normalized_positive_cosine_similarity",
            "weights": cosine_direct,
            "cost": cosine_cost[0],
            "similarity": cosine_similarity,
        },
    }
    out = {}
    for name, selector in selectors.items():
        weights = selector["weights"]
        ranked = []
        for idx, site_id in enumerate(site_ids):
            ranked.append(
                {
                    "site_id": site_id,
                    "weight": float(weights[idx]),
                    "cost": float(selector["cost"][idx]),
                    "similarity": None
                    if selector["similarity"] is None
                    else float(selector["similarity"][idx]),
                    "signature": signatures[idx],
                }
            )
        out[name] = {
            "cost_mode": selector["cost_mode"],
            "coupling_rule": selector["coupling_rule"],
            "ranked_sites": sorted(ranked, key=lambda row: (-float(row["weight"]), float(row["cost"]))),
        }
    return {
        "site_ids": site_ids,
        "desired_signature": tuple(float(x) for x in desired[0].tolist()),
        "signatures": signatures,
        "selectors": out,
    }


def _topk_weight_map(ranked_sites: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, float]:
    chosen = list(ranked_sites[: int(k)])
    total = sum(float(row["weight"]) for row in chosen)
    if total <= 0.0:
        return {str(row["site_id"]): 1.0 / len(chosen) for row in chosen}
    return {str(row["site_id"]): float(row["weight"]) / total for row in chosen}


def _quote_positions_for_sites(sites: Sequence[ChannelSite], positions: Mapping[str, int | str]) -> dict[str, list[int]]:
    return {site.site_id: [site_patch_position(site, positions)] for site in sites}


def _bracket_positions_for_sites(sites: Sequence[ChannelSite], positions: Mapping[str, int]) -> dict[str, list[int]]:
    pos = int(positions["final"])
    return {site.site_id: [pos] for site in sites}


def _run_quote_soft_records(
    *,
    model: Any,
    handle_id: str,
    sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    specs: Sequence[Any],
    examples: Mapping[str, StringClosingExample],
    runs: Mapping[str, Any],
    single_token_id: int,
    double_token_id: int,
) -> list[dict[str, Any]]:
    records = []
    node_ids = tuple(site.site_id for site in sites)
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = runs[spec.base_id]
        source = runs[spec.source_id]
        patched_logits = run_with_weighted_group_patch(
            model,
            base.token_ids,
            sites=sites,
            source_cache=source.cache,
            positions_by_site=_quote_positions_for_sites(sites, base.positions),
            source_positions_by_site=_quote_positions_for_sites(sites, source.positions),
            weights_by_site=weights_by_site,
            strength=strength,
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
                "handle_id": handle_id,
                "handle_label": handle_id,
                "handle_kind": "soft_singleton_topk",
                "node_ids": node_ids,
                "weights_by_site": dict(weights_by_site),
                "strength": float(strength),
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


def _run_bracket_soft_records(
    *,
    model: Any,
    handle_id: str,
    sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    specs: Sequence[Any],
    examples: Mapping[str, BracketExample],
    runs: Mapping[str, Any],
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    records = []
    node_ids = tuple(site.site_id for site in sites)
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = runs[spec.base_id]
        source = runs[spec.source_id]
        patched_logits = run_with_weighted_group_patch(
            model,
            base.token_ids,
            sites=sites,
            source_cache=source.cache,
            positions_by_site=_bracket_positions_for_sites(sites, base.positions),
            source_positions_by_site=_bracket_positions_for_sites(sites, source.positions),
            weights_by_site=weights_by_site,
            strength=strength,
        )
        patched_margin = bracket_margin(
            patched_logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        base_sign = base_ex.sign()
        source_sign = source_ex.sign()
        patched_sign = bracket_sign_from_margin(patched_margin)
        records.append(
            {
                "handle_id": handle_id,
                "handle_label": handle_id,
                "handle_kind": "soft_singleton_topk",
                "node_ids": node_ids,
                "weights_by_site": dict(weights_by_site),
                "strength": float(strength),
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_example_id": base_ex.example_id,
                "source_example_id": source_ex.example_id,
                "base_tail": base_ex.tail,
                "source_tail": source_ex.tail,
                "base_depth": base_ex.depth,
                "source_depth": source_ex.depth,
                "base_close_count": base_ex.close_count,
                "source_close_count": source_ex.close_count,
                "base_margin": base.margin,
                "source_margin": source.margin,
                "patched_margin": patched_margin,
                "patched_sign": patched_sign,
                "patched_preserves_base_sign": patched_sign == base_sign,
                "patched_matches_source_sign": patched_sign == source_sign,
                "moves_toward_source_sign": (patched_margin - base.margin) * source_sign > 0,
                "source_signed_shift": (patched_margin - base.margin) * source_sign,
                "abs_margin_delta": abs(patched_margin - base.margin),
            }
        )
    return records


def _score_quote(row: Mapping[str, Any]) -> float:
    sig = quote_signature(row)
    return float(sum(sig) / len(sig))


def _score_bracket(row: Mapping[str, Any]) -> float:
    sig = bracket_signature(row)
    return float(sum(sig) / len(sig))


def _calibrate_soft_selector(
    *,
    selector_name: str,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_by_id: Mapping[str, ChannelSite],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    run_soft_records,
    summarize_records,
    score_fn,
    specs: Sequence[Any],
) -> dict[str, Any]:
    rows = []
    for k in k_grid:
        k_eff = min(int(k), len(ranked_sites))
        if k_eff <= 0:
            continue
        weights = _topk_weight_map(ranked_sites, k=k_eff)
        sites = tuple(site_by_id[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"{selector_name}_top{k_eff}_lambda{strength:g}"
            records = run_soft_records(
                handle_id=handle_id,
                sites=sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
            )
            summary = summarize_records(records)[handle_id]
            rows.append(
                {
                    "selector": selector_name,
                    "handle_id": handle_id,
                    "k": k_eff,
                    "strength": float(strength),
                    "site_ids": tuple(weights),
                    "weights_by_site": weights,
                    "score": score_fn(summary),
                    "summary": summary,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]


def _quote_singleton_handles() -> tuple[QuoteHandle, ...]:
    return tuple(
        QuoteHandle(
            handle_id=spec.node_id,
            label=spec.label,
            node_ids=(spec.node_id,),
            kind="singleton_openai_node",
        )
        for spec in PAPER_BACKED_NODE_SPECS
    )


def _bracket_singleton_handles(
    path: Path,
    *,
    limit: int = 0,
    allowed_site_ids: set[str] | None = None,
) -> tuple[BracketHandle, ...]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    handles = []
    for row in rows:
        node_id = row["node_id"]
        if allowed_site_ids is not None and node_id not in allowed_site_ids:
            continue
        ChannelSite.from_node_id(node_id)
        handles.append(
            BracketHandle(
                handle_id=node_id,
                label=row.get("published_label") or node_id,
                node_ids=(node_id,),
                kind="singleton_openai_node",
                position_role="final",
            )
        )
    if limit > 0:
        return tuple(handles[:limit])
    return tuple(handles)


def _run_quote(args: argparse.Namespace, out_dir: Path, *, k_grid: Sequence[int], strength_grid: Sequence[float]) -> dict[str, Any]:
    device = "cuda" if args.cuda else "cpu"
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.quote_model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    pairs = build_effect_prompt_pairs(max_pairs=args.quote_max_pairs)
    handles = _quote_singleton_handles()
    record_sites = tuple({site.site_id: site for handle in handles for site in handle.sites()}.values())
    runs = collect_quote_runs(
        model,
        enc,
        pairs,
        sites=record_sites,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    kept_pairs = filter_correct_pairs(pairs, runs, min_abs_margin=args.quote_min_abs_margin)
    calibration_pairs, heldout_pairs = split_quote_pairs(kept_pairs)
    lookup = quote_example_lookup(kept_pairs)
    calibration_specs = build_quote_specs(calibration_pairs, runs, max_records_per_relation=args.max_records_per_relation)
    heldout_specs = build_quote_specs(heldout_pairs, runs, max_records_per_relation=args.max_records_per_relation)

    singleton_records = []
    for handle in handles:
        print(f"quote calibration singleton {handle.handle_id}", flush=True)
        singleton_records.extend(
            run_quote_records_for_handle(
                model=model,
                handle=handle,
                specs=calibration_specs,
                examples=lookup,
                runs=runs,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
        )
    singleton_summary = summarize_quote_records(singleton_records)
    selectors = _selector_weights(
        singleton_summary,
        signature_fn=quote_signature,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    site_by_id = {handle.handle_id: handle.sites()[0] for handle in handles}

    calibrated = {}
    heldout = {}
    for selector_name, selector in selectors["selectors"].items():
        print(f"quote calibrating soft selector {selector_name}", flush=True)
        best = _calibrate_soft_selector(
            selector_name=selector_name,
            ranked_sites=selector["ranked_sites"],
            site_by_id=site_by_id,
            k_grid=k_grid,
            strength_grid=strength_grid,
            run_soft_records=lambda **kwargs: _run_quote_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
                **kwargs,
            ),
            summarize_records=summarize_quote_records,
            score_fn=_score_quote,
            specs=calibration_specs,
        )
        calibrated[selector_name] = best
        heldout_records = _run_quote_soft_records(
            model=model,
            handle_id=best["handle_id"],
            sites=tuple(site_by_id[site_id] for site_id in best["weights_by_site"]),
            weights_by_site=best["weights_by_site"],
            strength=best["strength"],
            specs=heldout_specs,
            examples=lookup,
            runs=runs,
            single_token_id=tokens["single"],
            double_token_id=tokens["double"],
        )
        heldout[selector_name] = summarize_quote_records(heldout_records)[best["handle_id"]]

    return {
        "task": "quote",
        "model_info": model_info,
        "kept_pairs": len(kept_pairs),
        "calibration_pairs": [pair[0].pair_id for pair in calibration_pairs],
        "heldout_pairs": [pair[0].pair_id for pair in heldout_pairs],
        "candidate_sites": [handle.__dict__ for handle in handles],
        "singleton_calibration_summary": singleton_summary,
        "selectors": selectors,
        "calibrated_soft_handles": calibrated,
        "heldout_soft_summary": heldout,
    }


def _run_bracket(args: argparse.Namespace, out_dir: Path, *, k_grid: Sequence[int], strength_grid: Sequence[float]) -> dict[str, Any]:
    device = "cuda" if args.cuda else "cpu"
    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.bracket_viz_path)
    examples = _load_released_examples(viz_data, enc)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    allowed_site_ids = None if args.bracket_site_id is None else set(args.bracket_site_id)
    handles = _bracket_singleton_handles(
        args.bracket_node_csv,
        limit=args.bracket_max_sites,
        allowed_site_ids=allowed_site_ids,
    )
    if allowed_site_ids is not None and len(handles) != len(allowed_site_ids):
        found = {handle.handle_id for handle in handles}
        missing = sorted(allowed_site_ids - found)
        raise ValueError(f"bracket site IDs not found in node CSV: {missing}")
    model, model_info = load_sparse_gpt_model(
        model_name=args.bracket_model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=True,
        grad_checkpointing=False,
    )
    record_sites = bracket_record_sites(handles)
    runs = collect_bracket_runs(
        model,
        examples,
        sites=record_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = bracket_clean_summary(examples, runs)
    lookup = {ex.example_id: ex for ex in examples}
    calibration_specs = build_bracket_specs(
        examples,
        split="calibration",
        max_records_per_relation=args.max_records_per_relation,
    )
    heldout_specs = build_bracket_specs(
        examples,
        split="heldout",
        max_records_per_relation=args.max_records_per_relation,
    )

    singleton_records = []
    for handle in handles:
        print(f"bracket calibration singleton {handle.handle_id}", flush=True)
        singleton_records.extend(
            run_bracket_records_for_handle(
                model=model,
                handle=handle,
                specs=calibration_specs,
                examples=lookup,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
        )
    singleton_summary = summarize_bracket_records(singleton_records)
    selectors = _selector_weights(
        singleton_summary,
        signature_fn=bracket_signature,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    site_by_id = {handle.handle_id: handle.sites()[0] for handle in handles}

    calibrated = {}
    heldout = {}
    for selector_name, selector in selectors["selectors"].items():
        print(f"bracket calibrating soft selector {selector_name}", flush=True)
        best = _calibrate_soft_selector(
            selector_name=selector_name,
            ranked_sites=selector["ranked_sites"],
            site_by_id=site_by_id,
            k_grid=k_grid,
            strength_grid=strength_grid,
            run_soft_records=lambda **kwargs: _run_bracket_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
                **kwargs,
            ),
            summarize_records=summarize_bracket_records,
            score_fn=_score_bracket,
            specs=calibration_specs,
        )
        calibrated[selector_name] = best
        heldout_records = _run_bracket_soft_records(
            model=model,
            handle_id=best["handle_id"],
            sites=tuple(site_by_id[site_id] for site_id in best["weights_by_site"]),
            weights_by_site=best["weights_by_site"],
            strength=best["strength"],
            specs=heldout_specs,
            examples=lookup,
            runs=runs,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        heldout[selector_name] = summarize_bracket_records(heldout_records)[best["handle_id"]]

    return {
        "task": "bracket",
        "model_info": model_info,
        "viz_path": args.bracket_viz_path,
        "clean": clean,
        "candidate_site_count": len(handles),
        "candidate_sites": [handle.__dict__ for handle in handles],
        "singleton_calibration_summary": singleton_summary,
        "selectors": selectors,
        "calibrated_soft_handles": calibrated,
        "heldout_soft_summary": heldout,
    }


def _metric_row(task: str, row: Mapping[str, Any]) -> dict[str, float]:
    if task == "quote":
        return {
            "same": quote_metric(row, "same_u_preserve_rate"),
            "flip": quote_metric(row, "opposite_u_flip_rate"),
            "wrong_preserve": _mean(
                [
                    quote_metric(row, "wrong_position_preserve_rate"),
                    quote_metric(row, "wrong_content_preserve_rate"),
                    quote_metric(row, "wrong_length_preserve_rate"),
                ]
            ),
            "shift": quote_metric(row, "opposite_u_mean_source_signed_shift"),
        }
    return {
        "same": bracket_metric(row, "same_depth_preserve_rate"),
        "flip": bracket_metric(row, "different_depth_flip_rate"),
        "wrong_preserve": _mean(
            [
                bracket_metric(row, "wrong_length_preserve_rate"),
                bracket_metric(row, "wrong_content_preserve_rate"),
            ]
        ),
        "shift": bracket_metric(row, "different_depth_mean_source_signed_shift"),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Singleton Soft-Handle PLOT Runs",
        "",
        "Candidate neural variables are single OpenAI-localized scalar sites. The coupling row over those singletons is converted to a top-K soft handle and calibrated only by `K` and patch strength.",
        "",
    ]
    for task_name, task in payload["tasks"].items():
        lines.extend(
            [
                f"## {task_name.title()}",
                "",
                f"- candidate singleton sites: `{len(task['candidate_sites'])}`",
            ]
        )
        if task_name == "quote":
            lines.extend(
                [
                    f"- kept quote pairs: `{task['kept_pairs']}`",
                    f"- calibration pairs: `{len(task['calibration_pairs'])}`",
                    f"- heldout pairs: `{len(task['heldout_pairs'])}`",
                ]
            )
        else:
            lines.extend(
                [
                    f"- clean accuracy: `{task['clean']['accuracy']:.3f}`",
                    f"- released samples: `{task['clean']['n']}`",
                ]
            )
        lines.extend(
            [
                "",
                "### Top singleton sites by coupling",
                "",
                "| method | rank | site | weight | cost | cosine sim |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for selector_name, selector in task["selectors"]["selectors"].items():
            for rank, row in enumerate(selector["ranked_sites"][:8], start=1):
                sim = row["similarity"]
                sim_text = "n/a" if sim is None else f"{sim:.3f}"
                lines.append(
                    f"| `{selector_name}` | {rank} | `{row['site_id']}` | "
                    f"{row['weight']:.3f} | {row['cost']:.3f} | {sim_text} |"
                )
        lines.extend(
            [
                "",
                "### Calibrated soft handles",
                "",
                "| method | K | strength | calibration score | heldout same | heldout flip | heldout wrong-preserve | heldout signed shift | sites |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for selector_name, best in task["calibrated_soft_handles"].items():
            heldout = _metric_row(task_name, task["heldout_soft_summary"][selector_name])
            sites = ", ".join(best["site_ids"])
            lines.append(
                f"| `{selector_name}` | {best['k']} | {best['strength']:.3f} | {best['score']:.3f} | "
                f"{heldout['same']:.3f} | {heldout['flip']:.3f} | {heldout['wrong_preserve']:.3f} | "
                f"{heldout['shift']:.3f} | `{sites}` |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)
    tasks = {}
    if args.task in {"quote", "both"}:
        quote_dir = args.out_dir / "quote"
        quote_dir.mkdir(parents=True, exist_ok=True)
        tasks["quote"] = _run_quote(args, quote_dir, k_grid=k_grid, strength_grid=strength_grid)
    if args.task in {"bracket", "both"}:
        bracket_dir = args.out_dir / "bracket"
        bracket_dir.mkdir(parents=True, exist_ok=True)
        tasks["bracket"] = _run_bracket(args, bracket_dir, k_grid=k_grid, strength_grid=strength_grid)

    payload = {
        "max_records_per_relation": int(args.max_records_per_relation),
        "k_grid": tuple(k_grid),
        "strength_grid": tuple(strength_grid),
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "tasks": tasks,
    }
    (args.out_dir / "singleton_soft_handle_abstraction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.out_dir / "singleton_soft_handle_abstraction.md", payload)
    compact = {
        task_name: {
            selector_name: {
                "k": row["k"],
                "strength": row["strength"],
                "site_ids": row["site_ids"],
                "heldout": _metric_row(task_name, task["heldout_soft_summary"][selector_name]),
            }
            for selector_name, row in task["calibrated_soft_handles"].items()
        }
        for task_name, task in tasks.items()
    }
    print(json.dumps({"out_dir": str(args.out_dir), "summary": compact}, indent=2))


if __name__ == "__main__":
    main()
