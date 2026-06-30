from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .artifacts import load_viz_data
from .effect_signatures import build_effect_prompt_pairs, collect_clean_runs as collect_quote_runs, filter_correct_pairs
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .run_bracket_counting_abstraction import (
    DEFAULT_HANDLES as BRACKET_HARD_HANDLES,
    DEFAULT_VIZ_PATH,
    _build_resampling_specs as build_bracket_specs,
    _clean_summary as bracket_clean_summary,
    _collect_runs as collect_bracket_runs,
    _load_released_examples,
    _metric as bracket_metric,
    _record_sites as bracket_record_sites,
    _run_records_for_handle as run_bracket_records_for_handle,
    _summarize_records as summarize_bracket_records,
)
from .run_singleton_soft_handle_abstraction import (
    _bracket_singleton_handles,
    _quote_singleton_handles,
    _run_bracket_soft_records,
    _run_quote_soft_records,
    _topk_weight_map,
)
from .run_unmatched_quote_abstraction import (
    CandidateHandle as QuoteHandle,
    DEFAULT_HANDLES as QUOTE_HARD_HANDLES,
    _build_resampling_specs as build_quote_specs,
    _example_lookup as quote_example_lookup,
    _metric as quote_metric,
    _run_records_for_handle as run_quote_records_for_handle,
    _split_pairs as split_quote_pairs,
    _summarize_records as summarize_quote_records,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


DEFAULT_QUOTE_HARD_JSON = Path(
    "eval/openai_sparse_plot/unmatched_quote_abstraction_csp_yolo1_template/unmatched_quote_abstraction.json"
)
DEFAULT_QUOTE_NODE_CSV = Path("eval/openai_sparse_plot/string_closing_prune_v2_64/string_closing_circuit_nodes.csv")
DEFAULT_BRACKET_HARD_JSON = Path(
    "eval/openai_sparse_plot/"
    "bracket_counting_abstraction_csp_yolo2_depth_vs_controls_flash_balanced_r8/"
    "bracket_counting_abstraction.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLOT matching with raw phi(y_swap)-phi(y_base) signatures.")
    parser.add_argument("--task", choices=("quote", "bracket", "both"), default="both")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/raw_delta_plot"))
    parser.add_argument("--quote-hard-json", type=Path, default=DEFAULT_QUOTE_HARD_JSON)
    parser.add_argument("--quote-node-csv", type=Path, default=DEFAULT_QUOTE_NODE_CSV)
    parser.add_argument(
        "--quote-candidate-source",
        choices=("node_csv", "interpreted12"),
        default="node_csv",
        help=(
            "Use singleton sites from --quote-node-csv by default. "
            "interpreted12 is a legacy/debug subset and must be requested explicitly."
        ),
    )
    parser.add_argument("--quote-site-id", action="append", default=None)
    parser.add_argument("--quote-max-sites", type=int, default=0, help="0 means all selected quote singleton sites.")
    parser.add_argument("--bracket-hard-json", type=Path, default=DEFAULT_BRACKET_HARD_JSON)
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
    parser.add_argument("--bracket-site-id", action="append", default=None)
    parser.add_argument("--bracket-max-sites", type=int, default=0, help="0 means all exported singleton sites.")
    parser.add_argument("--max-records-per-relation", type=int, default=6)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _parse_int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("empty integer grid")
    return values


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("empty float grid")
    return values


def _quote_handles_from_node_csv(
    path: Path,
    *,
    limit: int = 0,
    allowed_site_ids: set[str] | None = None,
) -> tuple[QuoteHandle, ...]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    handles: list[QuoteHandle] = []
    for row in rows:
        node_id = str(row["node_id"])
        if allowed_site_ids is not None and node_id not in allowed_site_ids:
            continue
        label = str(row.get("published_label") or row.get("node_kind") or node_id)
        handles.append(
            QuoteHandle(
                handle_id=node_id,
                label=label,
                node_ids=(node_id,),
                kind="singleton_exported_node",
            )
        )
        if limit > 0 and len(handles) >= limit:
            break
    if allowed_site_ids is not None:
        found = {handle.handle_id for handle in handles}
        missing = sorted(allowed_site_ids - found)
        if missing:
            raise ValueError(f"quote node CSV is missing requested site IDs: {missing}")
    return tuple(handles)


def _select_quote_singleton_handles(args: argparse.Namespace) -> tuple[QuoteHandle, ...]:
    allowed = None if args.quote_site_id is None else set(args.quote_site_id)
    if args.quote_candidate_source == "interpreted12":
        handles = _quote_singleton_handles()
        if allowed is not None:
            handles = tuple(handle for handle in handles if handle.handle_id in allowed)
            missing = sorted(allowed - {handle.handle_id for handle in handles})
            if missing:
                raise ValueError(f"interpreted quote site set is missing requested site IDs: {missing}")
        if args.quote_max_sites > 0:
            handles = handles[: int(args.quote_max_sites)]
        return handles
    return _quote_handles_from_node_csv(
        args.quote_node_csv,
        limit=int(args.quote_max_sites),
        allowed_site_ids=allowed,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _record_key(row: Mapping[str, Any]) -> tuple[str, str, str, str | None]:
    return (
        str(row["relation"]),
        str(row["base_example_id"]),
        str(row["source_example_id"]),
        None if row.get("wrong_variable") is None else str(row.get("wrong_variable")),
    )


def _quote_abstract_delta(row: Mapping[str, Any]) -> float:
    return float(row["source_sign"]) - float(row["base_sign"])


def _bracket_sign(close_count: int | float) -> int:
    return 1 if int(close_count) == 2 else -1


def _bracket_abstract_delta(row: Mapping[str, Any]) -> float:
    return float(_bracket_sign(row["source_close_count"]) - _bracket_sign(row["base_close_count"]))


def _neural_delta(row: Mapping[str, Any]) -> float:
    return float(row["patched_margin"]) - float(row["base_margin"])


def _raw_vectors_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], tuple[str, ...]]:
    by_handle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_handle[str(row["handle_id"])].append(row)
    if not by_handle:
        raise ValueError("no records")

    handle_ids = tuple(by_handle)
    reference_rows = by_handle[handle_ids[0]]
    coordinate_keys = tuple(_record_key(row) for row in reference_rows)
    abstract_fn = _quote_abstract_delta if task == "quote" else _bracket_abstract_delta
    abstract = tuple(float(abstract_fn(row)) for row in reference_rows)

    neural: dict[str, tuple[float, ...]] = {}
    for handle_id, rows in by_handle.items():
        by_key = {_record_key(row): row for row in rows}
        missing = [key for key in coordinate_keys if key not in by_key]
        if missing:
            raise ValueError(f"handle {handle_id} missing raw-delta coordinates: {missing[:3]}")
        neural[handle_id] = tuple(_neural_delta(by_key[key]) for key in coordinate_keys)
    feature_names = tuple(f"{relation}:{base}->{source}" for relation, base, source, _ in coordinate_keys)
    return abstract, neural, feature_names


def _raw_cost(abstract: Sequence[float], neural: Sequence[float], *, mode: str) -> float:
    return float(
        cost_matrix(
            torch.tensor([abstract], dtype=torch.float32),
            torch.tensor([neural], dtype=torch.float32),
            mode=mode,  # type: ignore[arg-type]
        )[0, 0]
    )


def _selector_payload_from_raw(
    *,
    abstract: Sequence[float],
    neural_by_id: Mapping[str, Sequence[float]],
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    site_ids = tuple(neural_by_id)
    abstract_tensor = torch.tensor([abstract], dtype=torch.float32)
    neural_tensor = torch.tensor([neural_by_id[site_id] for site_id in site_ids], dtype=torch.float32)

    squared_cost = cost_matrix(abstract_tensor, neural_tensor, mode="squared")
    squared_uot = sinkhorn_one_sided_uot(squared_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]

    cosine_cost = cost_matrix(abstract_tensor, neural_tensor, mode="cosine")
    cosine_uot = sinkhorn_one_sided_uot(cosine_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]
    cosine_similarity = 1.0 - cosine_cost[0]
    cosine_direct = cosine_similarity.clamp_min(0.0)
    if float(cosine_direct.sum()) <= 0.0:
        cosine_direct = torch.softmax(cosine_similarity, dim=0)
    else:
        cosine_direct = cosine_direct / cosine_direct.sum().clamp_min(1e-12)

    selector_specs = {
        "raw_squared_uot": {
            "cost_mode": "squared",
            "calibration_cost_mode": "squared",
            "coupling_rule": "one_sided_uot",
            "weights": squared_uot,
            "cost": squared_cost[0],
            "similarity": None,
        },
        "raw_cosine_uot": {
            "cost_mode": "cosine",
            "calibration_cost_mode": "cosine",
            "coupling_rule": "one_sided_uot",
            "weights": cosine_uot,
            "cost": cosine_cost[0],
            "similarity": cosine_similarity,
        },
        "raw_cosine_similarity": {
            "cost_mode": "cosine",
            "calibration_cost_mode": "cosine",
            "coupling_rule": "row_normalized_positive_cosine_similarity",
            "weights": cosine_direct,
            "cost": cosine_cost[0],
            "similarity": cosine_similarity,
        },
    }
    selectors = {}
    for name, spec in selector_specs.items():
        ranked = []
        for idx, site_id in enumerate(site_ids):
            ranked.append(
                {
                    "site_id": site_id,
                    "weight": float(spec["weights"][idx]),
                    "cost": float(spec["cost"][idx]),
                    "similarity": None if spec["similarity"] is None else float(spec["similarity"][idx]),
                    "raw_signature": tuple(float(x) for x in neural_by_id[site_id]),
                }
            )
        selectors[name] = {
            "cost_mode": spec["cost_mode"],
            "calibration_cost_mode": spec["calibration_cost_mode"],
            "coupling_rule": spec["coupling_rule"],
            "ranked_sites": sorted(ranked, key=lambda row: (-float(row["weight"]), float(row["cost"]))),
        }
    return {
        "abstract_signature": tuple(float(x) for x in abstract),
        "site_ids": site_ids,
        "selectors": selectors,
        "note": "Each coordinate is raw phi(y_swap)-phi(y_base). phi is binary output margin for the neural model and +/-1 class output for the abstract model.",
    }


def _behavior_summary(task: str, row: Mapping[str, Any]) -> dict[str, float]:
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


def _behavior_score(behavior: Mapping[str, float]) -> float:
    shift_score = (torch.tanh(torch.tensor(float(behavior["shift"]) / 10.0)).item() + 1.0) / 2.0
    return float(
        (
            float(behavior["same"])
            + float(behavior["flip"])
            + (1.0 - float(behavior["wrong_preserve"]))
            + shift_score
        )
        / 4.0
    )


def _hard_replay_from_json(path: Path, *, task: str, epsilon: float, beta: float) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration_records = payload["splits"]["calibration"]["records"]
    heldout_summary = payload["splits"]["heldout"]["summary"]
    abstract, neural, feature_names = _raw_vectors_from_records(calibration_records, task=task)
    selector = _selector_payload_from_raw(
        abstract=abstract,
        neural_by_id=neural,
        epsilon=epsilon,
        beta=beta,
    )
    for selector_payload in selector["selectors"].values():
        for row in selector_payload["ranked_sites"]:
            row["heldout"] = _behavior_summary(task, heldout_summary[row["site_id"]])
    return {
        "source_json": str(path),
        "feature_names": feature_names,
        "selector": selector,
    }


def _calibrate_soft_handle_raw(
    *,
    task: str,
    selector_name: str,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_by_id: Mapping[str, Any],
    abstract: Sequence[float],
    specs: Sequence[Any],
    run_soft_records: Callable[..., list[dict[str, Any]]],
    summarize_records: Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    cost_mode: str,
) -> dict[str, Any]:
    rows = []
    for k in k_grid:
        k_eff = min(int(k), len(ranked_sites))
        if k_eff <= 0:
            continue
        weights = _topk_weight_map(ranked_sites, k=k_eff)
        sites = tuple(site_by_id[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"{selector_name}_raw_top{k_eff}_lambda{strength:g}"
            records = run_soft_records(
                handle_id=handle_id,
                sites=sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
            )
            _, neural, _ = _raw_vectors_from_records(records, task=task)
            raw_signature = neural[handle_id]
            raw_cost = _raw_cost(abstract, raw_signature, mode=cost_mode)
            summary = summarize_records(records)[handle_id]
            rows.append(
                {
                    "selector": selector_name,
                    "handle_id": handle_id,
                    "k": k_eff,
                    "strength": float(strength),
                    "site_ids": tuple(weights),
                    "weights_by_site": weights,
                    "raw_signature": raw_signature,
                    "raw_cost": raw_cost,
                    "calibration_cost_mode": cost_mode,
                    "summary": summary,
                    "behavior": _behavior_summary(task, summary),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["raw_cost"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]


def _calibrate_soft_handle_behavior(
    *,
    task: str,
    selector_name: str,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_by_id: Mapping[str, Any],
    abstract: Sequence[float],
    specs: Sequence[Any],
    run_soft_records: Callable[..., list[dict[str, Any]]],
    summarize_records: Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    cost_mode: str,
) -> dict[str, Any]:
    rows = []
    for k in k_grid:
        k_eff = min(int(k), len(ranked_sites))
        if k_eff <= 0:
            continue
        weights = _topk_weight_map(ranked_sites, k=k_eff)
        sites = tuple(site_by_id[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"{selector_name}_behavior_top{k_eff}_lambda{strength:g}"
            records = run_soft_records(
                handle_id=handle_id,
                sites=sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
            )
            _, neural, _ = _raw_vectors_from_records(records, task=task)
            raw_signature = neural[handle_id]
            raw_cost = _raw_cost(abstract, raw_signature, mode=cost_mode)
            summary = summarize_records(records)[handle_id]
            behavior = _behavior_summary(task, summary)
            rows.append(
                {
                    "selector": selector_name,
                    "handle_id": handle_id,
                    "k": k_eff,
                    "strength": float(strength),
                    "site_ids": tuple(weights),
                    "weights_by_site": weights,
                    "raw_signature": raw_signature,
                    "raw_cost": raw_cost,
                    "calibration_cost_mode": cost_mode,
                    "summary": summary,
                    "behavior": behavior,
                    "behavior_score": _behavior_score(behavior),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            -float(row["behavior_score"]),
            float(row["raw_cost"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]


def _brute_force_singleton_behavior(
    *,
    task: str,
    summaries: Mapping[str, Mapping[str, Any]],
    abstract: Sequence[float],
    neural_by_id: Mapping[str, Sequence[float]],
    cost_mode: str = "cosine",
) -> dict[str, Any]:
    rows = []
    for site_id, summary in summaries.items():
        behavior = _behavior_summary(task, summary)
        raw_signature = neural_by_id[site_id]
        rows.append(
            {
                "selector": "brute_force",
                "handle_id": site_id,
                "site_ids": (site_id,),
                "weights_by_site": {site_id: 1.0},
                "raw_signature": tuple(float(x) for x in raw_signature),
                "raw_cost": _raw_cost(abstract, raw_signature, mode=cost_mode),
                "calibration_cost_mode": cost_mode,
                "summary": summary,
                "behavior": behavior,
                "behavior_score": _behavior_score(behavior),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["behavior_score"]),
            float(row["raw_cost"]),
            str(row["handle_id"]),
        ),
    )
    if not ranked:
        raise ValueError("no singleton summaries to brute-force")
    return {
        "ranked_sites": ranked,
        "selected": ranked[0],
        "note": "Ranks singleton sites by direct calibration intervention behavior, without PLOT/UOT ranking or top-K soft handles.",
    }


def _ground_truth_singleton_test_behavior(
    *,
    task: str,
    summaries: Mapping[str, Mapping[str, Any]],
    abstract: Sequence[float],
    neural_by_id: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    rows = []
    for site_id, summary in summaries.items():
        behavior = _behavior_summary(task, summary)
        raw_signature = neural_by_id[site_id]
        rows.append(
            {
                "selector": "ground_truth_singleton_test",
                "handle_id": site_id,
                "site_ids": (site_id,),
                "weights_by_site": {site_id: 1.0},
                "raw_signature": tuple(float(x) for x in raw_signature),
                "raw_squared_cost": _raw_cost(abstract, raw_signature, mode="squared"),
                "raw_cosine_cost": _raw_cost(abstract, raw_signature, mode="cosine"),
                "summary": summary,
                "behavior": behavior,
                "behavior_score": _behavior_score(behavior),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["behavior_score"]),
            float(row["raw_cosine_cost"]),
            str(row["handle_id"]),
        ),
    )
    if not ranked:
        raise ValueError("no singleton summaries for ground-truth test behavior")
    return {
        "ranked_sites": ranked,
        "selected": ranked[0],
        "split": "heldout",
        "note": (
            "Oracle diagnostic only: ranks every singleton site by heldout/test intervention behavior. "
            "This uses test labels and should not be treated as a calibration-time selector."
        ),
    }


def _quote_soft_run(
    args: argparse.Namespace,
    *,
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
) -> dict[str, Any]:
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
    handles = _select_quote_singleton_handles(args)
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
        print(f"quote raw singleton {handle.handle_id}", flush=True)
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
    abstract, neural, feature_names = _raw_vectors_from_records(singleton_records, task="quote")
    singleton_summary = summarize_quote_records(singleton_records)
    brute_force = _brute_force_singleton_behavior(
        task="quote",
        summaries=singleton_summary,
        abstract=abstract,
        neural_by_id=neural,
        cost_mode="cosine",
    )
    selector = _selector_payload_from_raw(
        abstract=abstract,
        neural_by_id=neural,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    site_by_id = {handle.handle_id: handle.sites()[0] for handle in handles}
    calibrated_raw_cost = {}
    calibrated_behavior = {}
    heldout = {}
    for selector_name, selector_payload in selector["selectors"].items():
        print(f"quote raw calibrating {selector_name}", flush=True)
        best_raw = _calibrate_soft_handle_raw(
            task="quote",
            selector_name=selector_name,
            ranked_sites=selector_payload["ranked_sites"],
            site_by_id=site_by_id,
            abstract=abstract,
            specs=calibration_specs,
            run_soft_records=lambda **kwargs: _run_quote_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
                **kwargs,
            ),
            summarize_records=summarize_quote_records,
            k_grid=k_grid,
            strength_grid=strength_grid,
            cost_mode=selector_payload["calibration_cost_mode"],
        )
        best_behavior = _calibrate_soft_handle_behavior(
            task="quote",
            selector_name=selector_name,
            ranked_sites=selector_payload["ranked_sites"],
            site_by_id=site_by_id,
            abstract=abstract,
            specs=calibration_specs,
            run_soft_records=lambda **kwargs: _run_quote_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
                **kwargs,
            ),
            summarize_records=summarize_quote_records,
            k_grid=k_grid,
            strength_grid=strength_grid,
            cost_mode=selector_payload["calibration_cost_mode"],
        )
        calibrated_raw_cost[selector_name] = best_raw
        calibrated_behavior[selector_name] = best_behavior
        heldout[selector_name] = {}
        for calibration_rule, best in (("raw_cost", best_raw), ("behavior", best_behavior)):
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
            held_abs, held_neural, held_features = _raw_vectors_from_records(heldout_records, task="quote")
            held_summary = summarize_quote_records(heldout_records)[best["handle_id"]]
            heldout[selector_name][calibration_rule] = {
                "raw_signature": held_neural[best["handle_id"]],
                "abstract_signature": held_abs,
                "feature_names": held_features,
                "raw_squared_cost": _raw_cost(held_abs, held_neural[best["handle_id"]], mode="squared"),
                "raw_cosine_cost": _raw_cost(held_abs, held_neural[best["handle_id"]], mode="cosine"),
                "summary": held_summary,
                "behavior": _behavior_summary("quote", held_summary),
            }
    brute_selected = brute_force["selected"]
    brute_handle = next(handle for handle in handles if handle.handle_id == brute_selected["handle_id"])
    brute_heldout_records = run_quote_records_for_handle(
        model=model,
        handle=brute_handle,
        specs=heldout_specs,
        examples=lookup,
        runs=runs,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
    )
    brute_held_abs, brute_held_neural, brute_held_features = _raw_vectors_from_records(
        brute_heldout_records,
        task="quote",
    )
    brute_held_summary = summarize_quote_records(brute_heldout_records)[brute_selected["handle_id"]]
    brute_force["heldout"] = {
        "raw_signature": brute_held_neural[brute_selected["handle_id"]],
        "abstract_signature": brute_held_abs,
        "feature_names": brute_held_features,
        "raw_squared_cost": _raw_cost(
            brute_held_abs,
            brute_held_neural[brute_selected["handle_id"]],
            mode="squared",
        ),
        "raw_cosine_cost": _raw_cost(
            brute_held_abs,
            brute_held_neural[brute_selected["handle_id"]],
            mode="cosine",
        ),
        "summary": brute_held_summary,
        "behavior": _behavior_summary("quote", brute_held_summary),
    }
    ground_truth_records = []
    for handle in handles:
        print(f"quote ground-truth singleton test {handle.handle_id}", flush=True)
        ground_truth_records.extend(
            run_quote_records_for_handle(
                model=model,
                handle=handle,
                specs=heldout_specs,
                examples=lookup,
                runs=runs,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
        )
    ground_truth_abs, ground_truth_neural, ground_truth_features = _raw_vectors_from_records(
        ground_truth_records,
        task="quote",
    )
    ground_truth = _ground_truth_singleton_test_behavior(
        task="quote",
        summaries=summarize_quote_records(ground_truth_records),
        abstract=ground_truth_abs,
        neural_by_id=ground_truth_neural,
    )
    ground_truth["abstract_signature"] = ground_truth_abs
    ground_truth["feature_names"] = ground_truth_features
    return {
        "model_info": model_info,
        "candidate_source": args.quote_candidate_source,
        "candidate_node_csv": str(args.quote_node_csv),
        "candidate_site_count": len(handles),
        "candidate_sites": [handle.__dict__ for handle in handles],
        "kept_pairs": len(kept_pairs),
        "calibration_pairs": [pair[0].pair_id for pair in calibration_pairs],
        "heldout_pairs": [pair[0].pair_id for pair in heldout_pairs],
        "raw_feature_names": feature_names,
        "selector": selector,
        "brute_force_singleton_behavior": brute_force,
        "ground_truth_singleton_test_behavior": ground_truth,
        "calibrated_soft_handles_raw_cost": calibrated_raw_cost,
        "calibrated_soft_handles_behavior": calibrated_behavior,
        "heldout_soft_summary": heldout,
    }


def _bracket_soft_run(
    args: argparse.Namespace,
    *,
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
) -> dict[str, Any]:
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
    runs = collect_bracket_runs(
        model,
        examples,
        sites=bracket_record_sites(handles),
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
        print(f"bracket raw singleton {handle.handle_id}", flush=True)
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
    abstract, neural, feature_names = _raw_vectors_from_records(singleton_records, task="bracket")
    selector = _selector_payload_from_raw(
        abstract=abstract,
        neural_by_id=neural,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    site_by_id = {handle.handle_id: handle.sites()[0] for handle in handles}
    calibrated_raw_cost = {}
    calibrated_behavior = {}
    heldout = {}
    for selector_name, selector_payload in selector["selectors"].items():
        print(f"bracket raw calibrating {selector_name}", flush=True)
        best_raw = _calibrate_soft_handle_raw(
            task="bracket",
            selector_name=selector_name,
            ranked_sites=selector_payload["ranked_sites"],
            site_by_id=site_by_id,
            abstract=abstract,
            specs=calibration_specs,
            run_soft_records=lambda **kwargs: _run_bracket_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
                **kwargs,
            ),
            summarize_records=summarize_bracket_records,
            k_grid=k_grid,
            strength_grid=strength_grid,
            cost_mode=selector_payload["calibration_cost_mode"],
        )
        best_behavior = _calibrate_soft_handle_behavior(
            task="bracket",
            selector_name=selector_name,
            ranked_sites=selector_payload["ranked_sites"],
            site_by_id=site_by_id,
            abstract=abstract,
            specs=calibration_specs,
            run_soft_records=lambda **kwargs: _run_bracket_soft_records(
                model=model,
                examples=lookup,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
                **kwargs,
            ),
            summarize_records=summarize_bracket_records,
            k_grid=k_grid,
            strength_grid=strength_grid,
            cost_mode=selector_payload["calibration_cost_mode"],
        )
        calibrated_raw_cost[selector_name] = best_raw
        calibrated_behavior[selector_name] = best_behavior
        heldout[selector_name] = {}
        for calibration_rule, best in (("raw_cost", best_raw), ("behavior", best_behavior)):
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
            held_abs, held_neural, held_features = _raw_vectors_from_records(heldout_records, task="bracket")
            held_summary = summarize_bracket_records(heldout_records)[best["handle_id"]]
            heldout[selector_name][calibration_rule] = {
                "raw_signature": held_neural[best["handle_id"]],
                "abstract_signature": held_abs,
                "feature_names": held_features,
                "raw_squared_cost": _raw_cost(held_abs, held_neural[best["handle_id"]], mode="squared"),
                "raw_cosine_cost": _raw_cost(held_abs, held_neural[best["handle_id"]], mode="cosine"),
                "summary": held_summary,
                "behavior": _behavior_summary("bracket", held_summary),
            }
    return {
        "model_info": model_info,
        "viz_path": args.bracket_viz_path,
        "clean": clean,
        "candidate_site_count": len(handles),
        "candidate_sites": [handle.__dict__ for handle in handles],
        "raw_feature_names": feature_names,
        "selector": selector,
        "calibrated_soft_handles_raw_cost": calibrated_raw_cost,
        "calibrated_soft_handles_behavior": calibrated_behavior,
        "heldout_soft_summary": heldout,
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Raw-Delta PLOT Abstraction Runs",
        "",
        "This version uses raw effect signatures. Each coordinate is exactly `phi(y_swap) - phi(y_base)` for one base/source intervention.",
        "",
        "`phi` for the neural model is the binary output margin. `phi` for the abstract model is the signed class output, so the abstract raw delta is `source_sign - base_sign`.",
        "",
    ]
    hard = payload.get("hard_replay", {})
    if hard:
        lines.extend(["## Hard-Handle Raw Replay", ""])
        for task_name, task in hard.items():
            lines.extend(
                [
                    f"### {task_name.title()}",
                    "",
                    f"- source JSON: `{task['source_json']}`",
                    "",
                    "| method | rank | handle | weight | cost | cosine sim | heldout same | heldout flip | heldout wrong-preserve |",
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for method, selector in task["selector"]["selectors"].items():
                for rank, row in enumerate(selector["ranked_sites"][:5], start=1):
                    held = row["heldout"]
                    sim = row["similarity"]
                    sim_text = "n/a" if sim is None else f"{sim:.3f}"
                    lines.append(
                        f"| `{method}` | {rank} | `{row['site_id']}` | {row['weight']:.3f} | "
                        f"{row['cost']:.3f} | {sim_text} | {held['same']:.3f} | {held['flip']:.3f} | "
                        f"{held['wrong_preserve']:.3f} |"
                    )
            lines.append("")
    soft = payload.get("soft_runs", {})
    if soft:
        lines.extend(["## Singleton Soft-Handle Raw Runs", ""])
        for task_name, task in soft.items():
            lines.extend(
                [
                    f"### {task_name.title()}",
                    "",
                    f"- candidate singleton sites: `{task['candidate_site_count']}`",
                ]
            )
            if task_name == "quote":
                lines.append(f"- kept quote pairs: `{task['kept_pairs']}`")
            else:
                lines.append(f"- clean accuracy: `{task['clean']['accuracy']:.3f}`")
            lines.extend(
                [
                    "",
                    "Top singleton sites by raw-delta coupling:",
                    "",
                    "| method | rank | site | weight | cost | cosine sim |",
                    "|---|---:|---|---:|---:|---:|",
                ]
            )
            for method, selector in task["selector"]["selectors"].items():
                for rank, row in enumerate(selector["ranked_sites"][:8], start=1):
                    sim = row["similarity"]
                    sim_text = "n/a" if sim is None else f"{sim:.3f}"
                    lines.append(
                        f"| `{method}` | {rank} | `{row['site_id']}` | {row['weight']:.3f} | "
                        f"{row['cost']:.3f} | {sim_text} |"
                    )
            brute_force = task.get("brute_force_singleton_behavior")
            if brute_force:
                selected = brute_force["selected"]
                held = brute_force["heldout"]
                held_behavior = held["behavior"]
                lines.extend(
                    [
                        "",
                        "Brute-force singleton behavior baseline:",
                        "",
                        "| rank | site | calibration behavior score | calibration same | calibration flip | calibration wrong-preserve | calibration shift | heldout raw cosine cost | heldout same | heldout flip | heldout wrong-preserve |",
                        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for rank, row in enumerate(brute_force["ranked_sites"][:8], start=1):
                    behavior = row["behavior"]
                    held_cosine = held["raw_cosine_cost"] if row["handle_id"] == selected["handle_id"] else None
                    held_same = held_behavior["same"] if row["handle_id"] == selected["handle_id"] else None
                    held_flip = held_behavior["flip"] if row["handle_id"] == selected["handle_id"] else None
                    held_wrong = held_behavior["wrong_preserve"] if row["handle_id"] == selected["handle_id"] else None
                    held_cosine_text = "n/a" if held_cosine is None else f"{held_cosine:.3f}"
                    held_same_text = "n/a" if held_same is None else f"{held_same:.3f}"
                    held_flip_text = "n/a" if held_flip is None else f"{held_flip:.3f}"
                    held_wrong_text = "n/a" if held_wrong is None else f"{held_wrong:.3f}"
                    lines.append(
                        f"| {rank} | `{row['handle_id']}` | {row['behavior_score']:.3f} | "
                        f"{behavior['same']:.3f} | {behavior['flip']:.3f} | "
                        f"{behavior['wrong_preserve']:.3f} | {behavior['shift']:.3f} | "
                        f"{held_cosine_text} | {held_same_text} | {held_flip_text} | {held_wrong_text} |"
                    )
            ground_truth = task.get("ground_truth_singleton_test_behavior")
            if ground_truth:
                lines.extend(
                    [
                        "",
                        "Ground-truth singleton heldout/test oracle:",
                        "",
                        "This ranks every singleton site by heldout/test intervention behavior; it is an oracle diagnostic, not a calibration-time selector.",
                        "",
                        "| rank | site | heldout behavior score | heldout same | heldout flip | heldout wrong-preserve | heldout shift | heldout raw cosine cost |",
                        "|---:|---|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for rank, row in enumerate(ground_truth["ranked_sites"][:8], start=1):
                    behavior = row["behavior"]
                    lines.append(
                        f"| {rank} | `{row['handle_id']}` | {row['behavior_score']:.3f} | "
                        f"{behavior['same']:.3f} | {behavior['flip']:.3f} | "
                        f"{behavior['wrong_preserve']:.3f} | {behavior['shift']:.3f} | "
                        f"{row['raw_cosine_cost']:.3f} |"
                    )
            lines.extend(
                [
                    "",
                    "Calibrated soft handles:",
                    "",
                    "| method | calibration | K | strength | calibration raw cost | behavior score | heldout raw cosine cost | heldout same | heldout flip | heldout wrong-preserve | sites |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for calibration_rule, key in (
                ("raw_cost", "calibrated_soft_handles_raw_cost"),
                ("behavior", "calibrated_soft_handles_behavior"),
            ):
                for method, best in task[key].items():
                    held = task["heldout_soft_summary"][method][calibration_rule]
                    behavior = held["behavior"]
                    sites = ", ".join(best["site_ids"])
                    behavior_score = best.get("behavior_score")
                    behavior_text = "n/a" if behavior_score is None else f"{behavior_score:.3f}"
                    lines.append(
                        f"| `{method}` | `{calibration_rule}` | {best['k']} | {best['strength']:.3f} | "
                        f"{best['raw_cost']:.3f} | {behavior_text} | {held['raw_cosine_cost']:.3f} | "
                        f"{behavior['same']:.3f} | {behavior['flip']:.3f} | {behavior['wrong_preserve']:.3f} | "
                        f"`{sites}` |"
                    )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)

    hard_replay = {}
    soft_runs = {}
    if args.task in {"quote", "both"}:
        hard_replay["quote"] = _hard_replay_from_json(
            args.quote_hard_json,
            task="quote",
            epsilon=args.selector_epsilon,
            beta=args.selector_beta,
        )
        soft_runs["quote"] = _quote_soft_run(args, k_grid=k_grid, strength_grid=strength_grid)
    if args.task in {"bracket", "both"}:
        hard_replay["bracket"] = _hard_replay_from_json(
            args.bracket_hard_json,
            task="bracket",
            epsilon=args.selector_epsilon,
            beta=args.selector_beta,
        )
        soft_runs["bracket"] = _bracket_soft_run(args, k_grid=k_grid, strength_grid=strength_grid)

    payload = {
        "max_records_per_relation": int(args.max_records_per_relation),
        "k_grid": tuple(k_grid),
        "strength_grid": tuple(strength_grid),
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "raw_signature_definition": {
            "neural_phi": "binary output margin",
            "abstract_phi": "signed class output in {-1,+1}",
            "coordinate": "phi(y_swap)-phi(y_base)",
        },
        "hard_replay": hard_replay,
        "soft_runs": soft_runs,
    }
    (args.out_dir / "raw_delta_plot_abstraction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.out_dir / "raw_delta_plot_abstraction.md", payload)

    compact: dict[str, Any] = {}
    for task_name, task in soft_runs.items():
        compact[task_name] = {}
        for calibration_rule, key in (
            ("raw_cost", "calibrated_soft_handles_raw_cost"),
            ("behavior", "calibrated_soft_handles_behavior"),
        ):
            compact[task_name][calibration_rule] = {}
            for method, best in task[key].items():
                compact[task_name][calibration_rule][method] = {
                    "k": best["k"],
                    "strength": best["strength"],
                    "sites": best["site_ids"],
                    "heldout": task["heldout_soft_summary"][method][calibration_rule]["behavior"],
                }
        brute_force = task.get("brute_force_singleton_behavior")
        if brute_force:
            compact[task_name]["brute_force"] = {
                "site": brute_force["selected"]["handle_id"],
                "calibration_behavior_score": brute_force["selected"]["behavior_score"],
                "calibration_behavior": brute_force["selected"]["behavior"],
                "heldout": brute_force["heldout"]["behavior"],
            }
        ground_truth = task.get("ground_truth_singleton_test_behavior")
        if ground_truth:
            compact[task_name]["ground_truth_singleton_test"] = {
                "site": ground_truth["selected"]["handle_id"],
                "heldout_behavior_score": ground_truth["selected"]["behavior_score"],
                "heldout": ground_truth["selected"]["behavior"],
            }
    print(json.dumps({"out_dir": str(args.out_dir), "soft_summary": compact}, indent=2))


if __name__ == "__main__":
    main()


