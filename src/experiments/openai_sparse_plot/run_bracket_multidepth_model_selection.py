from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import (
    ChannelSite,
    make_weighted_multi_channel_patch,
    record_activations,
)
from .bracket_multidepth import (
    DEFAULT_RELATIONS,
    MultiDepthBracketExample,
    MultiDepthResamplingSpec,
    build_relation_specs,
    generate_multidepth_examples,
    parse_depths,
    relation_counts,
)
from .model_selection import (
    CandidateModelSpec,
    score_candidate_model,
    scores_to_jsonable,
    select_simplest_passing,
)
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_D_CANDIDATE_SITES: tuple[str, ...] = (
    "7.mlp.act_in:1079",
    "7.mlp.act_in:1249",
    "4.attn.act_in:1249",
    "4.attn.q:1292",
    "4.attn.q:1284",
    "3.attn.act_in:1249",
    "2.attn.resid_delta:1249",
    "2.mlp.act_in:1249",
    "7.mlp.post_act:6561",
    "7.mlp.post_act:2511",
    "7.mlp.resid_delta:607",
    "7.mlp.resid_delta:1200",
)


@dataclass(frozen=True)
class MultiDepthRun:
    example: MultiDepthBracketExample
    token_ids: torch.Tensor
    logits: torch.Tensor
    cache: dict[str, torch.Tensor]
    margin: float
    predicted_close_count: int
    r_probe: float
    positions: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bracket multi-depth causal-model selection experiment.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--out-dir", type=Path, default=Path("results/bracket/causal_model_selection/bracket_multidepth"))
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--examples-per-depth", type=int, default=16)
    parser.add_argument("--max-records-per-relation", type=int, default=8)
    parser.add_argument("--k-grid", default="1,2,3,5")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--r-handle-site", action="append", default=None)
    parser.add_argument("--d-candidate-site", action="append", default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    return parser.parse_args()


def _parse_int_grid(text: str) -> tuple[int, ...]:
    vals = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not vals:
        raise ValueError("empty integer grid")
    return vals


def _parse_float_grid(text: str) -> tuple[float, ...]:
    vals = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not vals:
        raise ValueError("empty float grid")
    return vals


def _unique_sites(node_ids: Iterable[str], *, label: str | None = None) -> tuple[ChannelSite, ...]:
    out: dict[str, ChannelSite] = {}
    for node_id in node_ids:
        out.setdefault(node_id, ChannelSite.from_node_id(node_id, label=label))
    return tuple(out.values())


def _hook_regex(sites: Sequence[ChannelSite]) -> str:
    hooks = sorted({site.hook_key for site in sites})
    if not hooks:
        return "^$"
    return "^(?:" + "|".join(re.escape(hook) for hook in hooks) + ")$"


def _token_tensor(example: MultiDepthBracketExample, *, device: str) -> torch.Tensor:
    return torch.tensor(example.token_ids, dtype=torch.long, device=device).unsqueeze(0)


def _bracket_margin(logits: torch.Tensor, *, single_close_token_id: int, double_close_token_id: int) -> float:
    last = logits[0, -1]
    return float(last[double_close_token_id] - last[single_close_token_id])


def _site_value(cache: Mapping[str, torch.Tensor], site: ChannelSite, position: int) -> float:
    return float(cache[site.hook_key][0, int(position), int(site.channel)])


def _mean(values: Iterable[float | bool]) -> float:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _metric(row: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    value = float(value)
    if math.isnan(value):
        return default
    return value


def _sign_from_margin(margin: float) -> int:
    return 1 if float(margin) > 0 else -1


def _r_probe(cache: Mapping[str, torch.Tensor], r_sites: Sequence[ChannelSite], position: int) -> float:
    if not r_sites:
        return float("nan")
    return _mean(_site_value(cache, site, position) for site in r_sites)


def _collect_runs(
    model: Any,
    examples: Sequence[MultiDepthBracketExample],
    *,
    sites: Sequence[ChannelSite],
    r_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
) -> dict[str, MultiDepthRun]:
    runs: dict[str, MultiDepthRun] = {}
    for example in examples:
        token_ids = _token_tensor(example, device=device)
        logits, cache = record_activations(model, token_ids, sites)
        margin = _bracket_margin(
            logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        predicted = 2 if margin > 0 else 1
        final_position = len(example.token_ids) - 1
        runs[example.example_id] = MultiDepthRun(
            example=example,
            token_ids=token_ids,
            logits=logits.detach().cpu(),
            cache=cache,
            margin=margin,
            predicted_close_count=predicted,
            r_probe=_r_probe(cache, r_sites, final_position),
            positions={"final": final_position},
        )
    return runs


def _positions_for_sites(sites: Sequence[ChannelSite], position: int) -> dict[str, list[int]]:
    return {site.site_id: [int(position)] for site in sites}


def _run_weighted_patch_with_probe(
    model: Any,
    base: MultiDepthRun,
    source: MultiDepthRun,
    *,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    r_sites: Sequence[ChannelSite],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = make_weighted_multi_channel_patch(
        patch_sites,
        source_cache=source.cache,
        positions_by_site=_positions_for_sites(patch_sites, base.positions["final"]),
        source_positions_by_site=_positions_for_sites(patch_sites, source.positions["final"]),
        weights_by_site=weights_by_site,
        strength=float(strength),
    )
    with torch.no_grad():
        with hook_recorder(regex=_hook_regex(r_sites), interventions=interventions) as ctx:
            logits, _, _ = model(base.token_ids)
    return logits, {k: v.detach().cpu() for k, v in ctx.items()}


def _run_records_for_weighted_handle(
    *,
    model: Any,
    handle_id: str,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    runs: Mapping[str, MultiDepthRun],
    r_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = runs[spec.base_id]
        source = runs[spec.source_id]
        patched_logits, patched_cache = _run_weighted_patch_with_probe(
            model,
            base,
            source,
            patch_sites=patch_sites,
            weights_by_site=weights_by_site,
            strength=strength,
            r_sites=r_sites,
        )
        patched_margin = _bracket_margin(
            patched_logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        patched_sign = _sign_from_margin(patched_margin)
        base_sign = base_ex.sign()
        source_sign = source_ex.sign()
        patched_r_probe = _r_probe(patched_cache, r_sites, base.positions["final"])
        base_to_source_probe_gap = abs(source.r_probe - base.r_probe)
        patched_to_source_probe_gap = abs(source.r_probe - patched_r_probe)
        records.append(
            {
                "handle_id": handle_id,
                "node_ids": [site.site_id for site in patch_sites],
                "weights_by_site": dict(weights_by_site),
                "strength": float(strength),
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_example_id": base_ex.example_id,
                "source_example_id": source_ex.example_id,
                "base_depth": base_ex.depth,
                "source_depth": source_ex.depth,
                "base_close_count": base_ex.close_count,
                "source_close_count": source_ex.close_count,
                "base_context_family": base_ex.context_family,
                "source_context_family": source_ex.context_family,
                "base_surface_open_count": base_ex.surface_open_count,
                "source_surface_open_count": source_ex.surface_open_count,
                "base_margin": base.margin,
                "source_margin": source.margin,
                "patched_margin": patched_margin,
                "patched_sign": patched_sign,
                "patched_preserves_base_sign": patched_sign == base_sign,
                "patched_matches_source_sign": patched_sign == source_sign,
                "same_R": base_ex.close_count == source_ex.close_count,
                "output_success_for_R": (patched_sign == base_sign)
                if base_ex.close_count == source_ex.close_count
                else (patched_sign == source_sign),
                "base_r_probe": base.r_probe,
                "source_r_probe": source.r_probe,
                "patched_r_probe": patched_r_probe,
                "r_probe_source_gap": base_to_source_probe_gap,
                "r_probe_patched_gap": patched_to_source_probe_gap,
                "r_probe_moves_toward_source": (
                    base_to_source_probe_gap > 1e-6
                    and patched_to_source_probe_gap + 1e-6 < base_to_source_probe_gap
                ),
                "r_probe_abs_shift": abs(patched_r_probe - base.r_probe),
                "source_signed_output_shift": (patched_margin - base.margin) * source_sign,
            }
        )
    return records


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)
    same_r_rows = [row for row in records if bool(row["same_R"])]
    different_r_rows = [row for row in records if not bool(row["same_R"])]

    def rows(name: str) -> list[Mapping[str, Any]]:
        return by_relation.get(name, [])

    summary = {
        "records": len(records),
        "relation_counts": {key: len(val) for key, val in sorted(by_relation.items())},
        "same_R_output_success_rate": _mean(row["output_success_for_R"] for row in same_r_rows),
        "different_R_output_success_rate": _mean(row["output_success_for_R"] for row in different_r_rows),
        "same_D_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in rows("same_D")),
        "different_D_same_R_preserve_rate": _mean(
            row["patched_preserves_base_sign"] for row in rows("different_D_same_R")
        ),
        "different_D_same_R_probe_move_rate": _mean(
            row["r_probe_moves_toward_source"] for row in rows("different_D_same_R")
        ),
        "different_D_same_R_mean_probe_gap": _mean(row["r_probe_source_gap"] for row in rows("different_D_same_R")),
        "different_D_same_R_mean_probe_shift": _mean(row["r_probe_abs_shift"] for row in rows("different_D_same_R")),
        "different_D_different_R_flip_rate": _mean(
            row["patched_matches_source_sign"] for row in rows("different_D_different_R")
        ),
        "same_surface_output_success_rate": _mean(
            row["output_success_for_R"] for row in rows("same_surface_different_active_context")
        ),
        "same_surface_probe_move_rate": _mean(
            row["r_probe_moves_toward_source"] for row in rows("same_surface_different_active_context")
        ),
        "wrong_numeric_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in rows("wrong_numeric_content")),
        "wrong_tail_length_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in rows("wrong_tail_length")),
        "mean_source_signed_output_shift": _mean(row["source_signed_output_shift"] for row in records),
    }
    return summary


def _clean_summary(
    examples: Sequence[MultiDepthBracketExample],
    runs: Mapping[str, MultiDepthRun],
) -> dict[str, Any]:
    rows = []
    for ex in examples:
        run = runs[ex.example_id]
        rows.append(
            {
                "example_id": ex.example_id,
                "split": ex.split,
                "depth": ex.depth,
                "close_count": ex.close_count,
                "context_family": ex.context_family,
                "surface_open_count": ex.surface_open_count,
                "margin": run.margin,
                "r_probe": run.r_probe,
                "predicted_close_count": run.predicted_close_count,
                "correct": run.predicted_close_count == ex.close_count,
                "prompt_tail": ex.prompt[-120:],
            }
        )
    by_depth = {}
    for depth in sorted({row["depth"] for row in rows}):
        depth_rows = [row for row in rows if row["depth"] == depth]
        by_depth[str(depth)] = {
            "n": len(depth_rows),
            "accuracy": _mean(row["correct"] for row in depth_rows),
            "mean_margin": _mean(row["margin"] for row in depth_rows),
            "mean_r_probe": _mean(row["r_probe"] for row in depth_rows),
        }
    by_context = {}
    for context in sorted({row["context_family"] for row in rows}):
        context_rows = [row for row in rows if row["context_family"] == context]
        by_context[context] = {
            "n": len(context_rows),
            "accuracy": _mean(row["correct"] for row in context_rows),
            "mean_margin": _mean(row["margin"] for row in context_rows),
        }
    return {
        "n": len(rows),
        "accuracy": _mean(row["correct"] for row in rows),
        "by_depth": by_depth,
        "by_context": by_context,
        "rows": rows,
    }


def _depth_signature_selector(
    specs: Sequence[MultiDepthResamplingSpec],
    *,
    examples: Mapping[str, MultiDepthBracketExample],
    runs: Mapping[str, MultiDepthRun],
    d_sites: Sequence[ChannelSite],
    max_depth: int,
    epsilon: float,
    beta_neural: float,
) -> dict[str, Any]:
    if not specs:
        raise ValueError("cannot build selector without calibration specs")
    denom = max(1, int(max_depth) - 1)
    abstract = torch.tensor(
        [[(examples[spec.source_id].depth - examples[spec.base_id].depth) / denom for spec in specs]],
        dtype=torch.float32,
    )
    neural_rows = []
    for site in d_sites:
        vals = []
        for spec in specs:
            base = runs[spec.base_id]
            source = runs[spec.source_id]
            pos_base = base.positions["final"]
            pos_source = source.positions["final"]
            vals.append(_site_value(source.cache, site, pos_source) - _site_value(base.cache, site, pos_base))
        neural_rows.append(vals)
    neural = torch.tensor(neural_rows, dtype=torch.float32)
    cost = cost_matrix(abstract, neural, mode="centered_cosine")
    coupling = sinkhorn_one_sided_uot(cost, epsilon=float(epsilon), beta_neural=float(beta_neural), n_iter=300)
    weights = coupling[0]
    ranked = sorted(
        (
            {
                "site_id": site.site_id,
                "weight": float(weights[idx]),
                "cost": float(cost[0, idx]),
                "signature": [float(x) for x in neural[idx].tolist()],
            }
            for idx, site in enumerate(d_sites)
        ),
        key=lambda row: (-float(row["weight"]), float(row["cost"]), str(row["site_id"])),
    )
    return {
        "abstract_variable": "D_mid",
        "abstract_signature": [float(x) for x in abstract[0].tolist()],
        "candidate_site_ids": [site.site_id for site in d_sites],
        "cost_mode": "centered_cosine",
        "epsilon": float(epsilon),
        "beta_neural": float(beta_neural),
        "cost": cost.tolist(),
        "coupling": coupling.tolist(),
        "ranked_sites": ranked,
        "note": "Neural signatures are source-minus-base activation deltas at each candidate site.",
    }


def _weights_from_ranked(ranked: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, float]:
    chosen = list(ranked[: max(1, int(k))])
    total = sum(float(row.get("weight", 0.0)) for row in chosen)
    if total <= 0.0:
        return {str(row["site_id"]): 1.0 / len(chosen) for row in chosen}
    return {str(row["site_id"]): float(row["weight"]) / total for row in chosen}


def _calibration_score(summary: Mapping[str, Any], *, kind: str) -> float:
    if kind == "R":
        keys = (
            "same_R_output_success_rate",
            "different_R_output_success_rate",
            "wrong_numeric_preserve_rate",
            "wrong_tail_length_preserve_rate",
            "same_surface_output_success_rate",
        )
    elif kind == "D":
        keys = (
            "same_D_preserve_rate",
            "different_D_same_R_preserve_rate",
            "different_D_same_R_probe_move_rate",
            "different_D_different_R_flip_rate",
            "wrong_numeric_preserve_rate",
            "wrong_tail_length_preserve_rate",
        )
    else:
        raise ValueError(kind)
    return _mean(_metric(summary, key) for key in keys)


def _calibrate_r_handle(
    *,
    model: Any,
    r_sites: Sequence[ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    runs: Mapping[str, MultiDepthRun],
    r_probe_sites: Sequence[ChannelSite],
    strengths: Sequence[float],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    rows = []
    weights = {site.site_id: 1.0 / len(r_sites) for site in r_sites}
    for strength in strengths:
        records = _run_records_for_weighted_handle(
            model=model,
            handle_id=f"R_late_lambda{strength:g}",
            patch_sites=r_sites,
            weights_by_site=weights,
            strength=float(strength),
            specs=specs,
            examples=examples,
            runs=runs,
            r_sites=r_probe_sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        summary = _summarize_records(records)
        rows.append(
            {
                "handle_id": f"R_late_lambda{strength:g}",
                "site_ids": [site.site_id for site in r_sites],
                "weights_by_site": weights,
                "strength": float(strength),
                "summary": summary,
                "calibration_score": _calibration_score(summary, kind="R"),
            }
        )
    best = sorted(rows, key=lambda row: (-float(row["calibration_score"]), float(row["strength"])))[0]
    return {"grid": rows, "best": best}


def _calibrate_d_handle(
    *,
    model: Any,
    ranked_sites: Sequence[Mapping[str, Any]],
    d_site_lookup: Mapping[str, ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    runs: Mapping[str, MultiDepthRun],
    r_probe_sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    strengths: Sequence[float],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    rows = []
    for k in k_grid:
        weights = _weights_from_ranked(ranked_sites, k=int(k))
        patch_sites = tuple(d_site_lookup[site_id] for site_id in weights)
        for strength in strengths:
            records = _run_records_for_weighted_handle(
                model=model,
                handle_id=f"D_mid_top{k}_lambda{strength:g}",
                patch_sites=patch_sites,
                weights_by_site=weights,
                strength=float(strength),
                specs=specs,
                examples=examples,
                runs=runs,
                r_sites=r_probe_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            summary = _summarize_records(records)
            rows.append(
                {
                    "handle_id": f"D_mid_top{k}_lambda{strength:g}",
                    "site_ids": [site.site_id for site in patch_sites],
                    "weights_by_site": weights,
                    "k": int(k),
                    "strength": float(strength),
                    "summary": summary,
                    "calibration_score": _calibration_score(summary, kind="D"),
                }
            )
    best = sorted(
        rows,
        key=lambda row: (
            -float(row["calibration_score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]
    return {"grid": rows, "best": best}


def _score_bracket_models(
    *,
    clean: Mapping[str, Any],
    r_summary: Mapping[str, Any],
    d_summary: Mapping[str, Any],
) -> dict[str, Any]:
    clean_accuracy = _metric(clean, "accuracy")
    b0_metrics = {
        "clean_accuracy": clean_accuracy,
        "same_R_output_success": _metric(r_summary, "same_R_output_success_rate"),
        "different_R_output_success": _metric(r_summary, "different_R_output_success_rate"),
        "wrong_numeric_preserve": _metric(r_summary, "wrong_numeric_preserve_rate"),
        "wrong_tail_length_preserve": _metric(r_summary, "wrong_tail_length_preserve_rate"),
    }
    b1_metrics = {
        "clean_accuracy": clean_accuracy,
        "same_R_output_success": _metric(r_summary, "same_R_output_success_rate"),
        "different_R_output_success": _metric(r_summary, "different_R_output_success_rate"),
        "same_D_preserve": _metric(d_summary, "same_D_preserve_rate"),
        "different_D_same_R_preserve": _metric(d_summary, "different_D_same_R_preserve_rate"),
        "different_D_same_R_probe_move": _metric(d_summary, "different_D_same_R_probe_move_rate"),
        "different_D_different_R_flip": _metric(d_summary, "different_D_different_R_flip_rate"),
        "wrong_numeric_preserve": _metric(d_summary, "wrong_numeric_preserve_rate"),
        "wrong_tail_length_preserve": _metric(d_summary, "wrong_tail_length_preserve_rate"),
    }
    b2_metrics = {
        **b1_metrics,
        "same_surface_output_success": _metric(d_summary, "same_surface_output_success_rate"),
    }
    specs = [
        CandidateModelSpec(
            "B0",
            "X -> R -> Y",
            1,
            2,
            (
                "clean_accuracy",
                "same_R_output_success",
                "different_R_output_success",
                "wrong_numeric_preserve",
                "wrong_tail_length_preserve",
            ),
        ),
        CandidateModelSpec(
            "B1",
            "X -> D -> R -> Y",
            2,
            2 + len(d_summary.get("node_ids", ())),
            (
                "clean_accuracy",
                "same_R_output_success",
                "different_R_output_success",
                "same_D_preserve",
                "different_D_same_R_preserve",
                "different_D_same_R_probe_move",
                "different_D_different_R_flip",
                "wrong_numeric_preserve",
                "wrong_tail_length_preserve",
            ),
        ),
        CandidateModelSpec(
            "B2",
            "X -> E, D -> R -> Y",
            3,
            2 + len(d_summary.get("node_ids", ())),
            (
                "clean_accuracy",
                "same_R_output_success",
                "different_R_output_success",
                "same_D_preserve",
                "different_D_same_R_preserve",
                "different_D_same_R_probe_move",
                "different_D_different_R_flip",
                "wrong_numeric_preserve",
                "wrong_tail_length_preserve",
                "same_surface_output_success",
            ),
        ),
    ]
    metric_sets = {"B0": b0_metrics, "B1": b1_metrics, "B2": b2_metrics}
    scores = [score_candidate_model(spec, metric_sets[spec.model_id]) for spec in specs]
    selected = select_simplest_passing(scores)
    return {
        "scores": scores_to_jsonable(scores),
        "selected_model_id": None if selected is None else selected.model_id,
        "selected_label": None if selected is None else selected.label,
        "note": (
            "B1 is accepted only if a D handle changes the downstream R probe on same-R/different-D pairs "
            "while preserving binary output."
        ),
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    clean = payload["clean"]
    r_best = payload["R_late"]["heldout"]
    d_best = payload["D_mid"]["heldout"]
    selected = payload["model_selection"]["selected_model_id"]
    lines = [
        "# Bracket Multi-Depth Model Selection",
        "",
        "Question: can the localized bracket circuit support a richer active-depth variable `D`, or only the saturated binary readout `R`?",
        "",
        f"- clean examples: `{clean['n']}`",
        f"- clean accuracy: `{clean['accuracy']:.3f}`",
        f"- selected bracket model: `{selected}`",
        "",
        "## Clean Behavior By Depth",
        "",
        "| depth | n | accuracy | mean margin | mean R probe |",
        "|---:|---:|---:|---:|---:|",
    ]
    for depth, row in clean["by_depth"].items():
        lines.append(
            f"| {depth} | {row['n']} | {row['accuracy']:.3f} | {row['mean_margin']:.3f} | {row['mean_r_probe']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Model Scores",
            "",
            "| model | variables | sites | pass | score | failed metrics |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for row in payload["model_selection"]["scores"]:
        failed = ", ".join(row["failed_metrics"]) if row["failed_metrics"] else ""
        lines.append(
            f"| `{row['model_id']}` {row['label']} | {row['variable_count']} | {row['neural_site_count']} | "
            f"{'yes' if row['passed'] else 'no'} | {row['score']:.3f} | `{failed}` |"
        )
    lines.extend(
        [
            "",
            "## R_late Heldout Validation",
            "",
            f"- sites: `{', '.join(r_best['site_ids'])}`",
            f"- strength: `{r_best['strength']:.3f}`",
            f"- same-R output success: `{r_best['summary']['same_R_output_success_rate']:.3f}`",
            f"- different-R output success: `{r_best['summary']['different_R_output_success_rate']:.3f}`",
            f"- wrong numeric preserve: `{r_best['summary']['wrong_numeric_preserve_rate']:.3f}`",
            f"- wrong tail-length preserve: `{r_best['summary']['wrong_tail_length_preserve_rate']:.3f}`",
            "",
            "## D_mid Heldout Validation",
            "",
            f"- sites: `{', '.join(d_best['site_ids'])}`",
            f"- strength: `{d_best['strength']:.3f}`",
            f"- same-D preserve: `{d_best['summary']['same_D_preserve_rate']:.3f}`",
            f"- different-D/same-R preserve: `{d_best['summary']['different_D_same_R_preserve_rate']:.3f}`",
            f"- different-D/same-R R-probe move: `{d_best['summary']['different_D_same_R_probe_move_rate']:.3f}`",
            f"- different-D/different-R flip: `{d_best['summary']['different_D_different_R_flip_rate']:.3f}`",
            f"- same-surface/context output success: `{d_best['summary']['same_surface_output_success_rate']:.3f}`",
            "",
            "## D_mid PLOT Ranking",
            "",
            "| rank | site | weight | cost |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(payload["D_mid"]["selector"]["ranked_sites"][:12], start=1):
        lines.append(f"| {rank} | `{row['site_id']}` | {row['weight']:.3f} | {row['cost']:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if selected == "B0":
        lines.append(
            "The multi-depth run supports the saturated binary readout `R` but does not validate a separate `D` handle."
        )
    elif selected == "B1":
        lines.append(
            "The multi-depth run validates a richer `D -> R` abstraction: same-R/different-D examples move the downstream probe while preserving the binary output."
        )
    elif selected == "B2":
        lines.append("The multi-depth run indicates context gating is needed in addition to depth.")
    else:
        lines.append("No bracket candidate model passed the configured thresholds.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    depths = parse_depths(args.depths)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)
    r_node_ids = tuple(args.r_handle_site or ("7.mlp.post_act:4133", "7.mlp.resid_delta:2041"))
    r_sites = _unique_sites(r_node_ids, label="R_late")
    r_site_ids = {site.site_id for site in r_sites}
    d_node_ids = tuple(args.d_candidate_site or DEFAULT_D_CANDIDATE_SITES)
    d_node_ids = tuple(node_id for node_id in d_node_ids if node_id not in r_site_ids and not node_id.startswith("final_resid:"))
    d_sites = _unique_sites(d_node_ids, label="D_mid_candidate")
    if not d_sites:
        raise ValueError("no D candidate sites after exclusions")

    device = "cuda" if args.cuda else "cpu"
    flash = not bool(args.no_flash)
    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    examples = generate_multidepth_examples(
        enc,
        depths=depths,
        examples_per_depth=args.examples_per_depth,
    )
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=flash,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    record_sites = _unique_sites([site.site_id for site in (*r_sites, *d_sites)])
    runs = _collect_runs(
        model,
        examples,
        sites=record_sites,
        r_sites=r_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = _clean_summary(examples, runs)
    lookup = {ex.example_id: ex for ex in examples}
    calibration_specs = build_relation_specs(
        examples,
        split="calibration",
        max_records_per_relation=args.max_records_per_relation,
        relations=DEFAULT_RELATIONS,
    )
    heldout_specs = build_relation_specs(
        examples,
        split="heldout",
        max_records_per_relation=args.max_records_per_relation,
        relations=DEFAULT_RELATIONS,
    )
    print(f"calibration relation counts: {relation_counts(calibration_specs)}", flush=True)
    print(f"heldout relation counts: {relation_counts(heldout_specs)}", flush=True)

    print("ranking D candidates", flush=True)
    d_selector = _depth_signature_selector(
        calibration_specs,
        examples=lookup,
        runs=runs,
        d_sites=d_sites,
        max_depth=max(depths),
        epsilon=args.selector_epsilon,
        beta_neural=args.selector_beta,
    )
    d_site_lookup = {site.site_id: site for site in d_sites}

    print("calibrating R_late handle", flush=True)
    r_calibration = _calibrate_r_handle(
        model=model,
        r_sites=r_sites,
        specs=calibration_specs,
        examples=lookup,
        runs=runs,
        r_probe_sites=r_sites,
        strengths=strength_grid,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    print("calibrating D_mid handle", flush=True)
    d_calibration = _calibrate_d_handle(
        model=model,
        ranked_sites=d_selector["ranked_sites"],
        d_site_lookup=d_site_lookup,
        specs=calibration_specs,
        examples=lookup,
        runs=runs,
        r_probe_sites=r_sites,
        k_grid=k_grid,
        strengths=strength_grid,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )

    print("running heldout selected handles", flush=True)
    r_best = r_calibration["best"]
    r_records = _run_records_for_weighted_handle(
        model=model,
        handle_id=str(r_best["handle_id"]),
        patch_sites=r_sites,
        weights_by_site=r_best["weights_by_site"],
        strength=float(r_best["strength"]),
        specs=heldout_specs,
        examples=lookup,
        runs=runs,
        r_sites=r_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    d_best = d_calibration["best"]
    d_patch_sites = tuple(d_site_lookup[site_id] for site_id in d_best["weights_by_site"])
    d_records = _run_records_for_weighted_handle(
        model=model,
        handle_id=str(d_best["handle_id"]),
        patch_sites=d_patch_sites,
        weights_by_site=d_best["weights_by_site"],
        strength=float(d_best["strength"]),
        specs=heldout_specs,
        examples=lookup,
        runs=runs,
        r_sites=r_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    r_heldout = {
        **{key: value for key, value in r_best.items() if key != "summary"},
        "records": r_records,
        "summary": _summarize_records(r_records),
    }
    d_heldout = {
        **{key: value for key, value in d_best.items() if key != "summary"},
        "records": d_records,
        "summary": {
            **_summarize_records(d_records),
            "node_ids": [site.site_id for site in d_patch_sites],
        },
    }
    model_selection = _score_bracket_models(
        clean=clean,
        r_summary=r_heldout["summary"],
        d_summary=d_heldout["summary"],
    )
    payload = {
        "model": args.model,
        "model_info": model_info,
        "depths": list(depths),
        "examples_per_depth": int(args.examples_per_depth),
        "max_records_per_relation": int(args.max_records_per_relation),
        "single_close_token_id": single_close_token_id,
        "double_close_token_id": double_close_token_id,
        "relations": list(DEFAULT_RELATIONS),
        "clean": clean,
        "examples": [ex.__dict__ for ex in examples],
        "calibration_specs": [spec.__dict__ for spec in calibration_specs],
        "heldout_specs": [spec.__dict__ for spec in heldout_specs],
        "R_late": {
            "site_ids": [site.site_id for site in r_sites],
            "calibration": r_calibration,
            "heldout": r_heldout,
        },
        "D_mid": {
            "candidate_site_ids": [site.site_id for site in d_sites],
            "selector": d_selector,
            "calibration": d_calibration,
            "heldout": d_heldout,
        },
        "model_selection": model_selection,
    }
    json_path = args.out_dir / "bracket_multidepth_model_selection.json"
    md_path = args.out_dir / "bracket_multidepth_model_selection.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "clean_accuracy": clean["accuracy"],
                "selected_model": model_selection["selected_model_id"],
                "r_best": r_best["handle_id"],
                "d_best": d_best["handle_id"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
