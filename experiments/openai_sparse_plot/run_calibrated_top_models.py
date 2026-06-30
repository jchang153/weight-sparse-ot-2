from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_causal_models import candidate_model_by_id
from .effect_signatures import (
    build_effect_prompt_pairs,
    build_effect_signature_table,
    collect_clean_runs,
    filter_correct_pairs,
    interpreted_channel_sites,
    write_effect_signature_table,
)
from .run_candidate_model_sweep import (
    candidate_effect_table,
    canonical_coverage,
    evaluate_group_iia,
    expected_rank_audit_for_model,
    model_accuracy,
    stage_aware_match_for_model,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


TOP_MODEL_IDS: tuple[str, ...] = (
    "m5_supernode_path_3",
    "m7_internal_path_supernode_3",
    "m6_two_supernodes_4",
)

COST_MODES: tuple[str, ...] = ("centered_cosine", "cosine")
EPSILONS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)
BETAS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)
STAGE_PENALTIES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate top OpenAI sparse PLOT causal models.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/calibrated_top_models"))
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _split_pairs() -> dict[str, tuple[Any, ...]]:
    pairs = build_effect_prompt_pairs(max_pairs=None)
    calibration = tuple(pair for pair in pairs if pair[0].template_id in {"assign", "print"})
    heldout = tuple(pair for pair in pairs if pair[0].template_id in {"paren_assign", "handler_arg"})
    if not calibration or not heldout:
        raise ValueError("expected nonempty calibration and heldout prompt-pair splits")
    return {"calibration": calibration, "heldout": heldout}


def _all_examples(pairs: Sequence[tuple[Any, Any]]) -> tuple[Any, ...]:
    out = []
    for left, right in pairs:
        out.extend([left, right])
    return tuple(out)


def _audit_metrics(audit: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    rows = list(audit.values())
    if not rows:
        return {"top1": 0.0, "mean_mass": 0.0, "mean_best_rank": 0.0}
    ranks = [float(row["best_expected_rank"] or 999.0) for row in rows]
    return {
        "top1": sum(1.0 if bool(row["expected_top1"]) else 0.0 for row in rows) / len(rows),
        "mean_mass": sum(float(row["expected_family_mass"]) for row in rows) / len(rows),
        "mean_best_rank": sum(ranks) / len(ranks),
    }


def _score_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row["calib_top1"]),
        float(row["calib_mean_mass"]),
        -float(row["calib_mean_best_rank"]),
        -float(row["stage_penalty"]),
        -float(row["epsilon"]),
    )


def _calibration_grid(model_obj: Any, calib_table: Any, heldout_table: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calib_candidate = candidate_effect_table(model_obj, calib_table)
    heldout_candidate = candidate_effect_table(model_obj, heldout_table)
    rows = []
    for cost_mode in COST_MODES:
        for epsilon in EPSILONS:
            for beta_neural in BETAS:
                for stage_penalty in STAGE_PENALTIES:
                    calib_result = stage_aware_match_for_model(
                        calib_candidate,
                        model_obj,
                        cost_mode=cost_mode,
                        epsilon=epsilon,
                        beta_neural=beta_neural,
                        stage_penalty=stage_penalty,
                    )
                    heldout_result = stage_aware_match_for_model(
                        heldout_candidate,
                        model_obj,
                        cost_mode=cost_mode,
                        epsilon=epsilon,
                        beta_neural=beta_neural,
                        stage_penalty=stage_penalty,
                    )
                    calib_audit = expected_rank_audit_for_model(calib_result, model_obj)
                    heldout_audit = expected_rank_audit_for_model(heldout_result, model_obj)
                    calib_metrics = _audit_metrics(calib_audit)
                    heldout_metrics = _audit_metrics(heldout_audit)
                    rows.append(
                        {
                            "model_id": model_obj.model_id,
                            "cost_mode": cost_mode,
                            "epsilon": epsilon,
                            "beta_neural": beta_neural,
                            "stage_penalty": stage_penalty,
                            "calib_top1": calib_metrics["top1"],
                            "calib_mean_mass": calib_metrics["mean_mass"],
                            "calib_mean_best_rank": calib_metrics["mean_best_rank"],
                            "heldout_top1": heldout_metrics["top1"],
                            "heldout_mean_mass": heldout_metrics["mean_mass"],
                            "heldout_mean_best_rank": heldout_metrics["mean_best_rank"],
                            "calib_top_matches": calib_result.top_matches(top_k=4),
                            "heldout_top_matches": heldout_result.top_matches(top_k=4),
                            "calib_expected_rank_audit": calib_audit,
                            "heldout_expected_rank_audit": heldout_audit,
                        }
                    )
    best = max(rows, key=_score_key)
    return rows, best


def _write_grid_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "model_id",
        "cost_mode",
        "epsilon",
        "beta_neural",
        "stage_penalty",
        "calib_top1",
        "calib_mean_mass",
        "calib_mean_best_rank",
        "heldout_top1",
        "heldout_mean_mass",
        "heldout_mean_best_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _heldout_iia(
    *,
    model_obj: Any,
    candidate: Any,
    heldout_pairs: Sequence[tuple[Any, Any]],
    enc: Any,
    tokens: Mapping[str, int],
    min_abs_margin: float,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    record_sites = interpreted_channel_sites(include_post_act=True)
    runs = collect_clean_runs(
        model_obj,
        enc,
        heldout_pairs,
        sites=record_sites,
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
    )
    kept_pairs = filter_correct_pairs(heldout_pairs, runs, min_abs_margin=min_abs_margin)
    examples = _all_examples(kept_pairs)
    summaries = []
    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for variable in candidate.variables:
        key = tuple(variable.node_ids)
        if key not in cache:
            cache[key] = evaluate_group_iia(
                model_obj=model_obj,
                runs=runs,
                examples=examples,
                variable=variable,
                single_token_id=tokens["single"],
                double_token_id=tokens["double"],
            )
        row = dict(cache[key])
        row["variable_id"] = variable.variable_id
        row["label"] = variable.label
        row["role"] = variable.role
        summaries.append(row)
    return summaries, model_accuracy(summaries), len(kept_pairs)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    sparse_model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)
    splits = _split_pairs()

    print("building calibration signatures", flush=True)
    calib_table, calib_diagnostics = build_effect_signature_table(
        sparse_model,
        enc,
        pairs=splits["calibration"],
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
        min_abs_margin=args.min_abs_margin,
    )
    write_effect_signature_table(calib_table, calib_diagnostics, out_dir=args.out_dir / "calibration_signatures")

    print("building heldout signatures", flush=True)
    heldout_table, heldout_diagnostics = build_effect_signature_table(
        sparse_model,
        enc,
        pairs=splits["heldout"],
        single_token_id=tokens["single"],
        double_token_id=tokens["double"],
        device=device,
        min_abs_margin=args.min_abs_margin,
    )
    write_effect_signature_table(heldout_table, heldout_diagnostics, out_dir=args.out_dir / "heldout_signatures")

    all_grid_rows = []
    results = []
    for model_id in TOP_MODEL_IDS:
        candidate = candidate_model_by_id(model_id)
        print(f"calibrating {model_id}", flush=True)
        grid_rows, best = _calibration_grid(candidate, calib_table, heldout_table)
        all_grid_rows.extend(grid_rows)
        iia_rows, iia_accuracy, kept_heldout = _heldout_iia(
            model_obj=sparse_model,
            candidate=candidate,
            heldout_pairs=splits["heldout"],
            enc=enc,
            tokens=tokens,
            min_abs_margin=args.min_abs_margin,
            device=device,
        )
        results.append(
            {
                "model_id": candidate.model_id,
                "label": candidate.label,
                "variable_count": candidate.variable_count,
                "native_node_count": candidate.native_node_count,
                "canonical_coverage": canonical_coverage(candidate.variables),
                "best_config": {
                    key: best[key]
                    for key in (
                        "cost_mode",
                        "epsilon",
                        "beta_neural",
                        "stage_penalty",
                        "calib_top1",
                        "calib_mean_mass",
                        "calib_mean_best_rank",
                        "heldout_top1",
                        "heldout_mean_mass",
                        "heldout_mean_best_rank",
                    )
                },
                "calib_top_matches": best["calib_top_matches"],
                "heldout_top_matches": best["heldout_top_matches"],
                "calib_expected_rank_audit": best["calib_expected_rank_audit"],
                "heldout_expected_rank_audit": best["heldout_expected_rank_audit"],
                "heldout_iia_by_variable": iia_rows,
                "heldout_model_accuracy": iia_accuracy,
                "heldout_kept_pairs": kept_heldout,
            }
        )

    _write_grid_csv(all_grid_rows, args.out_dir / "calibration_grid.csv")
    payload = {
        "model_info": model_info,
        "top_model_ids": TOP_MODEL_IDS,
        "splits": {
            "calibration_pair_ids": [left.pair_id for left, _ in splits["calibration"]],
            "heldout_pair_ids": [left.pair_id for left, _ in splits["heldout"]],
            "calibration_templates": sorted({left.template_id for left, _ in splits["calibration"]}),
            "heldout_templates": sorted({left.template_id for left, _ in splits["heldout"]}),
        },
        "grid": {
            "cost_modes": COST_MODES,
            "epsilons": EPSILONS,
            "betas": BETAS,
            "stage_penalties": STAGE_PENALTIES,
            "num_configs_per_model": len(COST_MODES) * len(EPSILONS) * len(BETAS) * len(STAGE_PENALTIES),
        },
        "results": results,
    }
    (args.out_dir / "calibrated_top_models.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Calibrated Top Causal Models", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- calibration templates: `{', '.join(payload['splits']['calibration_templates'])}`")
    lines.append(f"- heldout templates: `{', '.join(payload['splits']['heldout_templates'])}`")
    lines.append(f"- configs per model: `{payload['grid']['num_configs_per_model']}`")
    lines.extend(["", "## Summary", ""])
    for row in results:
        best = row["best_config"]
        acc = row["heldout_model_accuracy"]
        cov = row["canonical_coverage"]
        lines.append(
            f"- `{row['model_id']}` ({row['variable_count']} vars, {row['native_node_count']} nodes): "
            f"calib top1 `{best['calib_top1']:.3f}`, heldout top1 `{best['heldout_top1']:.3f}`, "
            f"heldout internal IIA `{acc['internal_strict_iia_accuracy']:.3f}`, "
            f"heldout strict IIA `{acc['strict_iia_accuracy']:.3f}`, coverage `{cov['canonical_coverage']:.3f}`"
        )
        lines.append(
            f"  - best config: cost `{best['cost_mode']}`, epsilon `{best['epsilon']}`, "
            f"beta `{best['beta_neural']}`, stage penalty `{best['stage_penalty']}`"
        )
    lines.extend(["", "## Heldout IIA By Variable", ""])
    for row in results:
        lines.append(f"### {row['model_id']}")
        for var in row["heldout_iia_by_variable"]:
            lines.append(
                f"- `{var['variable_id']}` {list(var['node_ids'])}: same `{var['same_preserve_accuracy']:.3f}`, "
                f"diff flip `{var['different_flip_accuracy']:.3f}`, diff move `{var['different_move_accuracy']:.3f}`, "
                f"strict `{var['strict_iia_accuracy']:.3f}`"
            )
        lines.append("")
    lines.extend(["", "## Heldout PLOT Top Matches", ""])
    for row in results:
        lines.append(f"### {row['model_id']}")
        for var_id, matches in row["heldout_top_matches"].items():
            top = matches[0]
            lines.append(f"- `{var_id}` -> `{top[0]}` ({top[1]:.3f})")
        lines.append("")
    (args.out_dir / "calibrated_top_models.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote calibrated top-model report to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
