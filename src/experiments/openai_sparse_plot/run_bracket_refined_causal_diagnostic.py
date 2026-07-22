from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .activation import ChannelSite
from .artifacts import load_viz_data
from .run_bracket_counting_abstraction import (
    DEFAULT_HANDLES,
    DEFAULT_VIZ_PATH,
    CandidateHandle,
    _build_resampling_specs,
    _clean_summary,
    _collect_runs,
    _metric,
    _record_sites,
    _run_records_for_handle,
    _summarize_records,
)
from .run_raw_delta_plot_abstraction import _behavior_score, _behavior_summary
from .run_singleton_soft_handle_abstraction import _run_bracket_soft_records
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .run_bracket_counting_abstraction import _load_released_examples


RAW_PLOT_SITES: tuple[tuple[str, float], ...] = (
    ("final_resid:1079", 0.08998174965381622),
    ("7.mlp.post_act:4133", 0.0871257558465004),
    ("7.mlp.resid_delta:2041", 0.0870715081691742),
)

HARD_HANDLE_IDS = (
    "depth_path_1249",
    "late_depth_signal_core",
    "late_depth_state_7_mlp_input",
    "late_depth_readout_7_mlp_post",
    "layer1_control_1643",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused bracket diagnostic: internal depth variable vs late readout handle."
    )
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/bracket/bracket_refined_causal_diagnostic"),
    )
    parser.add_argument("--max-records-per-relation", type=int, default=6)
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0,8.0")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    return parser.parse_args()


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("empty strength grid")
    return values


def _normalize_weights(items: Sequence[tuple[str, float]]) -> dict[str, float]:
    total = sum(float(weight) for _, weight in items)
    if total <= 0:
        raise ValueError("weights must have positive total")
    return {site_id: float(weight) / total for site_id, weight in items}


def _soft_variants() -> dict[str, dict[str, Any]]:
    no_final = tuple(item for item in RAW_PLOT_SITES if not item[0].startswith("final_resid:"))
    final_only = tuple(item for item in RAW_PLOT_SITES if item[0] == "final_resid:1079")
    return {
        "raw_plot_full_top3": {
            "label": "raw cosine-UOT behavior handle: full top-3",
            "weights_by_site": _normalize_weights(RAW_PLOT_SITES),
        },
        "raw_plot_no_final_top2": {
            "label": "same raw handle with final_resid removed",
            "weights_by_site": _normalize_weights(no_final),
        },
        "raw_plot_final_only": {
            "label": "final_resid component alone",
            "weights_by_site": _normalize_weights(final_only),
        },
    }


def _sites_from_weights(variants: Mapping[str, Mapping[str, Any]]) -> tuple[ChannelSite, ...]:
    by_id: dict[str, ChannelSite] = {}
    for variant in variants.values():
        for site_id in variant["weights_by_site"]:
            by_id.setdefault(site_id, ChannelSite.from_node_id(site_id))
    return tuple(by_id.values())


def _behavior_for(summary: Mapping[str, Mapping[str, Any]], handle_id: str) -> dict[str, float]:
    return _behavior_summary("bracket", summary[handle_id])


def _run_hard_handles(
    *,
    model: Any,
    handles: Sequence[CandidateHandle],
    specs: Sequence[Any],
    examples_by_id: Mapping[str, Any],
    runs: Mapping[str, Any],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    records: list[dict[str, Any]] = []
    for handle in handles:
        records.extend(
            _run_records_for_handle(
                model=model,
                handle=handle,
                specs=specs,
                examples=examples_by_id,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
        )
    return records, _summarize_records(records)


def _run_soft_grid(
    *,
    model: Any,
    variants: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
    examples_by_id: Mapping[str, Any],
    runs: Mapping[str, Any],
    single_close_token_id: int,
    double_close_token_id: int,
    strengths: Sequence[float],
) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    best: dict[str, dict[str, Any]] = {}
    for handle_id, variant in variants.items():
        rows[handle_id] = []
        weights_by_site = dict(variant["weights_by_site"])
        sites = tuple(ChannelSite.from_node_id(site_id) for site_id in weights_by_site)
        for strength in strengths:
            records = _run_bracket_soft_records(
                model=model,
                handle_id=f"{handle_id}_lambda{strength:g}",
                sites=sites,
                weights_by_site=weights_by_site,
                strength=float(strength),
                specs=specs,
                examples=examples_by_id,
                runs=runs,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            summary = _summarize_records(records)[f"{handle_id}_lambda{strength:g}"]
            behavior = _behavior_for({handle_id: summary}, handle_id)
            row = {
                "handle_id": handle_id,
                "label": variant["label"],
                "strength": float(strength),
                "weights_by_site": weights_by_site,
                "records": records,
                "summary": summary,
                "behavior": behavior,
                "behavior_score": _behavior_score(behavior),
            }
            rows[handle_id].append(row)
        best[handle_id] = sorted(
            rows[handle_id],
            key=lambda row: (
                -float(row["behavior_score"]),
                -float(row["behavior"]["flip"]),
                float(row["behavior"]["wrong_preserve"]),
                float(row["strength"]),
            ),
        )[0]
    return {"rows": rows, "best": best}


def _run_soft_fixed(
    *,
    model: Any,
    variants: Mapping[str, Mapping[str, Any]],
    best_by_handle: Mapping[str, Mapping[str, Any]],
    specs: Sequence[Any],
    examples_by_id: Mapping[str, Any],
    runs: Mapping[str, Any],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for handle_id, variant in variants.items():
        strength = float(best_by_handle[handle_id]["strength"])
        weights_by_site = dict(variant["weights_by_site"])
        sites = tuple(ChannelSite.from_node_id(site_id) for site_id in weights_by_site)
        records = _run_bracket_soft_records(
            model=model,
            handle_id=handle_id,
            sites=sites,
            weights_by_site=weights_by_site,
            strength=strength,
            specs=specs,
            examples=examples_by_id,
            runs=runs,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        summary = _summarize_records(records)[handle_id]
        behavior = _behavior_for({handle_id: summary}, handle_id)
        out[handle_id] = {
            "handle_id": handle_id,
            "label": variant["label"],
            "strength": strength,
            "weights_by_site": weights_by_site,
            "records": records,
            "summary": summary,
            "behavior": behavior,
            "behavior_score": _behavior_score(behavior),
        }
    return out


def _md_metric(row: Mapping[str, Any], key: str) -> float:
    return _metric(row["summary"], key)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Bracket Refined Causal Diagnostic",
        "",
        "Question: does raw-delta PLOT identify an internal depth variable, or mostly a late readout/output-margin variable?",
        "",
        "Refined model under test:",
        "",
        "```text",
        "X -> D_mid -> R_late -> Y",
        "```",
        "",
        "- `D_mid`: internal parsed bracket-depth state.",
        "- `R_late`: late residual/readout expression of the depth decision.",
        f"- released samples: `{payload['clean']['n']}`",
        f"- clean accuracy: `{payload['clean']['accuracy']:.3f}`",
        f"- max records per relation: `{payload['max_records_per_relation']}`",
        "",
        "## Hard Internal-Depth Handles",
        "",
        "| handle | calibration same | calibration flip | calibration wrong-preserve | heldout same | heldout flip | heldout wrong-preserve | heldout shift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for handle_id, row in payload["hard"]["heldout"].items():
        cal = payload["hard"]["calibration"][handle_id]
        held_behavior = _behavior_for({handle_id: row}, handle_id)
        cal_behavior = _behavior_for({handle_id: cal}, handle_id)
        lines.append(
            f"| `{handle_id}` | {cal_behavior['same']:.3f} | {cal_behavior['flip']:.3f} | "
            f"{cal_behavior['wrong_preserve']:.3f} | {held_behavior['same']:.3f} | "
            f"{held_behavior['flip']:.3f} | {held_behavior['wrong_preserve']:.3f} | "
            f"{held_behavior['shift']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Raw PLOT Handle Ablations",
            "",
            "| variant | selected strength | calibration score | calibration same | calibration flip | calibration wrong-preserve | heldout score | heldout same | heldout flip | heldout wrong-preserve | heldout shift |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for handle_id, held in payload["soft"]["heldout"].items():
        cal = payload["soft"]["calibration_best"][handle_id]
        lines.append(
            f"| `{handle_id}` | {held['strength']:.3f} | {cal['behavior_score']:.3f} | "
            f"{cal['behavior']['same']:.3f} | {cal['behavior']['flip']:.3f} | "
            f"{cal['behavior']['wrong_preserve']:.3f} | {held['behavior_score']:.3f} | "
            f"{held['behavior']['same']:.3f} | {held['behavior']['flip']:.3f} | "
            f"{held['behavior']['wrong_preserve']:.3f} | {held['behavior']['shift']:.3f} |"
        )

    lines.extend(["", "## Soft Variant Weights", ""])
    for handle_id, row in payload["soft"]["heldout"].items():
        weights = ", ".join(f"`{site}`={weight:.3f}" for site, weight in row["weights_by_site"].items())
        lines.append(f"- `{handle_id}`: {weights}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    strengths = _parse_float_grid(args.strength_grid)
    device = "cuda" if args.cuda else "cpu"
    flash = not bool(args.no_flash)

    print("loading tokenizer/artifact/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.viz_path)
    examples = _load_released_examples(viz_data, enc)
    examples_by_id = {ex.example_id: ex for ex in examples}
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    hard_handles = tuple(handle for handle in DEFAULT_HANDLES if handle.handle_id in HARD_HANDLE_IDS)
    if len(hard_handles) != len(HARD_HANDLE_IDS):
        found = {handle.handle_id for handle in hard_handles}
        missing = sorted(set(HARD_HANDLE_IDS) - found)
        raise ValueError(f"missing hard handles: {missing}")
    soft_variants = _soft_variants()
    record_sites = tuple({site.site_id: site for site in (*_record_sites(hard_handles), *_sites_from_weights(soft_variants))}.values())

    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=flash,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    runs = _collect_runs(
        model,
        examples,
        sites=record_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = _clean_summary(examples, runs)
    split_specs = {
        split: _build_resampling_specs(
            examples,
            split=split,
            max_records_per_relation=args.max_records_per_relation,
        )
        for split in ("calibration", "heldout")
    }

    print("running hard handles", flush=True)
    hard_cal_records, hard_cal = _run_hard_handles(
        model=model,
        handles=hard_handles,
        specs=split_specs["calibration"],
        examples_by_id=examples_by_id,
        runs=runs,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    hard_held_records, hard_held = _run_hard_handles(
        model=model,
        handles=hard_handles,
        specs=split_specs["heldout"],
        examples_by_id=examples_by_id,
        runs=runs,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )

    print("running soft calibration grid", flush=True)
    soft_cal = _run_soft_grid(
        model=model,
        variants=soft_variants,
        specs=split_specs["calibration"],
        examples_by_id=examples_by_id,
        runs=runs,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        strengths=strengths,
    )
    print("running soft heldout at selected strengths", flush=True)
    soft_held = _run_soft_fixed(
        model=model,
        variants=soft_variants,
        best_by_handle=soft_cal["best"],
        specs=split_specs["heldout"],
        examples_by_id=examples_by_id,
        runs=runs,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )

    payload = {
        "model_info": model_info,
        "model": args.model,
        "viz_path": args.viz_path,
        "max_records_per_relation": args.max_records_per_relation,
        "strength_grid": strengths,
        "clean": clean,
        "hard": {
            "handles": [handle.__dict__ for handle in hard_handles],
            "calibration": hard_cal,
            "calibration_records": hard_cal_records,
            "heldout": hard_held,
            "heldout_records": hard_held_records,
        },
        "soft": {
            "variants": soft_variants,
            "calibration_rows": soft_cal["rows"],
            "calibration_best": soft_cal["best"],
            "heldout": soft_held,
        },
    }
    json_path = args.out_dir / "bracket_refined_causal_diagnostic.json"
    md_path = args.out_dir / "bracket_refined_causal_diagnostic.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    print(json.dumps({"out_dir": str(args.out_dir), "markdown": str(md_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()


