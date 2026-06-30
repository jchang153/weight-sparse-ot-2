from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_viz_data
from .run_bracket_counting_abstraction import (
    DEFAULT_VIZ_PATH,
    _build_resampling_specs as build_bracket_specs,
    _clean_summary as bracket_clean_summary,
    _collect_runs as collect_bracket_runs,
    _load_released_examples,
    _record_sites as bracket_record_sites,
    _summarize_records as summarize_bracket_records,
)
from .run_raw_delta_plot_abstraction import (
    _behavior_summary,
    _calibrate_soft_handle_behavior,
    _calibrate_soft_handle_raw,
    _parse_float_grid,
    _parse_int_grid,
    _raw_cost,
    _raw_vectors_from_records,
    _selector_payload_from_raw,
    _write_markdown,
)
from .run_singleton_soft_handle_abstraction import (
    _bracket_singleton_handles,
    _run_bracket_soft_records,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build calibrated bracket raw-delta PLOT from checkpointed scan records.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument(
        "--bracket-node-csv",
        type=Path,
        default=Path("eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"),
    )
    parser.add_argument("--scan-json", type=Path, required=True)
    parser.add_argument("--records-jsonl", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-records-per-relation", type=int, default=6)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _count_by_handle(records: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        handle_id = str(row["handle_id"])
        counts[handle_id] = counts.get(handle_id, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)

    scan = json.loads(args.scan_json.read_text(encoding="utf-8"))
    records_path = args.records_jsonl
    if records_path is None:
        records_path = Path(scan.get("records_jsonl", args.scan_json.parent / "singleton_records.jsonl"))
    records = _read_records(records_path)

    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.viz_path)
    examples = _load_released_examples(viz_data, enc)
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
    expected_per_handle = len(calibration_specs)

    handles = _bracket_singleton_handles(args.bracket_node_csv)
    counts = _count_by_handle(records)
    incomplete = [handle.handle_id for handle in handles if counts.get(handle.handle_id, 0) < expected_per_handle]
    if incomplete:
        raise ValueError(
            f"scan records are incomplete for {len(incomplete)} handles; first missing/incomplete: {incomplete[:8]}"
        )
    records = [row for row in records if counts.get(str(row["handle_id"]), 0) >= expected_per_handle]

    abstract, neural, feature_names = _raw_vectors_from_records(records, task="bracket")
    selector = _selector_payload_from_raw(
        abstract=abstract,
        neural_by_id=neural,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )

    max_k = max(k_grid)
    top_site_ids = {
        str(row["site_id"])
        for payload in selector["selectors"].values()
        for row in payload["ranked_sites"][:max_k]
    }
    site_by_id = {handle.handle_id: handle.sites()[0] for handle in handles}
    missing_top = sorted(top_site_ids - set(site_by_id))
    if missing_top:
        raise ValueError(f"top selector sites missing from CSV handles: {missing_top}")

    device = "cuda" if args.cuda else "cpu"
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=True,
        grad_checkpointing=False,
    )
    runs = collect_bracket_runs(
        model,
        examples,
        sites=tuple(site_by_id[site_id] for site_id in sorted(top_site_ids)),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = bracket_clean_summary(examples, runs)
    lookup = {ex.example_id: ex for ex in examples}

    calibrated_raw_cost = {}
    calibrated_behavior = {}
    heldout = {}
    for selector_name, selector_payload in selector["selectors"].items():
        print(f"bracket full-scan calibrating {selector_name}", flush=True)
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

    bracket_payload = {
        "model_info": model_info,
        "viz_path": args.viz_path,
        "scan_json": str(args.scan_json),
        "records_jsonl": str(records_path),
        "clean": clean,
        "candidate_site_count": len(handles),
        "candidate_sites": [handle.__dict__ for handle in handles],
        "raw_feature_names": feature_names,
        "selector": selector,
        "calibrated_soft_handles_raw_cost": calibrated_raw_cost,
        "calibrated_soft_handles_behavior": calibrated_behavior,
        "heldout_soft_summary": heldout,
    }
    payload = {
        "raw_signature_definition": "phi(y_swap) - phi(y_base)",
        "max_records_per_relation": int(args.max_records_per_relation),
        "k_grid": list(k_grid),
        "strength_grid": list(strength_grid),
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "hard_replay": {},
        "soft_runs": {"bracket": bracket_payload},
    }
    json_path = args.out_dir / "raw_delta_plot_abstraction.json"
    md_path = args.out_dir / "raw_delta_plot_abstraction.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "top_sites": sorted(top_site_ids)}, indent=2))


if __name__ == "__main__":
    main()
