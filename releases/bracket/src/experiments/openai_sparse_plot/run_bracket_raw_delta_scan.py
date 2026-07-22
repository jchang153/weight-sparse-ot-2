from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .artifacts import load_viz_data
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .run_bracket_counting_abstraction import (
    DEFAULT_VIZ_PATH,
    _build_resampling_specs as build_bracket_specs,
    _clean_summary as bracket_clean_summary,
    _collect_runs as collect_bracket_runs,
    _load_released_examples,
    _record_sites as bracket_record_sites,
    _run_records_for_handle as run_bracket_records_for_handle,
)
from .run_raw_delta_plot_abstraction import (
    _bracket_singleton_handles,
    _raw_vectors_from_records,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed full-menu bracket raw-delta scan.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument(
        "--bracket-node-csv",
        type=Path,
        default=Path("eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/raw_delta_bracket_full_scan"))
    parser.add_argument("--max-records-per-relation", type=int, default=2)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _selector_payload(
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
    direct = cosine_similarity.clamp_min(0.0)
    if float(direct.sum()) <= 0.0:
        direct = torch.softmax(cosine_similarity, dim=0)
    else:
        direct = direct / direct.sum().clamp_min(1e-12)

    specs = {
        "raw_squared_uot": (squared_uot, squared_cost[0], None),
        "raw_cosine_uot": (cosine_uot, cosine_cost[0], cosine_similarity),
        "raw_cosine_similarity": (direct, cosine_cost[0], cosine_similarity),
    }
    selectors = {}
    for name, (weights, costs, sims) in specs.items():
        rows = []
        for idx, site_id in enumerate(site_ids):
            rows.append(
                {
                    "site_id": site_id,
                    "weight": float(weights[idx]),
                    "cost": float(costs[idx]),
                    "similarity": None if sims is None else float(sims[idx]),
                    "raw_signature": tuple(float(x) for x in neural_by_id[site_id]),
                }
            )
        selectors[name] = {
            "ranked_sites": sorted(rows, key=lambda row: (-float(row["weight"]), float(row["cost"]))),
        }
    return {
        "abstract_signature": tuple(float(x) for x in abstract),
        "selectors": selectors,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"

    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.viz_path)
    examples = _load_released_examples(viz_data, enc)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    handles = _bracket_singleton_handles(args.bracket_node_csv)

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
        sites=bracket_record_sites(handles),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = bracket_clean_summary(examples, runs)
    lookup = {ex.example_id: ex for ex in examples}
    specs = build_bracket_specs(
        examples,
        split="calibration",
        max_records_per_relation=args.max_records_per_relation,
    )

    records_path = args.out_dir / "singleton_records.jsonl"
    expected_records_per_handle = len(specs)
    existing_records: list[dict[str, Any]] = []
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
    record_counts: dict[str, int] = {}
    for row in existing_records:
        record_counts[str(row["handle_id"])] = record_counts.get(str(row["handle_id"]), 0) + 1
    completed_handles = {
        handle_id
        for handle_id, count in record_counts.items()
        if count >= expected_records_per_handle
    }
    all_records: list[dict[str, Any]] = [
        row for row in existing_records if str(row["handle_id"]) in completed_handles
    ]
    with records_path.open("w", encoding="utf-8") as f:
        for row in all_records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    for index, handle in enumerate(handles, start=1):
        if handle.handle_id in completed_handles:
            print(f"[{index}/{len(handles)}] bracket raw singleton {handle.handle_id} (cached)", flush=True)
            continue
        print(f"[{index}/{len(handles)}] bracket raw singleton {handle.handle_id}", flush=True)
        records = run_bracket_records_for_handle(
            model=model,
            handle=handle,
            specs=specs,
            examples=lookup,
            runs=runs,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        all_records.extend(records)
        with records_path.open("a", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    abstract, neural, feature_names = _raw_vectors_from_records(all_records, task="bracket")
    selector = _selector_payload(
        abstract=abstract,
        neural_by_id=neural,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    payload = {
        "model_info": model_info,
        "viz_path": args.viz_path,
        "clean": clean,
        "candidate_site_count": len(handles),
        "candidate_sites": [handle.__dict__ for handle in handles],
        "max_records_per_relation": int(args.max_records_per_relation),
        "feature_names": feature_names,
        "selector": selector,
        "records_jsonl": str(records_path),
    }
    (args.out_dir / "bracket_raw_delta_scan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Bracket Raw-Delta Full Singleton Scan",
        "",
        f"- candidate singleton sites: `{len(handles)}`",
        f"- clean accuracy: `{clean['accuracy']:.3f}`",
        f"- max records per relation: `{args.max_records_per_relation}`",
        "",
        "| method | rank | site | weight | cost | cosine sim |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for method, sel in selector["selectors"].items():
        for rank, row in enumerate(sel["ranked_sites"][:12], start=1):
            sim = row["similarity"]
            sim_text = "n/a" if sim is None else f"{sim:.3f}"
            lines.append(
                f"| `{method}` | {rank} | `{row['site_id']}` | {row['weight']:.3f} | "
                f"{row['cost']:.3f} | {sim_text} |"
            )
    (args.out_dir / "bracket_raw_delta_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    compact = {
        method: [row["site_id"] for row in sel["ranked_sites"][:8]]
        for method, sel in selector["selectors"].items()
    }
    print(json.dumps({"out_dir": str(args.out_dir), "top_sites": compact}, indent=2))


if __name__ == "__main__":
    main()
