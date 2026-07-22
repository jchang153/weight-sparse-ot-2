from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import (
    DEFAULT_MODEL,
    DEFAULT_TASK,
    candidate_viz_paths,
    circuit_sparsity_status,
    graph_from_viz_data,
    load_viz_data,
    write_graph_tables,
    write_inventory_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory OpenAI sparse-circuit artifacts for PLOT.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quote/inventory"))
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--sweep", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--viz-path", default=None, help="Local path or OpenAI blob URL to viz_data.pt/pkl.")
    parser.add_argument(
        "--try-first-candidate",
        action="store_true",
        help="Attempt to download/load the first candidate viz path if --viz-path is omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = circuit_sparsity_status(args.circuit_home)
    base_dir = status.get("model_base_dir") or None
    candidates = candidate_viz_paths(model=args.model, task=args.task, base_dir=base_dir)
    graph = None
    notes = []

    viz_path = args.viz_path
    if viz_path is None and args.try_first_candidate:
        viz_path = candidates[0]
        notes.append(f"Trying first candidate viz path: {viz_path}")

    if viz_path:
        viz_data = load_viz_data(viz_path)
        graph = graph_from_viz_data(
            viz_data,
            model=args.model,
            task=args.task,
            sweep=args.sweep,
            k=args.k,
            source_artifact=str(viz_path),
        )
        write_graph_tables(graph, out_dir=args.out_dir)
    else:
        notes.append("Pass --viz-path or --try-first-candidate to load/export a circuit artifact.")

    write_inventory_report(
        out_dir=args.out_dir,
        status=status,
        graph=graph,
        candidates=candidates,
        notes=tuple(notes),
    )
    print(f"wrote inventory to {args.out_dir}")


if __name__ == "__main__":
    main()
