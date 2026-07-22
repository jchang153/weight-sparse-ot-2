from __future__ import annotations

import argparse
from pathlib import Path

from .interpreted_circuit import write_interpreted_subcircuit
from .schema import SparseCircuitGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the interpreted string-closing subcircuit.")
    parser.add_argument(
        "--graph-json",
        type=Path,
        default=Path("results/quote/string_closing_prune_v2_64/string_closing_circuit.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/quote/string_closing_interpreted"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = SparseCircuitGraph.read_json(args.graph_json)
    sub_graph = write_interpreted_subcircuit(graph, out_dir=args.out_dir)
    canonical_count = sum(1 for edge in sub_graph.edges if edge.edge_kind == "canonical_interpreted")
    print(
        f"wrote interpreted subcircuit to {args.out_dir} "
        f"nodes={len(sub_graph.nodes)} canonical_edges={canonical_count} total_edges={len(sub_graph.edges)}"
    )


if __name__ == "__main__":
    main()
