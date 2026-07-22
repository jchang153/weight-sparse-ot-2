from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.openai_sparse_plot.interpreted_circuit import (
    CANONICAL_EDGE_ALLOWLIST,
    PAPER_BACKED_NODE_SPECS,
    build_interpreted_subcircuit,
    write_interpreted_subcircuit,
)
from experiments.openai_sparse_plot.schema import SparseCircuitEdge, SparseCircuitGraph, SparseCircuitNode


def _raw_test_graph() -> SparseCircuitGraph:
    nodes = tuple(
        SparseCircuitNode(
            node_id=spec.node_id,
            layer=None,
            module="test",
            node_kind="test",
            index=None,
            importance=1.0,
        )
        for spec in PAPER_BACKED_NODE_SPECS
    )
    edges = tuple(SparseCircuitEdge(src=src, dst=dst, weight=1.0) for src, dst in CANONICAL_EDGE_ALLOWLIST)
    return SparseCircuitGraph(model="m", task="single_double_quote", nodes=nodes, edges=edges)


class TestInterpretedCircuit(unittest.TestCase):
    def test_build_interpreted_subcircuit_preserves_specs(self) -> None:
        graph = _raw_test_graph()
        sub_graph, node_rows, edge_rows = build_interpreted_subcircuit(graph)
        self.assertEqual(len(sub_graph.nodes), len(PAPER_BACKED_NODE_SPECS))
        self.assertEqual(len(node_rows), len(PAPER_BACKED_NODE_SPECS))
        self.assertEqual(
            sum(1 for edge in sub_graph.edges if edge.edge_kind == "canonical_interpreted"),
            len(CANONICAL_EDGE_ALLOWLIST),
        )
        self.assertEqual(len(edge_rows), len(CANONICAL_EDGE_ALLOWLIST))

    def test_missing_paper_backed_node_fails(self) -> None:
        graph = _raw_test_graph()
        graph = SparseCircuitGraph(
            model=graph.model,
            task=graph.task,
            nodes=graph.nodes[:-1],
            edges=(),
        )
        with self.assertRaises(ValueError):
            build_interpreted_subcircuit(graph)

    def test_write_interpreted_subcircuit_outputs_files(self) -> None:
        graph = _raw_test_graph()
        with tempfile.TemporaryDirectory() as tmp:
            write_interpreted_subcircuit(graph, out_dir=tmp)
            names = {p.name for p in Path(tmp).iterdir()}
        self.assertIn("interpreted_string_closing_subcircuit.json", names)
        self.assertIn("interpreted_nodes.csv", names)
        self.assertIn("interpreted_edges.csv", names)
        self.assertIn("interpreted_subcircuit.md", names)


if __name__ == "__main__":
    unittest.main()
