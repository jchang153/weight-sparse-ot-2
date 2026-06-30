from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.openai_sparse_plot.artifacts import (
    candidate_viz_paths,
    graph_from_viz_data,
    write_graph_tables,
    write_inventory_report,
)
from experiments.openai_sparse_plot.schema import SparseCircuitEdge, SparseCircuitGraph, SparseCircuitNode


class TestArtifacts(unittest.TestCase):
    def test_graph_json_roundtrip(self) -> None:
        graph = SparseCircuitGraph(
            model="m",
            task="t",
            nodes=(
                SparseCircuitNode(node_id="a", layer=0, module="mlp", node_kind="neuron", index=1),
                SparseCircuitNode(node_id="b", layer=1, module="attn", node_kind="value", index=2),
            ),
            edges=(SparseCircuitEdge(src="a", dst="b", weight=0.5),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            graph.write_json(path)
            loaded = SparseCircuitGraph.read_json(path)
        self.assertEqual(loaded.nodes[0].node_id, "a")
        self.assertEqual(loaded.edges[0].dst, "b")

    def test_fake_viz_export_writes_expected_files(self) -> None:
        viz = {
            "circuit_data": {
                "0.mlp.act_in": torch.tensor([10]),
                "0.mlp.post_act": torch.tensor([20]),
            },
            "importances": {
                "ch_interv_losses": {
                    "0.mlp.act_in": torch.tensor([1.0]),
                    "0.mlp.post_act": torch.tensor([2.0]),
                },
                "pair_data": [
                    (
                        torch.tensor([[0.25]]),
                        [10],
                        [20],
                        ("0.mlp.act_in", "0.mlp.post_act"),
                    )
                ],
            }
        }
        graph = graph_from_viz_data(viz, model="m", task="t")
        with tempfile.TemporaryDirectory() as tmp:
            write_graph_tables(graph, out_dir=tmp)
            write_inventory_report(out_dir=tmp, status={"import_ok": False}, graph=graph)
            files = {p.name for p in Path(tmp).iterdir()}
            inventory = json.loads((Path(tmp) / "inventory.json").read_text(encoding="utf-8"))
        self.assertIn("string_closing_circuit.json", files)
        self.assertIn("string_closing_circuit_nodes.csv", files)
        self.assertIn("string_closing_circuit_edges.csv", files)
        self.assertEqual(inventory["graph"]["model"], "m")
        self.assertEqual(len(inventory["graph"]["nodes"]), 2)
        self.assertEqual(len(inventory["graph"]["edges"]), 1)
        self.assertEqual(inventory["graph"]["edges"][0]["src"], "0.mlp.act_in:10")

    def test_candidate_paths_match_openai_viz_layout(self) -> None:
        paths = candidate_viz_paths(
            model="csp_yolo1",
            task="single_double_quote",
            sweeps=("prune_v4",),
            ks=("k_optim", 12),
            base_dir="BASE",
        )
        self.assertIn("BASE/viz/csp_yolo1/single_double_quote/prune_v4/k_optim/viz_data.pt", paths)
        self.assertIn("BASE/viz/csp_yolo1/single_double_quote/prune_v4/12/viz_data.pt", paths)
        self.assertFalse(any("/viz/csp/" in path for path in paths))


if __name__ == "__main__":
    unittest.main()
