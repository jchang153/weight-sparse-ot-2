from __future__ import annotations

import unittest

import torch

from experiments.openai_sparse_plot.run_faithfulness_audit import audit_viz_data, compare_task_samples


class TestFaithfulnessAudit(unittest.TestCase):
    def test_audit_counts_retained_scalars_and_losses(self) -> None:
        viz_data = {
            "num_total_nodes": 10,
            "all_loss": [(0, 1.0), (1, 0.5)],
            "prune_config": {"epochs": 1},
            "circuit_data": {"x": torch.tensor([1, 2]), "y": [3], "prune_config": {}},
            "samples": {"x": torch.zeros(1)},
            "importances": {
                "loss": 0.1,
                "interv_loss": 0.2,
                "task_samples": [1, 2, 3],
                "task_samples_circuit": [1],
            },
        }
        audit = audit_viz_data(viz_data, viz_path="dummy")
        self.assertEqual(audit["retained_scalar_count"], 3)
        self.assertEqual(audit["reported_original_or_task_loss"], 0.1)
        self.assertEqual(audit["reported_pruned_intervention_loss"], 0.2)
        self.assertFalse(audit["executable_pruned_model_available"])

    def test_compare_task_samples_reports_tensor_diffs(self) -> None:
        task_samples = (
            torch.tensor([[1, 2]]),
            {"x": {3: torch.tensor([[1.0, 2.0]])}},
        )
        task_samples_circuit = (
            torch.tensor([[1, 2]]),
            {"x": {3: torch.tensor([[1.0, 3.0]])}},
        )
        comparison = compare_task_samples(task_samples, task_samples_circuit)
        self.assertTrue(comparison["available"])
        self.assertTrue(comparison["token_tensors_equal"])
        self.assertEqual(comparison["compared_retained_tensors"], 1)
        self.assertEqual(comparison["max_abs_diff"], 1.0)


if __name__ == "__main__":
    unittest.main()
