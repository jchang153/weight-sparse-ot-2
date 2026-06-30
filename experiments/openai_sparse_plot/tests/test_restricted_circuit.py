from __future__ import annotations

import unittest

import torch

from experiments.openai_sparse_plot.restricted_circuit import (
    RetainedActivationMask,
    make_retain_channel_intervention,
    retained_masks_from_node_ids,
    retained_masks_from_viz_data,
    summarize_margin_records,
)


class TestRestrictedCircuit(unittest.TestCase):
    def test_retained_masks_from_viz_data_skips_bias_and_prune_config(self) -> None:
        masks = retained_masks_from_viz_data({"circuit_data": {"x": torch.tensor([1, 2]), "y": ["bias", 3], "prune_config": {}}})
        by_hook = {mask.hook_key: mask.retained_channels for mask in masks}
        self.assertEqual(by_hook["x"], (1, 2))
        self.assertEqual(by_hook["y"], (3,))
        self.assertNotIn("prune_config", by_hook)

    def test_retained_masks_from_node_ids_groups_channels(self) -> None:
        masks = retained_masks_from_node_ids(("x:1", "x:2", "y:3"))
        by_hook = {mask.hook_key: mask.retained_channels for mask in masks}
        self.assertEqual(by_hook["x"], (1, 2))
        self.assertEqual(by_hook["y"], (3,))

    def test_make_retain_channel_intervention_zeros_unretained(self) -> None:
        mask = RetainedActivationMask(hook_key="x", retained_channels=(1, 3))
        tensor = torch.arange(8.0).reshape(1, 2, 4)
        out = make_retain_channel_intervention(mask)(tensor)
        self.assertTrue(torch.equal(out[..., 0], torch.zeros_like(out[..., 0])))
        self.assertTrue(torch.equal(out[..., 2], torch.zeros_like(out[..., 2])))
        self.assertTrue(torch.equal(out[..., 1], tensor[..., 1]))
        self.assertTrue(torch.equal(out[..., 3], tensor[..., 3]))

    def test_summarize_margin_records(self) -> None:
        summary = summarize_margin_records(
            [
                {
                    "restricted_matches_clean": True,
                    "restricted_matches_expected": True,
                    "restricted_margin": 1.0,
                    "clean_margin": 0.5,
                },
                {
                    "restricted_matches_clean": False,
                    "restricted_matches_expected": True,
                    "restricted_margin": -1.0,
                    "clean_margin": 1.0,
                },
            ]
        )
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["restricted_matches_clean_rate"], 0.5)
        self.assertEqual(summary["restricted_matches_expected_rate"], 1.0)
        self.assertEqual(summary["mean_abs_margin_delta"], 1.25)


if __name__ == "__main__":
    unittest.main()
