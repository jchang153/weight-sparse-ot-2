from __future__ import annotations

import unittest

import torch

from experiments.openai_sparse_plot.activation import (
    ChannelSite,
    extract_site_values,
    make_channel_patch,
    make_multi_channel_patch,
    make_weighted_multi_channel_patch,
)


class TestActivationUtilities(unittest.TestCase):
    def test_parse_channel_site(self) -> None:
        site = ChannelSite.from_node_id("10.attn.v:663", label="value")
        self.assertEqual(site.hook_key, "10.attn.v")
        self.assertEqual(site.channel, 663)
        self.assertEqual(site.label, "value")

    def test_parse_rejects_non_channel_node(self) -> None:
        with self.assertRaises(ValueError):
            ChannelSite.from_node_id("not-a-channel")

    def test_extract_site_values(self) -> None:
        site = ChannelSite.from_node_id("0.mlp.resid_delta:2")
        cache = {"0.mlp.resid_delta": torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])}
        values = extract_site_values(cache, [site], positions=[0, 1])
        self.assertEqual(values[site.site_id], [2.0, 5.0])

    def test_make_channel_patch_only_changes_requested_positions_and_channel(self) -> None:
        site = ChannelSite.from_node_id("x:1")
        source = {"x": torch.tensor([[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]])}
        target = torch.zeros((1, 2, 3))
        patch = make_channel_patch(site, source_cache=source, positions=[1], source_positions=[0])
        out = patch(target)
        self.assertEqual(float(out[0, 1, 1]), 11.0)
        self.assertEqual(float(out[0, 0, 1]), 0.0)
        self.assertEqual(float(out[0, 1, 0]), 0.0)
        self.assertEqual(float(out[0, 1, 2]), 0.0)

    def test_make_multi_channel_patch_groups_by_hook(self) -> None:
        sites = [ChannelSite.from_node_id("x:1"), ChannelSite.from_node_id("x:2")]
        source = {"x": torch.tensor([[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]])}
        target = torch.zeros((1, 2, 3))
        patches = make_multi_channel_patch(
            sites,
            source_cache=source,
            positions_by_site={"x:1": [1], "x:2": [0]},
            source_positions_by_site={"x:1": [0], "x:2": [1]},
        )
        out = patches["x"](target)
        self.assertEqual(float(out[0, 1, 1]), 11.0)
        self.assertEqual(float(out[0, 0, 2]), 22.0)
        self.assertEqual(float(out[0, 0, 1]), 0.0)
        self.assertEqual(float(out[0, 1, 2]), 0.0)

    def test_make_weighted_multi_channel_patch_applies_fractional_source_delta(self) -> None:
        sites = [ChannelSite.from_node_id("x:1"), ChannelSite.from_node_id("x:2")]
        source = {"x": torch.tensor([[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]])}
        target = torch.ones((1, 2, 3))
        patches = make_weighted_multi_channel_patch(
            sites,
            source_cache=source,
            positions_by_site={"x:1": [1], "x:2": [0]},
            source_positions_by_site={"x:1": [0], "x:2": [1]},
            weights_by_site={"x:1": 0.25, "x:2": 0.5},
            strength=2.0,
        )
        out = patches["x"](target)
        self.assertEqual(float(out[0, 1, 1]), 6.0)
        self.assertEqual(float(out[0, 0, 2]), 22.0)
        self.assertEqual(float(out[0, 0, 1]), 1.0)
        self.assertEqual(float(out[0, 1, 2]), 1.0)


if __name__ == "__main__":
    unittest.main()
