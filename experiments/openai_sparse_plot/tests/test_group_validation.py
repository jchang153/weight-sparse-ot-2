from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.run_group_validation import DEFAULT_GROUPS, _positions_for_group


class TestGroupValidation(unittest.TestCase):
    def test_default_groups_include_full_quote_type_path(self) -> None:
        groups = {group.group_id: group for group in DEFAULT_GROUPS}
        self.assertIn("opening_quote_detectors", groups)
        self.assertEqual(groups["opening_quote_detectors"].node_ids, ("0.mlp.post_act:863", "0.mlp.post_act:2790"))
        self.assertIn("full_quote_type_path", groups)
        self.assertIn("0.mlp.resid_delta:460", groups["full_quote_type_path"].node_ids)
        self.assertIn("final_resid:83", groups["full_quote_type_path"].node_ids)

    def test_positions_for_group_uses_final_for_output_sites(self) -> None:
        groups = {group.group_id: group for group in DEFAULT_GROUPS}
        positions = _positions_for_group(
            groups["output_preference"],
            {"opening_quote_position": 3, "final_position": 9, "quote_type": "double"},
        )
        self.assertEqual(positions["10.attn.resid_delta:83"], [9])
        self.assertEqual(positions["final_resid:83"], [9])


if __name__ == "__main__":
    unittest.main()
