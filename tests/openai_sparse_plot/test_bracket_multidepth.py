from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.bracket_multidepth import (
    build_active_tail,
    build_relation_specs,
    close_count_from_depth,
    depth_from_tail,
    generate_multidepth_examples,
    relation_counts,
)


class BracketMultiDepthTest(unittest.TestCase):
    def test_depth_and_saturated_readout(self) -> None:
        self.assertEqual(close_count_from_depth(1), 1)
        self.assertEqual(close_count_from_depth(2), 2)
        self.assertEqual(close_count_from_depth(4), 2)
        self.assertEqual(depth_from_tail(build_active_tail(3, "1, 2")), 3)

    def test_generation_balances_depths_and_context(self) -> None:
        examples = generate_multidepth_examples(None, depths=(1, 2, 3, 4), examples_per_depth=8)
        self.assertEqual(len(examples), 32)
        by_depth = {depth: [ex for ex in examples if ex.depth == depth] for depth in (1, 2, 3, 4)}
        self.assertTrue(all(len(rows) == 8 for rows in by_depth.values()))
        self.assertTrue(any(ex.context_family == "surface_balanced" for ex in examples))

    def test_relation_builder_splits_d_from_r(self) -> None:
        examples = generate_multidepth_examples(None, depths=(1, 2, 3, 4), examples_per_depth=12)
        specs = build_relation_specs(examples, split="calibration", max_records_per_relation=6)
        counts = relation_counts(specs)
        self.assertGreater(counts["same_D"], 0)
        self.assertGreater(counts["different_D_same_R"], 0)
        self.assertGreater(counts["different_D_different_R"], 0)
        self.assertGreater(counts["same_surface_different_active_context"], 0)
        self.assertGreater(counts["wrong_numeric_content"], 0)
        self.assertGreater(counts["wrong_tail_length"], 0)


if __name__ == "__main__":
    unittest.main()
