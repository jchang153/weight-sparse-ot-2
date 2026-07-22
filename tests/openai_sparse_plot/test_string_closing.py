from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.string_closing import (
    ABSTRACT_VARIABLES,
    abstract_effect_signature,
    build_balanced_dataset,
    build_matched_pair,
    split_dataset,
    string_closing_state,
)


class TestStringClosing(unittest.TestCase):
    def test_matched_pair_differs_only_in_quote_type(self) -> None:
        single, double = build_matched_pair(
            template_id="t",
            template="print({quote}{content}",
            content="alpha",
        )
        self.assertEqual(single.pair_id, double.pair_id)
        self.assertEqual(single.content, double.content)
        self.assertEqual(single.template_id, double.template_id)
        self.assertEqual(single.opening_quote_type, "single")
        self.assertEqual(double.opening_quote_type, "double")
        self.assertEqual(single.target_token, "'")
        self.assertEqual(double.target_token, '"')

    def test_scm_propagates_stored_quote_type(self) -> None:
        single, _ = build_matched_pair(template_id="t", template="x = ({quote}{content}", content="alpha")
        factual = string_closing_state(single)
        intervened = string_closing_state(single, {"StoredQuoteType": 1})
        self.assertEqual(factual.output, -1)
        self.assertEqual(intervened.copied_quote_type, 1)
        self.assertEqual(intervened.output, 1)

    def test_abstract_effect_signature_has_variable_dimension(self) -> None:
        single, _ = build_matched_pair(template_id="t", template="x = ({quote}{content}", content="alpha")
        sig = abstract_effect_signature(single, "CopiedQuoteTypeAtFinalPosition", 1)
        self.assertEqual(len(sig), len(ABSTRACT_VARIABLES))
        self.assertGreater(sig[-1], 0)

    def test_split_dataset_preserves_pairs(self) -> None:
        examples = build_balanced_dataset(seed=1)
        splits = split_dataset(examples, seed=2)
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                seen.setdefault(row.pair_id, split)
                self.assertEqual(seen[row.pair_id], split)


if __name__ == "__main__":
    unittest.main()
