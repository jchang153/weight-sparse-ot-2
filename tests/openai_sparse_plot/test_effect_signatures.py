from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.effect_signatures import (
    SIGNATURE_FEATURE_BASES,
    abstract_signature_for_variable,
    build_effect_prompt_pairs,
    interpreted_channel_sites,
    restricted_binary_kl_from_margins,
)
from experiments.openai_sparse_plot.string_closing import ABSTRACT_VARIABLES


class TestEffectSignatures(unittest.TestCase):
    def test_interpreted_sites_parse(self) -> None:
        sites = interpreted_channel_sites()
        ids = {site.site_id for site in sites}
        self.assertIn("0.mlp.resid_delta:460", ids)
        self.assertIn("10.attn.v:663", ids)
        self.assertIn("final_resid:83", ids)

    def test_abstract_signature_dimensions(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=2)
        row = abstract_signature_for_variable("StoredQuoteType", pairs)
        self.assertEqual(len(row), 2 * len(SIGNATURE_FEATURE_BASES))

    def test_source_aligned_quote_type_does_not_cancel(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=2)
        row = abstract_signature_for_variable("StoredQuoteType", pairs)
        feature_names = tuple(f"resample.{name}" for name in SIGNATURE_FEATURE_BASES) + tuple(
            f"zero.{name}" for name in SIGNATURE_FEATURE_BASES
        )
        idx = feature_names.index("resample.quote_type_channel_460")
        self.assertGreater(row[idx], 0.0)

    def test_all_abstract_variables_have_rows(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=1)
        rows = [abstract_signature_for_variable(var, pairs) for var in ABSTRACT_VARIABLES]
        self.assertEqual(len(rows), len(ABSTRACT_VARIABLES))
        self.assertTrue(all(len(row) == 2 * len(SIGNATURE_FEATURE_BASES) for row in rows))

    def test_opening_quote_type_and_stored_quote_type_are_not_identical(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=1)
        open_row = abstract_signature_for_variable("OpeningQuoteType", pairs)
        stored_row = abstract_signature_for_variable("StoredQuoteType", pairs)
        self.assertNotEqual(open_row, stored_row)
        feature_names = tuple(f"resample.{name}" for name in SIGNATURE_FEATURE_BASES) + tuple(
            f"zero.{name}" for name in SIGNATURE_FEATURE_BASES
        )
        detector_idx = feature_names.index("resample.detector_post_act_quote_balance")
        self.assertGreater(open_row[detector_idx], 0.0)
        self.assertEqual(stored_row[detector_idx], 0.0)

    def test_restricted_binary_kl_is_zero_for_equal_margins(self) -> None:
        self.assertAlmostEqual(restricted_binary_kl_from_margins(2.0, 2.0), 0.0)

    def test_output_signature_has_kl_effect_feature(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=1)
        row = abstract_signature_for_variable("Output", pairs)
        feature_names = tuple(f"resample.{name}" for name in SIGNATURE_FEATURE_BASES) + tuple(
            f"zero.{name}" for name in SIGNATURE_FEATURE_BASES
        )
        self.assertIn("resample.binary_quote_restricted_kl", feature_names)
        self.assertIn("zero.binary_quote_restricted_kl", feature_names)
        self.assertGreater(row[feature_names.index("resample.binary_quote_restricted_kl")], 0.0)
        self.assertGreater(row[feature_names.index("zero.binary_quote_restricted_kl")], 0.0)


if __name__ == "__main__":
    unittest.main()
