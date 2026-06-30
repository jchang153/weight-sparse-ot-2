from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.effect_signatures import build_effect_prompt_pairs
from experiments.openai_sparse_plot.run_unmatched_quote_abstraction import (
    _causal_score,
    _handle_signature,
    _selector_payload,
    _split_pairs,
)


class TestUnmatchedQuoteAbstraction(unittest.TestCase):
    def test_split_pairs_prefers_template_heldout(self) -> None:
        pairs = build_effect_prompt_pairs(max_pairs=16)
        calibration, heldout = _split_pairs(pairs)
        self.assertEqual({pair[0].template_id for pair in calibration}, {"assign", "print"})
        self.assertEqual({pair[0].template_id for pair in heldout}, {"paren_assign", "handler_arg"})

    def test_handle_signature_rewards_u_behavior(self) -> None:
        good = {
            "same_u_preserve_rate": 1.0,
            "opposite_u_flip_rate": 1.0,
            "wrong_position_preserve_rate": 0.0,
            "wrong_content_preserve_rate": 0.0,
            "wrong_length_preserve_rate": 0.0,
            "same_u_mean_abs_margin_delta": 0.1,
            "opposite_u_mean_source_signed_shift": 10.0,
        }
        bad = {
            "same_u_preserve_rate": 1.0,
            "opposite_u_flip_rate": 0.0,
            "wrong_position_preserve_rate": 1.0,
            "wrong_content_preserve_rate": 1.0,
            "wrong_length_preserve_rate": 1.0,
            "same_u_mean_abs_margin_delta": 0.1,
            "opposite_u_mean_source_signed_shift": 0.0,
        }
        self.assertGreater(_causal_score(good), _causal_score(bad))
        self.assertEqual(len(_handle_signature(good)), 7)

    def test_selector_ranks_better_signature_first(self) -> None:
        summary = {
            "good": {
                "same_u_preserve_rate": 1.0,
                "opposite_u_flip_rate": 1.0,
                "wrong_position_preserve_rate": 0.0,
                "wrong_content_preserve_rate": 0.0,
                "wrong_length_preserve_rate": 0.0,
                "same_u_mean_abs_margin_delta": 0.1,
                "opposite_u_mean_source_signed_shift": 10.0,
            },
            "bad": {
                "same_u_preserve_rate": 1.0,
                "opposite_u_flip_rate": 0.0,
                "wrong_position_preserve_rate": 1.0,
                "wrong_content_preserve_rate": 1.0,
                "wrong_length_preserve_rate": 1.0,
                "same_u_mean_abs_margin_delta": 0.1,
                "opposite_u_mean_source_signed_shift": 0.0,
            },
        }
        payload = _selector_payload(summary, epsilon=0.1, beta_neural=0.1)
        self.assertEqual(payload["ranked_handles"][0]["handle_id"], "good")


if __name__ == "__main__":
    unittest.main()
