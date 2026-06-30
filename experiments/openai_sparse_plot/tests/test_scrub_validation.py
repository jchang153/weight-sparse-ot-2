from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.run_scrub_validation import _quote_sign_from_margin, _summarize


class TestScrubValidation(unittest.TestCase):
    def test_quote_sign_from_margin(self) -> None:
        self.assertEqual(_quote_sign_from_margin(0.1), 1)
        self.assertEqual(_quote_sign_from_margin(-0.1), -1)

    def test_summarize_rates(self) -> None:
        rows = [
            {
                "site_id": "site",
                "site_label": "label",
                "relation": "same_quote_type",
                "patched_preserves_base_sign": True,
                "closer_to_base_than_source": True,
                "patched_margin": 1.5,
                "base_margin": 1.0,
                "source_margin": 2.0,
                "patched_matches_source_sign": True,
                "moves_toward_source_sign": True,
                "source_sign": 1,
            },
            {
                "site_id": "site",
                "site_label": "label",
                "relation": "different_quote_type",
                "patched_preserves_base_sign": False,
                "closer_to_base_than_source": False,
                "patched_margin": -1.0,
                "base_margin": 1.0,
                "source_margin": -2.0,
                "patched_matches_source_sign": True,
                "moves_toward_source_sign": True,
                "source_sign": -1,
            },
        ]
        summary = _summarize(rows)["site"]
        self.assertEqual(summary["same_preserve_sign_rate"], 1.0)
        self.assertEqual(summary["different_flip_to_source_rate"], 1.0)
        self.assertEqual(summary["different_mean_source_signed_shift"], 2.0)


if __name__ == "__main__":
    unittest.main()
