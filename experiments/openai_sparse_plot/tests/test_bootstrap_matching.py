from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.run_bootstrap_matching import _summarize


class TestBootstrapMatching(unittest.TestCase):
    def test_summarize_expected_top1_rate(self) -> None:
        samples = [
            {
                "expected_rank_audit": {
                    "Output": {
                        "top_site": "final_resid:83",
                        "expected_family_mass": 0.5,
                    }
                }
            },
            {
                "expected_rank_audit": {
                    "Output": {
                        "top_site": "wrong",
                        "expected_family_mass": 0.1,
                    }
                }
            },
        ]
        summary = _summarize(samples)["Output"]
        self.assertEqual(summary["expected_top1_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_expected_family_mass"], 0.3)

    def test_summarize_uses_requested_audit_key(self) -> None:
        samples = [
            {
                "stage_aware_expected_rank_audit": {
                    "Output": {
                        "top_site": "final_resid:83",
                        "expected_family_mass": 0.8,
                    }
                }
            }
        ]
        summary = _summarize(samples, audit_key="stage_aware_expected_rank_audit")["Output"]
        self.assertEqual(summary["expected_top1_rate"], 1.0)
        self.assertAlmostEqual(summary["mean_expected_family_mass"], 0.8)


if __name__ == "__main__":
    unittest.main()
