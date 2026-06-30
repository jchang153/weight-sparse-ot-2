from __future__ import annotations

import unittest

import torch

from experiments.openai_sparse_plot.plot_matching import cost_matrix, fit_matching
from experiments.openai_sparse_plot.run_plot_matching import (
    EXPECTED_SITE_FAMILIES,
    duplicate_abstract_signature_groups,
    expected_rank_audit,
    stage_aware_uot_payload,
)
from experiments.openai_sparse_plot.run_raw_delta_plot_abstraction import _brute_force_singleton_behavior
from experiments.openai_sparse_plot.schema import EffectSignatureTable


class TestPlotMatching(unittest.TestCase):
    def test_cost_matrix_prefers_identical_rows(self) -> None:
        x = torch.eye(3)
        cost = cost_matrix(x, x, mode="squared")
        self.assertTrue(torch.equal(torch.argmin(cost, dim=1), torch.arange(3)))

    def test_fit_matching_argmin(self) -> None:
        table = EffectSignatureTable.from_sequences(
            abstract_variable_ids=("A", "B"),
            neural_site_ids=("n0", "n1"),
            abstract_signatures=((1.0, 0.0), (0.0, 1.0)),
            neural_signatures=((0.0, 1.0), (1.0, 0.0)),
            feature_names=("f0", "f1"),
        )
        result = fit_matching(table, method="argmin", cost_mode="squared")
        self.assertEqual(result.top_matches(1)["A"][0][0], "n1")
        self.assertEqual(result.top_matches(1)["B"][0][0], "n0")

    def test_uot_rows_are_normalized(self) -> None:
        table = EffectSignatureTable.from_sequences(
            abstract_variable_ids=("A", "B"),
            neural_site_ids=("n0", "n1", "n2"),
            abstract_signatures=((1.0, 0.0), (0.0, 1.0)),
            neural_signatures=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            feature_names=("f0", "f1"),
        )
        result = fit_matching(table, method="uot", cost_mode="squared", epsilon=0.1, beta_neural=0.1, n_iter=40)
        self.assertTrue(torch.allclose(result.coupling.sum(dim=1), torch.ones(2), atol=1e-5))

    def test_duplicate_abstract_signature_groups(self) -> None:
        table = EffectSignatureTable.from_sequences(
            abstract_variable_ids=("OpeningQuoteType", "StoredQuoteType", "Output"),
            neural_site_ids=("n0",),
            abstract_signatures=((1.0, 2.0), (1.0, 2.0), (0.0, 1.0)),
            neural_signatures=((1.0, 2.0),),
            feature_names=("f0", "f1"),
        )
        self.assertEqual(
            duplicate_abstract_signature_groups(table),
            [["OpeningQuoteType", "StoredQuoteType"]],
        )

    def test_expected_rank_audit_reports_expected_family_rank(self) -> None:
        table = EffectSignatureTable.from_sequences(
            abstract_variable_ids=("Output",),
            neural_site_ids=("wrong", "final_resid:83", "10.attn.resid_delta:83"),
            abstract_signatures=((0.0, 1.0),),
            neural_signatures=((1.0, 0.0), (0.0, 1.0), (0.0, 0.5)),
            feature_names=("f0", "f1"),
        )
        result = fit_matching(table, method="argmin", cost_mode="squared")
        audit = expected_rank_audit(result)["Output"]
        self.assertEqual(audit["best_expected_rank"], 1)
        self.assertEqual(audit["best_expected_site"], "final_resid:83")

    def test_opening_quote_type_expected_family_is_not_stored_family(self) -> None:
        self.assertIn("0.mlp.post_act:863", EXPECTED_SITE_FAMILIES["OpeningQuoteType"])
        self.assertNotEqual(EXPECTED_SITE_FAMILIES["OpeningQuoteType"], EXPECTED_SITE_FAMILIES["StoredQuoteType"])

    def test_stage_aware_uot_can_prefer_stage_matched_output_site(self) -> None:
        table = EffectSignatureTable.from_sequences(
            abstract_variable_ids=("Output",),
            neural_site_ids=("0.mlp.resid_delta:460", "final_resid:83"),
            abstract_signatures=((1.0, 0.0),),
            neural_signatures=((1.0, 0.0), (0.9, 0.1)),
            feature_names=("f0", "f1"),
        )
        payload = stage_aware_uot_payload(table, top_k=1, penalty=1.0)
        self.assertEqual(payload["top_matches"]["Output"][0][0], "final_resid:83")

    def test_brute_force_singleton_behavior_selects_best_quote_site(self) -> None:
        summaries = {
            "weak": {
                "same_u_preserve_rate": 1.0,
                "opposite_u_flip_rate": 0.0,
                "wrong_position_preserve_rate": 0.0,
                "wrong_content_preserve_rate": 0.0,
                "wrong_length_preserve_rate": 0.0,
                "opposite_u_mean_source_signed_shift": 1.0,
            },
            "strong": {
                "same_u_preserve_rate": 1.0,
                "opposite_u_flip_rate": 1.0,
                "wrong_position_preserve_rate": 0.0,
                "wrong_content_preserve_rate": 0.0,
                "wrong_length_preserve_rate": 0.0,
                "opposite_u_mean_source_signed_shift": 12.0,
            },
        }
        result = _brute_force_singleton_behavior(
            task="quote",
            summaries=summaries,
            abstract=(1.0, 0.0),
            neural_by_id={"weak": (0.9, 0.1), "strong": (0.8, 0.2)},
        )

        self.assertEqual(result["selected"]["handle_id"], "strong")
        self.assertGreater(result["selected"]["behavior_score"], result["ranked_sites"][1]["behavior_score"])


if __name__ == "__main__":
    unittest.main()
