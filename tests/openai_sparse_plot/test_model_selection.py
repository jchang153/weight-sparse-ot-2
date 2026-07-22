from __future__ import annotations

import unittest

from experiments.openai_sparse_plot.model_selection import (
    CandidateModelSpec,
    score_candidate_model,
    select_simplest_passing,
)


class ModelSelectionTest(unittest.TestCase):
    def test_score_uses_min_and_max_thresholds(self) -> None:
        spec = CandidateModelSpec(
            model_id="m",
            label="model",
            variable_count=1,
            neural_site_count=2,
            required_min_metrics=("same", "flip"),
            required_max_metrics=("wrong_preserve",),
        )
        score = score_candidate_model(
            spec,
            {"same": 1.0, "flip": 0.95, "wrong_preserve": 0.05},
        )
        self.assertTrue(score.passed)
        self.assertEqual(score.failed_metrics, ())

        failed = score_candidate_model(
            spec,
            {"same": 1.0, "flip": 0.80, "wrong_preserve": 0.05},
        )
        self.assertFalse(failed.passed)
        self.assertEqual(failed.failed_metrics, ("flip",))

    def test_select_simplest_passing_prefers_smaller_model(self) -> None:
        rich = score_candidate_model(
            CandidateModelSpec("rich", "rich", 3, 2, ("same",), ()),
            {"same": 1.0},
        )
        simple = score_candidate_model(
            CandidateModelSpec("simple", "simple", 2, 5, ("same",), ()),
            {"same": 0.91},
        )
        self.assertEqual(select_simplest_passing([rich, simple]).model_id, "simple")


if __name__ == "__main__":
    unittest.main()
