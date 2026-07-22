import unittest

from experiments.openai_sparse_plot.candidate_causal_models import CANDIDATE_MODELS, candidate_model_by_id


class CandidateCausalModelTests(unittest.TestCase):
    def test_expected_model_count_and_lookup(self) -> None:
        self.assertGreaterEqual(len(CANDIDATE_MODELS), 8)
        model = candidate_model_by_id("m7_internal_path_supernode_3")
        self.assertEqual(model.variable_count, 3)

    def test_internal_path_model_keeps_output_separate(self) -> None:
        model = candidate_model_by_id("m7_internal_path_supernode_3")
        variables = {var.variable_id: var for var in model.variables}
        self.assertIn("InternalQuoteTypePath", variables)
        self.assertNotIn("final_resid:83", variables["InternalQuoteTypePath"].node_ids)
        self.assertEqual(variables["Output"].role, "observed_output")


if __name__ == "__main__":
    unittest.main()
