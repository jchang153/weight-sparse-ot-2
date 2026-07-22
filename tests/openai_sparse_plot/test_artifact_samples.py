import unittest

from experiments.openai_sparse_plot.artifact_samples import (
    expected_quote_from_paired_sample_index,
    prediction_from_margin,
    summarize_artifact_sample_records,
    trim_trailing_padding,
)


class ArtifactSampleTests(unittest.TestCase):
    def test_trim_trailing_padding_keeps_internal_zeros(self) -> None:
        self.assertEqual(trim_trailing_padding([4, 0, 5, 0, 0]), (4, 0, 5))

    def test_paired_expected_quote_labels(self) -> None:
        self.assertEqual(expected_quote_from_paired_sample_index(0, 32), "double")
        self.assertEqual(expected_quote_from_paired_sample_index(15, 32), "double")
        self.assertEqual(expected_quote_from_paired_sample_index(16, 32), "single")
        self.assertEqual(expected_quote_from_paired_sample_index(31, 32), "single")
        with self.assertRaises(ValueError):
            expected_quote_from_paired_sample_index(0, 31)

    def test_prediction_and_summary(self) -> None:
        self.assertEqual(prediction_from_margin(0.1), "double")
        self.assertEqual(prediction_from_margin(-0.1), "single")
        records = [
            {"correct": True, "double_minus_single_margin": 2.0},
            {"correct": False, "double_minus_single_margin": -0.2},
        ]
        summary = summarize_artifact_sample_records(records)
        self.assertEqual(summary["num_samples"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["num_margin_lt_1"], 1)
        self.assertEqual(len(summary["incorrect_samples"]), 1)


if __name__ == "__main__":
    unittest.main()
