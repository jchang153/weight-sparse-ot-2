from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from quote_repro.audit import audit
from quote_repro.cli import _run_certified
from quote_repro.common import candidate_ids, release_root, sha256_file
from quote_repro.methods import abstract_quote_delta, neural_quote_delta


class QuoteReleaseTests(unittest.TestCase):
    def test_full_candidate_universe(self) -> None:
        root = release_root()
        path = root / "data" / "quote_circuit_nodes.csv"
        self.assertEqual(len(candidate_ids(path)), 64)
        self.assertEqual(sha256_file(path), "c38db7b63313960577c6b214f3bdb8979d126532b7afe5ee1f853f6cdf2ae01a")

    def test_effect_signature_is_source_minus_base(self) -> None:
        self.assertEqual(abstract_quote_delta("double", "single"), 2.0)
        self.assertEqual(abstract_quote_delta("single", "double"), -2.0)
        self.assertAlmostEqual(neural_quote_delta(swapped_margin=3.5, base_margin=-1.0), 4.5)

    def test_q2_sites_are_explicit_diagnostics(self) -> None:
        config = json.loads((release_root() / "configs" / "quote_pointer_copy_diagnostics.json").read_text())
        self.assertEqual(config["status"], "diagnostic_only")
        self.assertEqual(config["Q"], "X_P restricted to single/double quote type")

    def test_frozen_audit(self) -> None:
        result = audit()
        self.assertEqual(result["Q1"]["status"], "certified")
        self.assertEqual(result["Q2"]["status"], "not_certified")

    def test_no_checkpoint_is_distributed(self) -> None:
        forbidden = [path for path in release_root().rglob("*.pt") if ".cache" not in path.parts]
        self.assertEqual(forbidden, [])

    def test_public_model_artifacts_are_pinned(self) -> None:
        metadata = json.loads((release_root() / "MODEL_ARTIFACTS.json").read_text())
        self.assertEqual(metadata["model_name"], "csp_yolo1")
        self.assertFalse(metadata["model_weights_in_zip"])
        self.assertEqual(metadata["circuit_sparsity_commit"], "dbf1fe0d27b76c19e10d2a715f28c2e5da535e08")
        artifacts = {row["name"]: row for row in metadata["artifacts"]}
        self.assertEqual(set(artifacts), {"beeg_config.json", "final_model.pt"})
        self.assertEqual(artifacts["final_model.pt"]["bytes"], 642319670)
        self.assertEqual(
            artifacts["final_model.pt"]["sha256"],
            "a3a68c34c07ec2b72f3239043d9169d6af7225c3d197e31cc871a9c1aba04b5d",
        )

    def test_full_certified_command_uses_all_64_sites(self) -> None:
        with patch("quote_repro.cli.run_module") as run_module:
            _run_certified(cuda=True, smoke=False)
        run_module.assert_called_once()
        module, args = run_module.call_args.args
        self.assertEqual(module, "experiments.openai_sparse_plot.run_raw_delta_plot_abstraction")
        self.assertEqual(args[args.index("--quote-max-sites") + 1], "0")
        self.assertEqual(args[args.index("--quote-candidate-source") + 1], "node_csv")
        self.assertIn("--cuda", args)


if __name__ == "__main__":
    unittest.main()
