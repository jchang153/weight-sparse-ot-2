from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from bracket_repro.audit import audit
from bracket_repro.cli import _run_chain, _run_coarse
from bracket_repro.common import candidate_ids, release_root, sha256_file
from bracket_repro.methods import abstract_joint_rows, abstract_r_delta, neural_output_delta, r_from_depth


class BracketReleaseTests(unittest.TestCase):
    def test_full_candidate_universe(self) -> None:
        root = release_root()
        path = root / "data" / "bracket_circuit_nodes.csv"
        self.assertEqual(len(candidate_ids(path)), 133)
        self.assertEqual(sha256_file(path), "4379a582f1d57051e5e8ebbf7e84252c738bd6708da4646e08f7e23d967be547")

    def test_binary_r_semantics(self) -> None:
        self.assertEqual([r_from_depth(value) for value in (1, 2, 3, 4)], [0, 1, 1, 1])
        self.assertEqual(abstract_r_delta(2, 1), 1.0)
        self.assertEqual(abstract_r_delta(4, 2), 0.0)
        self.assertEqual(neural_output_delta(swapped_margin=2.0, base_margin=-3.0), 5.0)

    def test_joint_rows(self) -> None:
        mid, late = abstract_joint_rows(delta_t2=1, delta_p_one=-0.5, delta_p_two=0.5)
        self.assertEqual(mid, (1.0, -0.5, 0.5))
        self.assertEqual(late, (0.0, -0.5, 0.5))

    def test_dte_is_not_a_selection_input(self) -> None:
        config = json.loads((release_root() / "configs" / "bracket_chain.json").read_text())
        self.assertEqual(config["candidate_count"], 133)
        self.assertEqual(config["candidate_filtering"], "none")
        self.assertEqual(config["Dte_selection_use"], "forbidden")

    def test_frozen_audit_rejects_uncertified_models(self) -> None:
        result = audit()
        self.assertEqual(result["B0"]["status"], "certified")
        self.assertTrue(result["R_mid"]["validated"])
        self.assertFalse(result["R_late"]["validated"])
        self.assertEqual(result["B1_pure_chain"]["status"], "not_certified")
        self.assertEqual(result["B2_bypass"]["status"], "not_certified")

    def test_no_checkpoint_is_distributed(self) -> None:
        forbidden = [path for path in release_root().rglob("*.pt") if ".cache" not in path.parts]
        self.assertEqual(forbidden, [])

    def test_public_model_and_task_artifacts_are_pinned(self) -> None:
        metadata = json.loads((release_root() / "MODEL_ARTIFACTS.json").read_text())
        self.assertEqual(metadata["model_name"], "csp_yolo2")
        self.assertFalse(metadata["model_weights_in_zip"])
        self.assertEqual(metadata["circuit_sparsity_commit"], "dbf1fe0d27b76c19e10d2a715f28c2e5da535e08")
        artifacts = {row["name"]: row for row in metadata["artifacts"]}
        self.assertEqual(
            set(artifacts),
            {"beeg_config.json", "final_model.pt", "bracket_counting_viz_data.pt"},
        )
        self.assertEqual(artifacts["final_model.pt"]["bytes"], 1676532054)
        self.assertEqual(
            artifacts["final_model.pt"]["sha256"],
            "988cf534a28ab296b45776a29ea91eb3e942c3895fd18eaabfa3f5bd1e7fbd85",
        )
        self.assertEqual(artifacts["bracket_counting_viz_data.pt"]["bytes"], 13623136)
        self.assertEqual(
            artifacts["bracket_counting_viz_data.pt"]["sha256"],
            "f6c39c970d6aaa37834bfbc5dedbf0f2a974fed3a235f2625e95b2b20dbf16d4",
        )

    def test_full_coarse_command_has_no_candidate_limit(self) -> None:
        with patch("bracket_repro.cli.run_module") as run_module:
            _run_coarse(cuda=True, smoke=False)
        self.assertEqual(run_module.call_count, 2)
        scan_args = run_module.call_args_list[0].args[1]
        self.assertIn("--bracket-node-csv", scan_args)
        self.assertNotIn("--max-sites", scan_args)
        self.assertIn("--cuda", scan_args)

    def test_chain_orchestrates_all_four_resumable_stages(self) -> None:
        with patch("bracket_repro.cli.run_module") as run_module:
            _run_chain(cuda=True, resume=True, smoke=False)
        self.assertEqual(
            [call.args[0] for call in run_module.call_args_list],
            [
                "experiments.openai_sparse_plot.run_bracket_d_large_bank_frozen_readout_plot",
                "experiments.openai_sparse_plot.run_bracket_joint_rmid_rlate_plot",
                "experiments.openai_sparse_plot.run_bracket_rmid_branching_factorial",
                "experiments.openai_sparse_plot.run_bracket_rmid_branching_balanced_recalibration",
            ],
        )
        for call in run_module.call_args_list:
            args = call.args[1]
            self.assertEqual(args[args.index("--expected-node-count") + 1], "133")
            self.assertNotIn("--no-resume", args)
            self.assertIn("--cuda", args)


if __name__ == "__main__":
    unittest.main()
