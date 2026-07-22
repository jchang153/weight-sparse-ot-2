from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .audit import audit
from .common import download_model, release_root, run_module


def _run_certified(*, cuda: bool, smoke: bool) -> None:
    if smoke:
        from .smoke import run as run_smoke

        run_smoke(cuda=cuda)
        return
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    args = [
        "--task", "quote",
        "--circuit-home", str(circuit),
        "--out-dir", str(root / "outputs" / "quote_certified"),
        "--quote-hard-json", str(root / "artifacts" / "inputs" / "unmatched_quote_abstraction.json"),
        "--quote-node-csv", str(root / "data" / "quote_circuit_nodes.csv"),
        "--quote-candidate-source", "node_csv",
        "--quote-max-sites", "0",
        "--max-records-per-relation", "6",
        "--k-grid", "1,2,3,5,8",
        "--strength-grid", "0.5,1.0,2.0,4.0",
        "--selector-epsilon", "0.08",
        "--selector-beta", "0.08",
    ]
    if cuda:
        args.append("--cuda")
    run_module("experiments.openai_sparse_plot.run_raw_delta_plot_abstraction", args)


def _run_pointer(*, cuda: bool, smoke: bool) -> None:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    common = ["--circuit-home", str(circuit)]
    if cuda:
        common.append("--cuda")
    run_module(
        "experiments.openai_sparse_plot.run_position_routing_diagnostic",
        [*common, "--out-dir", str(root / "outputs" / "pointer_routing")],
    )
    if not smoke:
        run_module(
            "experiments.openai_sparse_plot.run_nonquote_route_value_diagnostic",
            [*common, "--out-dir", str(root / "outputs" / "nonquote_copy")],
        )


def _report(destination: Path | None) -> None:
    root = release_root()
    source = root / "REPORT_QUOTE.md"
    destination = root / "outputs" / "REPORT_QUOTE.md" if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the OpenAI localized quote-circuit PLOT results.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Recompute the compact frozen-record audit.")
    sub.add_parser("download-model", help="Download and verify csp_yolo1.")
    report = sub.add_parser("report", help="Write the self-contained Markdown report.")
    report.add_argument("--out", type=Path)
    run = sub.add_parser("run", help="Run an actual-model experiment.")
    run.add_argument("--experiment", choices=("certified", "pointer-copy"), required=True)
    run.add_argument("--cuda", action="store_true")
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2, sort_keys=True))
    elif args.command == "download-model":
        download_model("csp_yolo1")
    elif args.command == "report":
        _report(args.out)
    elif args.experiment == "certified":
        _run_certified(cuda=args.cuda, smoke=args.smoke)
    else:
        _run_pointer(cuda=args.cuda, smoke=args.smoke)
