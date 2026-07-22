from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .audit import audit
from .common import download_model, release_root, run_module


def _flag(cuda: bool) -> list[str]:
    return ["--cuda"] if cuda else []


def _run_coarse(*, cuda: bool, smoke: bool) -> None:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    scan = root / "outputs" / ("coarse_smoke_scan" if smoke else "coarse_scan")
    plot = root / "outputs" / ("coarse_smoke_plot" if smoke else "coarse_plot")
    common = [
        "--circuit-home", str(circuit),
        "--bracket-node-csv", str(root / "data" / "bracket_circuit_nodes.csv"),
        "--max-records-per-relation", "1" if smoke else "6",
        *_flag(cuda),
    ]
    run_module("experiments.openai_sparse_plot.run_bracket_raw_delta_scan", [*common, "--out-dir", str(scan)])
    run_module(
        "experiments.openai_sparse_plot.run_bracket_raw_delta_plot_from_scan",
        [
            *common,
            "--scan-json", str(scan / "bracket_raw_delta_scan.json"),
            "--records-jsonl", str(scan / "singleton_records.jsonl"),
            "--out-dir", str(plot),
            "--k-grid", "1" if smoke else "1,2,3,5,8",
            "--strength-grid", "1.0" if smoke else "0.5,1.0,2.0,4.0",
        ],
    )


def _run_chain(*, cuda: bool, resume: bool, smoke: bool) -> None:
    root = release_root()
    circuit = Path(os.environ.get("CIRCUIT_SPARSITY_HOME", root / ".external" / "circuit_sparsity"))
    candidate = root / "data" / "bracket_circuit_nodes.csv"
    if smoke:
        from .smoke import run as run_smoke

        run_smoke(cuda=cuda)
        return
    chain = root / "outputs" / "chain"
    parent = chain / "01_frozen_t2"
    joint = chain / "02_joint_plot"
    factorial = chain / "03_factorial"
    balanced = chain / "04_balanced_final"
    no_resume = [] if resume else ["--no-resume"]
    base = ["--circuit-home", str(circuit), "--model", "csp_yolo2", "--candidate-node-csv", str(candidate), "--expected-node-count", "133", *_flag(cuda)]
    run_module(
        "experiments.openai_sparse_plot.run_bracket_d_large_bank_frozen_readout_plot",
        [*base, "--out-dir", str(parent), "--contents", "96", "--fit-contents", "48", "--cal-contents", "24", "--test-contents", "24", "--records-per-relation", "100", "--k-grid", "1,2,3,5,8", "--strength-grid", "0.5,1.0,2.0,4.0", "--selector-epsilon", "0.08", "--selector-beta", "0.08", *no_resume],
    )
    run_module(
        "experiments.openai_sparse_plot.run_bracket_joint_rmid_rlate_plot",
        [*base, "--parent-run-dir", str(parent), "--out-dir", str(joint), "--records-per-relation", "100", "--epsilon-grid", "0.02,0.08,0.32", "--beta-grid", "0.02,0.08,0.32", "--k-grid", "1,2,3,5,8", "--strength-grid", "0.5,1.0,2.0,4.0", *no_resume],
    )
    run_module(
        "experiments.openai_sparse_plot.run_bracket_rmid_branching_factorial",
        [*base, "--prior-run-dir", str(joint), "--parent-run-dir", str(parent), "--out-dir", str(factorial), "--records-per-relation", "100", "--k-grid", "1,2,3,5,8", "--strength-grid", "0.5,1.0,2.0,4.0", *no_resume],
    )
    run_module(
        "experiments.openai_sparse_plot.run_bracket_rmid_branching_balanced_recalibration",
        [*base, "--prior-run-dir", str(factorial), "--parent-run-dir", str(parent), "--out-dir", str(balanced), "--fresh-content-offset", "6000", "--cal-contents", "24", "--test-contents", "24", "--records-per-relation", "100", *no_resume],
    )


def _report(destination: Path | None) -> None:
    root = release_root()
    source = root / "REPORT_BRACKET.md"
    destination = root / "outputs" / "REPORT_BRACKET.md" if destination is None else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the OpenAI localized bracket-circuit PLOT results.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("download-model")
    report = sub.add_parser("report")
    report.add_argument("--out", type=Path)
    run = sub.add_parser("run")
    run.add_argument("--experiment", choices=("coarse", "chain"), required=True)
    run.add_argument("--cuda", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2, sort_keys=True))
    elif args.command == "download-model":
        download_model("csp_yolo2")
    elif args.command == "report":
        _report(args.out)
    elif args.experiment == "coarse":
        _run_coarse(cuda=args.cuda, smoke=args.smoke)
    else:
        _run_chain(cuda=args.cuda, resume=args.resume, smoke=args.smoke)
