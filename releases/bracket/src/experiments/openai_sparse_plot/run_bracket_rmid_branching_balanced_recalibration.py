from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .bracket_d_rich_signatures import load_full_localized_candidate_universe
from .bracket_group_depth_plot import build_unique_discovery_bank
from .bracket_joint_rmid_rlate import summarize_validation, validation_acceptance, validation_records
from .bracket_rmid_branching import (
    build_content_transition_balanced_specs,
    select_mediation_aware_candidate,
    summarize_blocking_records,
    summarize_factorial_records,
)
from .run_bracket_d_large_bank_frozen_readout_plot import _atomic_json, _atomic_jsonl, _collect_or_resume_clean_runs
from .run_bracket_group_depth_plot import _attach_phi, batched_patch_outputs
from .run_bracket_joint_rmid_rlate_plot import _base_phi, _clean_t2_accuracy
from .run_bracket_rmid_branching_factorial import (
    _blocking_records,
    _crossed_config,
    _factorial_records,
    _group_config,
    _report,
    batched_crossed_outputs,
)
from .run_bracket_threshold_component_plot import _load_frozen_readout
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import build_prefix_cache_bank, convert_transformer_linears_to_sparse


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_PRIOR_DIR = Path("eval/openai_sparse_plot/bracket_rmid_branching_factorial_20260713")
DEFAULT_PARENT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_rmid_branching_balanced_recalibration_20260713")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Balanced recalibration and confirmation of R_mid branching.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--prior-run-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fresh-content-offset", type=int, default=6000)
    parser.add_argument("--cal-contents", type=int, default=24)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--acceptance-threshold", type=float, default=0.90)
    parser.add_argument("--direct-fraction-tolerance", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260713)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--dense-kernels", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _load_direct_valid_candidates(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible = [row for row in rows if row["chain_eligible"] and row["direct_validated"]]
    grouped: dict[tuple[tuple[str, ...], tuple[float, ...]], list[Mapping[str, Any]]] = {}
    for row in eligible:
        key = (
            tuple(str(value) for value in row["site_ids"]),
            tuple(round(float(value), 10) for value in row["coefficients"]),
        )
        grouped.setdefault(key, []).append(row)
    candidates = []
    for index, (key, source_rows) in enumerate(grouped.items(), start=1):
        representative = dict(source_rows[0])
        representative["config_id"] = f"balanced-candidate-{index}"
        representative["source_settings"] = [
            {
                "epsilon": row["epsilon"],
                "beta": row["beta"],
                "k": row["k"],
                "strength": row["strength"],
            }
            for row in source_rows
        ]
        representative["site_ids"] = list(key[0])
        representative["coefficients"] = list(key[1])
        candidates.append(representative)
    if not candidates:
        raise ValueError("no Dcal-direct-valid distinct handles were found")
    return candidates


def _subset(array: Any, indices: list[int]) -> Any:
    return array[indices]


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = not bool(args.no_resume)
    device = "cuda" if args.cuda else "cpu"
    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv, expected_node_count=int(args.expected_node_count)
    )
    sites = universe.sites
    site_ids = [site.site_id for site in sites]
    prior_result = json.loads((args.prior_run_dir / "bracket_rmid_branching_factorial.json").read_text(encoding="utf-8"))
    prior_manifest = json.loads((args.prior_run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if int(prior_manifest["candidate_count"]) != 133 or not prior_manifest["no_filtering_applied_in_matching"]:
        raise ValueError("prior candidate rankings were not the full unfiltered all-133 singleton matching")
    rmid = prior_result["rmid"]
    candidates = _load_direct_valid_candidates(args.prior_run_dir / "calibration" / "grid.jsonl")
    manifest = {
        "schema_version": 1,
        "candidate_count": len(sites),
        "candidate_csv_sha256": universe.csv_sha256,
        "matching_source": "prior all-133 singleton coupling",
        "candidate_refinement": "all exact distinct R_late handles passing prior Dcal direct validation",
        "refined_candidate_count": len(candidates),
        "refined_candidates": candidates,
        "rmid_frozen": rmid,
        "pair_or_triple_matching": False,
        "record_balancing": "ordered depth transition plus base numeric content",
        "bootstrap_unit": "base numeric content",
        "fresh_content_offset": int(args.fresh_content_offset),
        "Dte_policy": "selection_manifest.json written before any Dte clean run or intervention",
    }
    _atomic_json(out_dir / "run_manifest.json", manifest)

    encoding = make_tinypython_encoding(args.circuit_home)
    readout = _load_frozen_readout(args.parent_run_dir / "frozen_depth_readout.json", site_ids)
    bank = build_unique_discovery_bank(
        encoding,
        contents=int(args.cal_contents) + int(args.test_contents),
        fit_contents=0,
        cal_contents=int(args.cal_contents),
        test_contents=int(args.test_contents),
        content_offset=int(args.fresh_content_offset),
        depths=(1, 2, 3, 4),
    )
    examples = {row.example_id: row for row in bank}
    specs = {
        split: build_content_transition_balanced_specs(
            bank, split=split, records_per_relation=int(args.records_per_relation)
        )
        for split in ("Dcal", "Dte")
    }
    _atomic_json(
        out_dir / "record_manifest.json",
        {
            split: [f"{row.relation}|{row.base_id}|{row.source_id}|{row.wrong_variable or ''}" for row in values]
            for split, values in specs.items()
        },
    )

    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not bool(args.no_flash),
        grad_checkpointing=False,
    )
    sparse_records = () if args.dense_kernels else convert_transformer_linears_to_sparse(model)
    model_info["execution_linear_kernel"] = "dense" if args.dense_kernels else "exact_sparse_csr"
    model_info["sparse_conversion"] = [row.to_json() for row in sparse_records]
    _atomic_json(out_dir / "model_info.json", model_info)
    single_close_token_id = int(encoding.encode("]\n")[0])
    double_close_token_id = int(encoding.encode("]]\n")[0])

    cal_ids = sorted({value for spec in specs["Dcal"] for value in (spec.base_id, spec.source_id)})
    cal_examples = [examples[value] for value in cal_ids]
    cal_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=cal_examples,
        candidate_sites=sites,
        checkpoint_path=out_dir / "clean_runs_Dcal.jsonl",
        resume=resume,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _attach_phi(readout, cal_runs)
    if not all(row.correct for row in cal_runs.values()) or _clean_t2_accuracy(cal_examples, cal_runs) < 1.0:
        raise RuntimeError("balanced Dcal clean model/readout accuracy is below 1.0")
    cal_prefix = build_prefix_cache_bank(
        model,
        {base_id: cal_runs[base_id].token_ids for base_id in sorted({row.base_id for row in specs["Dcal"]})},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    late_phi, late_margins = batched_patch_outputs(
        model,
        configs=[_group_config(row, sites) for row in candidates],
        specs=specs["Dcal"],
        clean_runs=cal_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=cal_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        include_margin=True,
    )
    if late_margins is None:
        raise AssertionError("balanced Dcal direct handles returned no margins")
    different_indices = [
        index
        for index, spec in enumerate(specs["Dcal"])
        if int(examples[spec.base_id].close_count) != int(examples[spec.source_id].close_count)
    ]
    different_specs = tuple(specs["Dcal"][index] for index in different_indices)
    mid_phi, mid_margins = batched_patch_outputs(
        model,
        configs=[_group_config(rmid, sites)],
        specs=different_specs,
        clean_runs=cal_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=cal_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        include_margin=True,
    )
    if mid_margins is None:
        raise AssertionError("balanced Dcal R_mid returned no margins")
    blocked_phi, blocked_margins = batched_crossed_outputs(
        model,
        configs=[
            _crossed_config(
                config_id=f"block:{row['config_id']}",
                mode="mid_source_late_base",
                rmid=rmid,
                rlate=row,
                sites=sites,
            )
            for row in candidates
        ],
        specs=different_specs,
        clean_runs=cal_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=cal_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
    )
    grid = []
    for index, candidate in enumerate(candidates):
        direct_records = validation_records(
            target="R_late",
            specs=specs["Dcal"],
            examples=examples,
            base_phi=_base_phi(specs["Dcal"], cal_runs),
            patched_delta_phi=late_phi[index].tolist(),
            patched_margins=late_margins[index].tolist(),
        )
        direct_summary = summarize_validation(direct_records)
        direct_acceptance = validation_acceptance(direct_summary, threshold=float(args.acceptance_threshold))
        block_records = _blocking_records(
            specs=different_specs,
            examples=examples,
            clean_runs=cal_runs,
            mid_phi=mid_phi[0],
            mid_margins=mid_margins[0],
            late_phi=_subset(late_phi[index], different_indices),
            late_margins=_subset(late_margins[index], different_indices),
            blocked_phi=blocked_phi[index],
            blocked_margins=blocked_margins[index],
        )
        block_summary = summarize_blocking_records(block_records, threshold=float(args.acceptance_threshold))
        grid.append(
            {
                **candidate,
                "direct_validated": bool(direct_acceptance["validated"]),
                "blocking_validated": bool(block_summary["validated"]),
                "direct_summary": direct_summary,
                "blocking_summary": block_summary,
                "score": (float(direct_summary["score"]) + float(block_summary["score"])) / 2.0,
                "worst_gate": min(float(direct_summary["worst_gate"]), float(block_summary["worst_gate"])),
            }
        )
    selected = dict(select_mediation_aware_candidate(grid))
    _atomic_jsonl(out_dir / "calibration" / "grid.jsonl", grid)
    _atomic_json(
        out_dir / "selection_manifest.json",
        {
            "selection_completed_before_Dte": True,
            "Dte_inputs_used": [],
            "selected": selected,
        },
    )

    test_ids = sorted({value for spec in specs["Dte"] for value in (spec.base_id, spec.source_id)})
    test_examples = [examples[value] for value in test_ids]
    test_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=test_examples,
        candidate_sites=sites,
        checkpoint_path=out_dir / "clean_runs_Dte.jsonl",
        resume=resume,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _attach_phi(readout, test_runs)
    if not all(row.correct for row in test_runs.values()) or _clean_t2_accuracy(test_examples, test_runs) < 1.0:
        raise RuntimeError("balanced Dte clean model/readout accuracy is below 1.0")
    test_prefix = build_prefix_cache_bank(
        model,
        {base_id: test_runs[base_id].token_ids for base_id in sorted({row.base_id for row in specs["Dte"]})},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    direct_phi, direct_margins = batched_patch_outputs(
        model,
        configs=[_group_config(rmid, sites), _group_config(selected, sites)],
        specs=specs["Dte"],
        clean_runs=test_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=test_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        include_margin=True,
    )
    if direct_margins is None:
        raise AssertionError("balanced Dte direct handles returned no margins")
    heldout_direct = {}
    for index, target in enumerate(("R_mid", "R_late")):
        records = validation_records(
            target=target,
            specs=specs["Dte"],
            examples=examples,
            base_phi=_base_phi(specs["Dte"], test_runs),
            patched_delta_phi=direct_phi[index].tolist(),
            patched_margins=direct_margins[index].tolist(),
        )
        summary = summarize_validation(records)
        heldout_direct[target] = {
            "summary": summary,
            "acceptance": validation_acceptance(summary, threshold=float(args.acceptance_threshold)),
        }
    test_different_indices = [
        index
        for index, spec in enumerate(specs["Dte"])
        if int(examples[spec.base_id].close_count) != int(examples[spec.source_id].close_count)
    ]
    test_different_specs = tuple(specs["Dte"][index] for index in test_different_indices)
    crossed_phi, crossed_margins = batched_crossed_outputs(
        model,
        configs=[
            _crossed_config(
                config_id="mid_source_late_base",
                mode="mid_source_late_base",
                rmid=rmid,
                rlate=selected,
                sites=sites,
            ),
            _crossed_config(
                config_id="mid_source_late_source",
                mode="mid_source_late_source",
                rmid=rmid,
                rlate=selected,
                sites=sites,
            ),
        ],
        specs=test_different_specs,
        clean_runs=test_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=test_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
    )
    factorial_records = _factorial_records(
        specs=test_different_specs,
        examples=examples,
        clean_runs=test_runs,
        mid_phi=_subset(direct_phi[0], test_different_indices),
        mid_margins=_subset(direct_margins[0], test_different_indices),
        late_phi=_subset(direct_phi[1], test_different_indices),
        late_margins=_subset(direct_margins[1], test_different_indices),
        blocked_phi=crossed_phi[0],
        blocked_margins=crossed_margins[0],
        both_phi=crossed_phi[1],
        both_margins=crossed_margins[1],
    )
    factorial = summarize_factorial_records(
        factorial_records,
        threshold=float(args.acceptance_threshold),
        direct_fraction_tolerance=float(args.direct_fraction_tolerance),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    all_direct_valid = all(row["acceptance"]["validated"] for row in heldout_direct.values())
    decision = {
        "all_direct_handles_validated": all_direct_valid,
        "pure_chain_validated": bool(all_direct_valid and factorial["pure_chain_validated"]),
        "branching_model_validated": bool(all_direct_valid and factorial["symmetric_branch_validated"]),
        "conclusion": factorial["conclusion"] if all_direct_valid else "direct_handle_validation_failed",
    }
    _atomic_jsonl(out_dir / "heldout" / "factorial_records.jsonl", factorial_records)
    result = {
        "rmid": rmid,
        "selection": selected,
        "selection_completed_before_Dte": True,
        "clean": {"Dcal_accuracy": 1.0, "Dte_accuracy": 1.0},
        "heldout_direct": heldout_direct,
        "factorial": factorial,
        "decision": decision,
    }
    _atomic_json(out_dir / "bracket_rmid_branching_balanced_recalibration.json", result)
    _report(out_dir / "bracket_rmid_branching_balanced_recalibration.md", result)
    print(json.dumps({"status": "complete", **decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
