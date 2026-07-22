from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .activation import ChannelSite
from .bracket_d_large_bank_frozen_readout import (
    build_transition_balanced_specs,
    canonical_sha256,
    spec_key,
)
from .bracket_d_rich_signatures import FrozenDepthReadout, load_full_localized_candidate_universe
from .bracket_group_depth_plot import GroupCandidate, build_unique_discovery_bank
from .bracket_joint_rmid_rlate import (
    TARGETS,
    abstract_joint_signatures,
    fit_joint_coupling,
    neural_joint_signatures,
    ranked_row,
    restricted_output_probabilities,
    select_global_coupling,
    summarize_validation,
    topk_handle,
    validation_acceptance,
    validation_records,
)
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .run_bracket_d1249_r1079_mediation_test import PatchSpec, _run_patch_and_record, mediation_fraction
from .run_bracket_d_large_bank_frozen_readout_plot import _atomic_json, _atomic_jsonl, _collect_or_resume_clean_runs
from .run_bracket_group_depth_plot import (
    PatchConfiguration,
    _attach_phi,
    _load_parent,
    _save_prefix_cache_bank,
    batched_patch_outputs,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import build_prefix_cache_bank, convert_transformer_linears_to_sparse


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_PARENT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_joint_rmid_rlate_20260713")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint singleton PLOT for X -> R_mid -> R_late -> Y.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--fresh-content-offset", type=int, default=2000)
    parser.add_argument("--cal-contents", type=int, default=24)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--epsilon-grid", default="0.02,0.08,0.32")
    parser.add_argument("--beta-grid", default="0.02,0.08,0.32")
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--acceptance-threshold", type=float, default=0.90)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--dense-kernels", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("float grids must contain positive values")
    return values


def _int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer grids must contain positive values")
    return values


def _fresh_manifest(examples: Sequence[MultiDepthBracketExample]) -> dict[str, Any]:
    out: dict[str, Any] = {"total_examples": len(examples), "splits": {}}
    for split in ("Dcal", "Dte"):
        rows = [example for example in examples if example.split == split]
        out["splits"][split] = {
            "examples": len(rows),
            "content_count": len({example.numeric_content for example in rows}),
            "example_ids": [example.example_id for example in rows],
            "depths": sorted({int(example.depth) for example in rows}),
            "contexts": sorted({str(example.context_family) for example in rows}),
        }
    return out


def _clean_t2_accuracy(
    examples: Sequence[MultiDepthBracketExample],
    runs: Mapping[str, Any],
) -> float:
    rows = [
        int(float(runs[example.example_id].depth_phi[1]) >= 0.5) == int(example.depth >= 2)
        for example in examples
    ]
    return sum(float(value) for value in rows) / len(rows)


def _base_margins(specs: Sequence[MultiDepthResamplingSpec], runs: Mapping[str, Any]) -> list[float]:
    return [float(runs[spec.base_id].margin) for spec in specs]


def _base_phi(specs: Sequence[MultiDepthResamplingSpec], runs: Mapping[str, Any]) -> list[tuple[float, ...]]:
    return [tuple(float(value) for value in runs[spec.base_id].depth_phi) for spec in specs]


def _singletons(sites: Sequence[ChannelSite]) -> tuple[GroupCandidate, ...]:
    return tuple(
        GroupCandidate(group_id=site.site_id, site_indices=(index,), site_ids=(site.site_id,))
        for index, site in enumerate(sites)
    )


def _scan_dfit_singletons(
    model: Any,
    *,
    sites: Sequence[ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    clean_runs: Mapping[str, Any],
    readout: FrozenDepthReadout,
    prefix_cache_bank: Mapping[str, Any],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
    max_batch_size: int,
    out_dir: Path,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray]:
    phi_path = out_dir / "signatures" / "Dfit_singleton_delta_phi.npy"
    margin_path = out_dir / "signatures" / "Dfit_singleton_patched_margins.npy"
    if resume and phi_path.exists() and margin_path.exists():
        phi = np.load(phi_path)
        margins = np.load(margin_path)
        if phi.shape != (len(sites), len(specs), 4) or margins.shape != (len(sites), len(specs)):
            raise ValueError("persisted Dfit singleton signature shapes differ")
        return phi, margins
    configs = [
        PatchConfiguration(
            config_id=group.group_id,
            group=group,
            strength=1.0,
            ranking_position=index + 1,
        )
        for index, group in enumerate(_singletons(sites))
    ]
    phi, margins = batched_patch_outputs(
        model,
        configs=configs,
        specs=specs,
        clean_runs=clean_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=prefix_cache_bank,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(max_batch_size),
        include_margin=True,
    )
    if margins is None:
        raise AssertionError("Dfit singleton scan did not return margins")
    phi_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(phi_path, phi)
    np.save(margin_path, margins)
    return phi, margins


def _coupling_sweep(
    *,
    abstract: torch.Tensor,
    neural: torch.Tensor,
    site_ids: Sequence[str],
    epsilons: Sequence[float],
    betas: Sequence[float],
) -> dict[tuple[float, float], dict[str, Any]]:
    out = {}
    for epsilon in epsilons:
        for beta in betas:
            cost, coupling = fit_joint_coupling(abstract, neural, epsilon=float(epsilon), beta=float(beta))
            out[(float(epsilon), float(beta))] = {
                "cost": cost,
                "coupling": coupling,
                "rankings": {
                    target: ranked_row(target=target, site_ids=site_ids, cost=cost, coupling=coupling)
                    for target in TARGETS
                },
            }
    return out


def _coupling_payload(sweep: Mapping[tuple[float, float], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (epsilon, beta), payload in sorted(sweep.items()):
        rows.append(
            {
                "epsilon": epsilon,
                "beta": beta,
                "cost": payload["cost"].tolist(),
                "coupling": payload["coupling"].tolist(),
                "rankings": payload["rankings"],
            }
        )
    return rows


def _calibration_configs(
    *,
    sweep: Mapping[tuple[float, float], Mapping[str, Any]],
    sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
) -> tuple[list[PatchConfiguration], list[dict[str, Any]]]:
    site_index = {site.site_id: index for index, site in enumerate(sites)}
    configs: list[PatchConfiguration] = []
    metadata: list[dict[str, Any]] = []
    for (epsilon, beta), payload in sorted(sweep.items()):
        for target in TARGETS:
            ranked = payload["rankings"][target]
            for k in k_grid:
                site_ids, weights = topk_handle(ranked, k=int(k))
                group = GroupCandidate(
                    group_id=" + ".join(site_ids),
                    site_indices=tuple(site_index[site_id] for site_id in site_ids),
                    site_ids=site_ids,
                )
                for strength in strength_grid:
                    handle_id = f"{target}:eps{epsilon:g}:beta{beta:g}:K{k}:lambda{strength:g}"
                    configs.append(
                        PatchConfiguration(
                            config_id=handle_id,
                            group=group,
                            strength=1.0,
                            ranking_position=1,
                            coefficients=tuple(float(strength) * float(weight) for weight in weights),
                        )
                    )
                    metadata.append(
                        {
                            "handle_id": handle_id,
                            "target": target,
                            "epsilon": float(epsilon),
                            "beta": float(beta),
                            "k": int(k),
                            "strength": float(strength),
                            "site_ids": list(site_ids),
                            "weights": list(weights),
                            "coefficients": [float(strength) * float(weight) for weight in weights],
                        }
                    )
    return configs, metadata


def _evaluate_config_grid(
    *,
    metadata: Sequence[Mapping[str, Any]],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, Any],
    delta_phi: np.ndarray,
    margins: np.ndarray,
    acceptance_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    base_phi = _base_phi(specs, clean_runs)
    grid = []
    record_map: dict[str, list[dict[str, Any]]] = {}
    for index, meta in enumerate(metadata):
        records = validation_records(
            target=str(meta["target"]),
            specs=specs,
            examples=examples,
            base_phi=base_phi,
            patched_delta_phi=delta_phi[index].tolist(),
            patched_margins=margins[index].tolist(),
        )
        summary = summarize_validation(records)
        acceptance = validation_acceptance(summary, threshold=float(acceptance_threshold))
        row = {
            **dict(meta),
            "score": float(summary["score"]),
            "worst_gate": float(summary["worst_gate"]),
            "gates": summary["gates"],
            "validated": bool(acceptance["validated"]),
        }
        grid.append(row)
        record_map[str(meta["handle_id"])] = records
    return grid, record_map


def _selected_config(
    selected: Mapping[str, Any],
    *,
    sites: Sequence[ChannelSite],
) -> PatchConfiguration:
    site_index = {site.site_id: index for index, site in enumerate(sites)}
    site_ids = tuple(str(site_id) for site_id in selected["site_ids"])
    group = GroupCandidate(
        group_id=" + ".join(site_ids),
        site_indices=tuple(site_index[site_id] for site_id in site_ids),
        site_ids=site_ids,
    )
    return PatchConfiguration(
        config_id=str(selected["handle_id"]),
        group=group,
        strength=1.0,
        ranking_position=1,
        coefficients=tuple(float(value) for value in selected["coefficients"]),
    )


def _handle_patches(
    selected: Mapping[str, Any],
    *,
    site_lookup: Mapping[str, ChannelSite],
    base: Any,
    source: Any,
    restore_base: bool,
) -> tuple[PatchSpec, ...]:
    patches = []
    for site_id, coefficient in zip(selected["site_ids"], selected["coefficients"]):
        site = site_lookup[str(site_id)]
        patches.append(
            PatchSpec(
                site=site,
                source_site_id=site.site_id,
                source_position=base.final_position if restore_base else source.final_position,
                target_position=base.final_position,
                source_value=float(base.features_by_site[site.site_id] if restore_base else source.features_by_site[site.site_id]),
                strength=1.0 if restore_base else float(coefficient),
                label="restore frozen R_late handle" if restore_base else "patch calibrated handle from source",
            )
        )
    return tuple(patches)


def _mediation_test(
    model: Any,
    *,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, Any],
    sites: Sequence[ChannelSite],
    rmid: Mapping[str, Any],
    rlate: Mapping[str, Any],
    readout: FrozenDepthReadout,
    single_close_token_id: int,
    double_close_token_id: int,
    threshold: float,
) -> dict[str, Any]:
    site_lookup = {site.site_id: site for site in sites}
    records = []
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        if int(base_ex.close_count) == int(source_ex.close_count):
            continue
        base = clean_runs[spec.base_id]
        source = clean_runs[spec.source_id]
        mid_patch = _handle_patches(rmid, site_lookup=site_lookup, base=base, source=source, restore_base=False)
        late_block = _handle_patches(rlate, site_lookup=site_lookup, base=base, source=source, restore_base=True)
        late_patch = _handle_patches(rlate, site_lookup=site_lookup, base=base, source=source, restore_base=False)
        mid_margin, mid_close, _features, mid_vector = _run_patch_and_record(
            model,
            base=base,
            patches=mid_patch,
            record_sites=sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        blocked_margin, blocked_close, _features, blocked_vector = _run_patch_and_record(
            model,
            base=base,
            patches=tuple(mid_patch) + tuple(late_block),
            record_sites=sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        late_margin, late_close, _features, late_vector = _run_patch_and_record(
            model,
            base=base,
            patches=late_patch,
            record_sites=sites,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        mid_t2 = int(float(readout.predict(mid_vector)[1]) >= 0.5)
        blocked_t2 = int(float(readout.predict(blocked_vector)[1]) >= 0.5)
        late_t2 = int(float(readout.predict(late_vector)[1]) >= 0.5)
        records.append(
            {
                "base_id": spec.base_id,
                "source_id": spec.source_id,
                "direction": "one_to_two" if int(base_ex.close_count) == 1 else "two_to_one",
                "R_mid_T2_source_match": mid_t2 == int(source_ex.depth >= 2),
                "R_mid_output_source_match": int(mid_close) == int(source_ex.close_count),
                "blocked_output_base_preserve": int(blocked_close) == int(base_ex.close_count),
                "blocked_T2_source_match": blocked_t2 == int(source_ex.depth >= 2),
                "R_late_T2_base_preserve": late_t2 == int(base_ex.depth >= 2),
                "R_late_output_source_match": int(late_close) == int(source_ex.close_count),
                "margin_mediation_fraction": mediation_fraction(
                    base=float(base.margin), clean_patch=float(mid_margin), blocked_patch=float(blocked_margin)
                ),
            }
        )
    def mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
        if not rows:
            return float("nan")
        return sum(float(bool(row[key])) for row in rows) / len(rows)

    rows_by_direction = {
        direction: [row for row in records if row["direction"] == direction]
        for direction in ("one_to_two", "two_to_one")
    }
    fractions = [
        float(row["margin_mediation_fraction"])
        for row in records
        if math.isfinite(float(row["margin_mediation_fraction"]))
    ]
    metrics = {
        "R_mid_T2_source_match": mean(records, "R_mid_T2_source_match"),
        "R_mid_output_source_match": mean(records, "R_mid_output_source_match"),
        "blocked_output_base_preserve": mean(records, "blocked_output_base_preserve"),
        "blocked_output_base_preserve_one_to_two": mean(
            rows_by_direction["one_to_two"], "blocked_output_base_preserve"
        ),
        "blocked_output_base_preserve_two_to_one": mean(
            rows_by_direction["two_to_one"], "blocked_output_base_preserve"
        ),
        "blocked_T2_source_match": mean(records, "blocked_T2_source_match"),
        "R_late_T2_base_preserve": mean(records, "R_late_T2_base_preserve"),
        "R_late_output_source_match": mean(records, "R_late_output_source_match"),
        "mean_margin_mediation_fraction": sum(fractions) / len(fractions),
    }
    checks = {
        "R_mid_T2_source_match": metrics["R_mid_T2_source_match"] >= float(threshold),
        "R_mid_output_source_match": metrics["R_mid_output_source_match"] >= float(threshold),
        "blocked_output_base_preserve": metrics["blocked_output_base_preserve"] >= float(threshold),
        "blocked_output_base_preserve_one_to_two": (
            metrics["blocked_output_base_preserve_one_to_two"] >= float(threshold)
        ),
        "blocked_output_base_preserve_two_to_one": (
            metrics["blocked_output_base_preserve_two_to_one"] >= float(threshold)
        ),
        "blocked_T2_source_match": metrics["blocked_T2_source_match"] >= float(threshold),
        "R_late_T2_base_preserve": metrics["R_late_T2_base_preserve"] >= float(threshold),
        "R_late_output_source_match": metrics["R_late_output_source_match"] >= float(threshold),
        "margin_mediation_fraction": metrics["mean_margin_mediation_fraction"] >= 0.50,
    }
    return {"records": records, "metrics": metrics, "checks": checks, "validated": all(checks.values())}


def _report(path: Path, payload: Mapping[str, Any]) -> None:
    selection = payload["selection"]["best"]
    results = payload["heldout"]
    lines = [
        "# Joint R_mid / R_late PLOT",
        "",
        "## Method",
        "",
        "```text",
        "X -> R_mid -> R_late -> Y",
        "```",
        "",
        "- candidate universe: all 133 localized singleton sites",
        "- candidate groups during matching: none",
        "- coupling shape: `2 x 133`",
        "- shared signature: `[delta T2, delta P(one close), delta P(two closes)]`",
        f"- selected epsilon/beta: `{selection['epsilon']}` / `{selection['beta']}`",
        "- top-K and intervention strength calibrated separately for each coupling row",
        "- Dte was not used in fitting or calibration",
        "",
        "## Selected Handles",
        "",
        "| target | sites | K | lambda | Dcal valid | Dte valid |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        selected = selection["selected"][target]
        heldout = results[target]
        lines.append(
            f"| {target} | `{' + '.join(selected['site_ids'])}` | {selected['k']} | "
            f"{selected['strength']:.3f} | {selected['validated']} | {heldout['acceptance']['validated']} |"
        )
    for target in TARGETS:
        supports = {
            tuple(row["selected"][target]["site_ids"])
            for row in payload["selection"]["candidates"]
        }
        lines.append(
            f"\n- {target} calibrated support is identical across all epsilon/beta settings: "
            f"`{len(supports) == 1}`"
        )
    lines.extend(
        [
            "",
            "## Chain Test",
            "",
            f"- supports are distinct: `{payload['decision']['supports_distinct']}`",
            f"- downstream blocking test passes: `{payload['mediation']['validated']}`",
            f"- blocking preserves the base output overall: "
            f"`{payload['mediation']['metrics']['blocked_output_base_preserve']:.3f}`",
            f"- blocking preserves the base output for one-to-two interventions: "
            f"`{payload['mediation']['metrics']['blocked_output_base_preserve_one_to_two']:.3f}`",
            f"- blocking preserves the base output for two-to-one interventions: "
            f"`{payload['mediation']['metrics']['blocked_output_base_preserve_two_to_one']:.3f}`",
            f"- mean signed-margin mediation fraction: "
            f"`{payload['mediation']['metrics']['mean_margin_mediation_fraction']:.3f}`",
            f"- final accepted model: `{payload['decision']['accepted_model']}`",
            "",
            "## Epsilon/Beta Sweep",
            "",
            "| epsilon | beta | both Dcal-valid | joint worst gate | joint mean score |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["selection"]["candidates"]:
        lines.append(
            f"| {row['epsilon']:.3f} | {row['beta']:.3f} | {row['both_validated']} | "
            f"{row['joint_worst_gate']:.3f} | {row['joint_mean_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = not bool(args.no_resume)
    device = "cuda" if args.cuda else "cpu"
    depths = parse_depths(args.depths)
    epsilons = _float_grid(args.epsilon_grid)
    betas = _float_grid(args.beta_grid)
    k_grid = _int_grid(args.k_grid)
    strengths = _float_grid(args.strength_grid)

    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv,
        expected_node_count=int(args.expected_node_count),
    )
    sites = universe.sites
    encoding = make_tinypython_encoding(args.circuit_home)
    parent = _load_parent(
        args.parent_run_dir,
        candidate_sites=sites,
        candidate_csv_sha256=universe.csv_sha256,
        encoding=encoding,
        depths=depths,
        records_per_relation=int(args.records_per_relation),
    )
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "causal_model": "X -> R_mid -> R_late -> Y",
        "abstract_variables": list(TARGETS),
        "candidate_count": len(sites),
        "candidate_csv_sha256": universe.csv_sha256,
        "all_candidate_node_ids": [site.site_id for site in sites],
        "no_filtering_applied": True,
        "matching_resolution": "133 singleton sites only",
        "calibration_resolution": "top-K weighted handles may contain multiple singleton sites",
        "Dfit_records": len(parent["specs"]["Dfit"]),
        "signature_components": ["T2", "P_one_close", "P_two_closes"],
        "signature_semantics": "phi(y_swap_singleton)-phi(y_base)",
        "Dfit_T2_component_source": "exact stored singleton effects from the read-only parent Dfit scan",
        "Dfit_output_component_source": "new singleton patch scan on the same Dfit records",
        "coupling_shape": [2, len(sites)],
        "epsilon_grid": list(epsilons),
        "beta_grid": list(betas),
        "k_grid": list(k_grid),
        "strength_grid": list(strengths),
        "coupling_selection": "one global epsilon/beta selected on Dcal by both-valid, max-min gate, then mean score",
        "row_calibration": "top-K and strength selected separately for R_mid and R_late",
        "Dte_policy": "no Dte forward pass before selection_manifest.json",
        "fresh_content_offset": int(args.fresh_content_offset),
        "cal_contents": int(args.cal_contents),
        "test_contents": int(args.test_contents),
    }
    _atomic_json(out_dir / "run_manifest.json", manifest)
    _atomic_json(
        out_dir / "candidate_manifest.json",
        {
            "candidate_count": len(sites),
            "candidate_csv": str(args.candidate_node_csv),
            "candidate_csv_sha256": universe.csv_sha256,
            "all_candidate_node_ids": [site.site_id for site in sites],
            "no_filtering_applied": True,
            "groups_used_in_matching": False,
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
    model_info["sparse_conversion"] = [record.to_json() for record in sparse_records]
    _atomic_json(out_dir / "model_info.json", model_info)
    if args.cuda:
        torch.cuda.empty_cache()
    single_close_token_id = int(encoding.encode("]\n")[0])
    double_close_token_id = int(encoding.encode("]]\n")[0])

    dfit_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=parent["examples"],
        candidate_sites=sites,
        checkpoint_path=args.parent_run_dir / "clean_runs.jsonl",
        resume=True,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _attach_phi(parent["readout"], dfit_runs)
    dfit_base_ids = sorted({spec.base_id for spec in parent["specs"]["Dfit"]})
    dfit_prefix = build_prefix_cache_bank(
        model,
        {base_id: dfit_runs[base_id].token_ids for base_id in dfit_base_ids},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _save_prefix_cache_bank(out_dir / "Dfit_prefix_cache.pt", dfit_prefix)
    dfit_delta_phi, dfit_patched_margins = _scan_dfit_singletons(
        model,
        sites=sites,
        specs=parent["specs"]["Dfit"],
        clean_runs=dfit_runs,
        readout=parent["readout"],
        prefix_cache_bank=dfit_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        out_dir=out_dir,
        resume=resume,
    )
    parent_t2 = np.stack([parent["singleton_signatures"][site.site_id][:, 1] for site in sites])
    t2_absolute_error = np.abs(dfit_delta_phi[:, :, 1] - parent_t2)
    equivalence = {
        "canonical_T2_source": "parent Dfit singleton signatures",
        "recomputed_T2_role": "numerical audit only",
        "values_compared": int(t2_absolute_error.size),
        "max_absolute_error": float(np.max(t2_absolute_error)),
        "p99_absolute_error": float(np.quantile(t2_absolute_error, 0.99)),
        "mean_absolute_error": float(np.mean(t2_absolute_error)),
        "count_above_1e-3": int(np.count_nonzero(t2_absolute_error > 1e-3)),
        "p99_tolerance": 1e-4,
        "mean_tolerance": 1e-5,
    }
    equivalence["passed"] = bool(
        equivalence["p99_absolute_error"] <= equivalence["p99_tolerance"]
        and equivalence["mean_absolute_error"] <= equivalence["mean_tolerance"]
    )
    _atomic_json(out_dir / "signatures" / "Dfit_equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise ValueError(f"new singleton T2 effects disagree broadly with parent signatures: {equivalence}")

    abstract = abstract_joint_signatures(parent["specs"]["Dfit"], parent["examples_by_id"])
    neural = neural_joint_signatures(
        delta_t2=torch.from_numpy(parent_t2),
        patched_margins=torch.from_numpy(dfit_patched_margins),
        base_margins=_base_margins(parent["specs"]["Dfit"], dfit_runs),
    )
    if tuple(abstract.shape) != (2, len(parent["specs"]["Dfit"]) * 3):
        raise ValueError(f"unexpected abstract signature shape: {tuple(abstract.shape)}")
    if tuple(neural.shape) != (len(sites), len(parent["specs"]["Dfit"]) * 3):
        raise ValueError(f"unexpected neural signature shape: {tuple(neural.shape)}")
    sweep = _coupling_sweep(
        abstract=abstract,
        neural=neural,
        site_ids=[site.site_id for site in sites],
        epsilons=epsilons,
        betas=betas,
    )
    _atomic_json(
        out_dir / "joint_couplings.json",
        {
            "shape": [2, len(sites)],
            "abstract_signature": abstract.tolist(),
            "site_ids": [site.site_id for site in sites],
            "sweep": _coupling_payload(sweep),
        },
    )

    fresh_examples = build_unique_discovery_bank(
        encoding,
        contents=int(args.cal_contents) + int(args.test_contents),
        fit_contents=0,
        cal_contents=int(args.cal_contents),
        test_contents=int(args.test_contents),
        content_offset=int(args.fresh_content_offset),
        depths=depths,
    )
    fresh_by_id = {example.example_id: example for example in fresh_examples}
    fresh_specs = {
        split: build_transition_balanced_specs(
            fresh_examples,
            split=split,
            records_per_relation=int(args.records_per_relation),
        )
        for split in ("Dcal", "Dte")
    }
    _atomic_json(out_dir / "fresh_bank_manifest.json", _fresh_manifest(fresh_examples))
    _atomic_json(
        out_dir / "record_manifest.json",
        {split: [spec_key(spec) for spec in specs] for split, specs in fresh_specs.items()},
    )

    cal_examples = [example for example in fresh_examples if example.split == "Dcal"]
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
    _attach_phi(parent["readout"], cal_runs)
    if not all(run.correct for run in cal_runs.values()):
        raise RuntimeError("fresh Dcal clean model accuracy is below 1.0")
    cal_t2_accuracy = _clean_t2_accuracy(cal_examples, cal_runs)
    if cal_t2_accuracy < float(args.acceptance_threshold):
        raise RuntimeError(f"frozen T2 readout fails Dcal: {cal_t2_accuracy}")
    cal_base_ids = sorted({spec.base_id for spec in fresh_specs["Dcal"]})
    cal_prefix = build_prefix_cache_bank(
        model,
        {base_id: cal_runs[base_id].token_ids for base_id in cal_base_ids},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    cal_configs, cal_metadata = _calibration_configs(
        sweep=sweep,
        sites=sites,
        k_grid=k_grid,
        strength_grid=strengths,
    )
    cal_delta_phi, cal_margins = batched_patch_outputs(
        model,
        configs=cal_configs,
        specs=fresh_specs["Dcal"],
        clean_runs=cal_runs,
        sites=sites,
        readout=parent["readout"],
        prefix_cache_bank=cal_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        include_margin=True,
    )
    if cal_margins is None:
        raise AssertionError("Dcal did not return margins")
    calibration_grid, calibration_records = _evaluate_config_grid(
        metadata=cal_metadata,
        specs=fresh_specs["Dcal"],
        examples=fresh_by_id,
        clean_runs=cal_runs,
        delta_phi=cal_delta_phi,
        margins=cal_margins,
        acceptance_threshold=float(args.acceptance_threshold),
    )
    selection = select_global_coupling(calibration_grid)
    _atomic_jsonl(out_dir / "calibration" / "grid.jsonl", calibration_grid)
    _atomic_json(out_dir / "calibration" / "global_selection.json", selection)
    for target in TARGETS:
        selected = selection["best"]["selected"][target]
        _atomic_jsonl(
            out_dir / "calibration" / f"{target}_selected_records.jsonl",
            calibration_records[str(selected["handle_id"])],
        )
    _atomic_json(
        out_dir / "selection_manifest.json",
        {
            "selection_completed_before_Dte": True,
            "selection_inputs": ["Dfit joint coupling", "fresh Dcal calibration"],
            "Dte_inputs_used": [],
            "selected": selection["best"],
        },
    )

    test_examples = [example for example in fresh_examples if example.split == "Dte"]
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
    _attach_phi(parent["readout"], test_runs)
    if not all(run.correct for run in test_runs.values()):
        raise RuntimeError("fresh Dte clean model accuracy is below 1.0")
    test_t2_accuracy = _clean_t2_accuracy(test_examples, test_runs)
    if test_t2_accuracy < float(args.acceptance_threshold):
        raise RuntimeError(f"frozen T2 readout fails Dte: {test_t2_accuracy}")
    test_base_ids = sorted({spec.base_id for spec in fresh_specs["Dte"]})
    test_prefix = build_prefix_cache_bank(
        model,
        {base_id: test_runs[base_id].token_ids for base_id in test_base_ids},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    selected_rows = selection["best"]["selected"]
    test_configs = [_selected_config(selected_rows[target], sites=sites) for target in TARGETS]
    test_delta_phi, test_margins = batched_patch_outputs(
        model,
        configs=test_configs,
        specs=fresh_specs["Dte"],
        clean_runs=test_runs,
        sites=sites,
        readout=parent["readout"],
        prefix_cache_bank=test_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        include_margin=True,
    )
    if test_margins is None:
        raise AssertionError("Dte did not return margins")
    heldout = {}
    for index, target in enumerate(TARGETS):
        records = validation_records(
            target=target,
            specs=fresh_specs["Dte"],
            examples=fresh_by_id,
            base_phi=_base_phi(fresh_specs["Dte"], test_runs),
            patched_delta_phi=test_delta_phi[index].tolist(),
            patched_margins=test_margins[index].tolist(),
        )
        summary = summarize_validation(records)
        acceptance = validation_acceptance(summary, threshold=float(args.acceptance_threshold))
        heldout[target] = {"summary": summary, "acceptance": acceptance}
        _atomic_jsonl(out_dir / "heldout" / f"{target}_records.jsonl", records)

    mediation = _mediation_test(
        model,
        specs=fresh_specs["Dte"],
        examples=fresh_by_id,
        clean_runs=test_runs,
        sites=sites,
        rmid=selected_rows["R_mid"],
        rlate=selected_rows["R_late"],
        readout=parent["readout"],
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        threshold=float(args.acceptance_threshold),
    )
    _atomic_jsonl(out_dir / "mediation" / "records.jsonl", mediation["records"])
    _atomic_json(
        out_dir / "mediation" / "summary.json",
        {key: value for key, value in mediation.items() if key != "records"},
    )
    supports_distinct = set(selected_rows["R_mid"]["site_ids"]) != set(selected_rows["R_late"]["site_ids"])
    full_pass = (
        all(bool(heldout[target]["acceptance"]["validated"]) for target in TARGETS)
        and supports_distinct
        and bool(mediation["validated"])
    )
    decision = {
        "supports_distinct": supports_distinct,
        "accepted_model": "X -> R_mid -> R_late -> Y" if full_pass else "X -> R_late -> Y",
        "full_model_validated": full_pass,
    }
    result = {
        "run_manifest_sha256": canonical_sha256(manifest),
        "clean": {
            "Dcal_accuracy": 1.0,
            "Dte_accuracy": 1.0,
            "Dcal_T2_readout_accuracy": cal_t2_accuracy,
            "Dte_T2_readout_accuracy": test_t2_accuracy,
        },
        "selection": selection,
        "heldout": heldout,
        "mediation": {key: value for key, value in mediation.items() if key != "records"},
        "decision": decision,
    }
    _atomic_json(out_dir / "joint_rmid_rlate_plot.json", result)
    _report(out_dir / "joint_rmid_rlate_plot.md", result)
    print(json.dumps({"status": "complete", **decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
