from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .activation import ChannelSite
from .bracket_d_large_bank_frozen_readout import spec_key
from .bracket_d_rich_signatures import FrozenDepthReadout, load_full_localized_candidate_universe
from .bracket_group_depth_plot import GroupCandidate, build_unique_discovery_bank
from .bracket_joint_rmid_rlate import (
    summarize_validation,
    validation_acceptance,
    validation_records,
)
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .bracket_rmid_branching import (
    build_content_transition_balanced_specs,
    build_rlate_calibration_candidates,
    direction,
    select_mediation_aware_candidate,
    signed_controlled_direct_fraction,
    summarize_blocking_records,
    summarize_factorial_records,
)
from .run_bracket_d_large_bank_frozen_readout_plot import _atomic_json, _atomic_jsonl, _collect_or_resume_clean_runs
from .run_bracket_group_depth_plot import (
    PatchConfiguration,
    _attach_phi,
    _hook_layout,
    _save_prefix_cache_bank,
    batched_patch_outputs,
)
from .run_bracket_joint_rmid_rlate_plot import _base_margins, _base_phi, _clean_t2_accuracy, _fresh_manifest
from .run_bracket_threshold_component_plot import _load_frozen_readout
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import (
    PrefixKVCache,
    bracket_incremental_final_forward,
    build_prefix_cache_bank,
    convert_transformer_linears_to_sparse,
)


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_PRIOR_DIR = Path("eval/openai_sparse_plot/bracket_joint_rmid_rlate_20260713")
DEFAULT_PARENT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_rmid_branching_factorial_20260713")


@dataclass(frozen=True)
class CrossedConfiguration:
    config_id: str
    mode: str
    mid_site_indices: tuple[int, ...]
    mid_coefficients: tuple[float, ...]
    late_site_indices: tuple[int, ...]
    late_coefficients: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test pure-chain versus R_mid-to-output branching.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--prior-run-dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--fresh-content-offset", type=int, default=3000)
    parser.add_argument("--cal-contents", type=int, default=24)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
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


def _float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("float grid values must be positive")
    return values


def _int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("integer grid values must be positive")
    return values


def _group_config(candidate: Mapping[str, Any], sites: Sequence[ChannelSite]) -> PatchConfiguration:
    lookup = {site.site_id: index for index, site in enumerate(sites)}
    site_ids = tuple(str(value) for value in candidate["site_ids"])
    config_id = candidate.get("config_id", candidate.get("handle_id"))
    if config_id is None:
        raise ValueError("handle is missing both config_id and handle_id")
    return PatchConfiguration(
        config_id=str(config_id),
        group=GroupCandidate(
            group_id=" + ".join(site_ids),
            site_indices=tuple(lookup[site_id] for site_id in site_ids),
            site_ids=site_ids,
        ),
        strength=1.0,
        ranking_position=1,
        coefficients=tuple(float(value) for value in candidate["coefficients"]),
    )


def _crossed_config(
    *,
    config_id: str,
    mode: str,
    rmid: Mapping[str, Any],
    rlate: Mapping[str, Any],
    sites: Sequence[ChannelSite],
) -> CrossedConfiguration:
    lookup = {site.site_id: index for index, site in enumerate(sites)}
    mid_ids = tuple(str(value) for value in rmid["site_ids"])
    late_ids = tuple(str(value) for value in rlate["site_ids"])
    if set(mid_ids).intersection(late_ids):
        raise ValueError("crossed interventions require distinct R_mid and R_late supports")
    return CrossedConfiguration(
        config_id=config_id,
        mode=mode,
        mid_site_indices=tuple(lookup[value] for value in mid_ids),
        mid_coefficients=tuple(float(value) for value in rmid["coefficients"]),
        late_site_indices=tuple(lookup[value] for value in late_ids),
        late_coefficients=tuple(float(value) for value in rlate["coefficients"]),
    )


def _operation_rows(
    configs: Sequence[CrossedConfiguration],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    operations: list[list[tuple[int, float, bool]]] = []
    for config in configs:
        row: list[tuple[int, float, bool]] = []
        if config.mode in {"mid_source_late_base", "mid_source_late_source"}:
            row.extend(
                (site, coefficient, True)
                for site, coefficient in zip(config.mid_site_indices, config.mid_coefficients)
            )
        if config.mode == "mid_source_late_base":
            row.extend((site, 1.0, False) for site in config.late_site_indices)
        elif config.mode == "mid_source_late_source":
            row.extend(
                (site, coefficient, True)
                for site, coefficient in zip(config.late_site_indices, config.late_coefficients)
            )
        else:
            raise ValueError(f"unknown crossed intervention mode: {config.mode}")
        operations.append(row)
    width = max(len(row) for row in operations)
    members = torch.full((len(configs), width), -1, dtype=torch.long, device=device)
    coefficients = torch.zeros((len(configs), width), dtype=torch.float32, device=device)
    use_source = torch.zeros((len(configs), width), dtype=torch.bool, device=device)
    for index, row in enumerate(operations):
        for column, (site, coefficient, source_flag) in enumerate(row):
            members[index, column] = int(site)
            coefficients[index, column] = float(coefficient)
            use_source[index, column] = bool(source_flag)
    return members, coefficients, use_source


def batched_crossed_outputs(
    model: Any,
    *,
    configs: Sequence[CrossedConfiguration],
    specs: Sequence[MultiDepthResamplingSpec],
    clean_runs: Mapping[str, Any],
    sites: Sequence[ChannelSite],
    readout: FrozenDepthReadout,
    prefix_cache_bank: Mapping[str, PrefixKVCache],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
    max_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not configs or not specs:
        raise ValueError("configs and specs must be nonempty")
    output_phi = np.empty((len(configs), len(specs), 4), dtype=np.float32)
    output_margin = np.empty((len(configs), len(specs)), dtype=np.float32)
    source_features = torch.tensor(
        [clean_runs[spec.source_id].feature_vector for spec in specs], dtype=torch.float32, device=device
    )
    base_features = torch.tensor(
        [clean_runs[spec.base_id].feature_vector for spec in specs], dtype=torch.float32, device=device
    )
    base_phi = torch.tensor(
        [clean_runs[spec.base_id].depth_phi for spec in specs], dtype=torch.float32, device=device
    )
    members, coefficients, use_source = _operation_rows(configs, device=device)
    site_channels = torch.tensor([site.channel for site in sites], dtype=torch.long, device=device)
    hook_names = tuple(sorted({site.hook_key for site in sites}))
    hook_code = {hook: index for index, hook in enumerate(hook_names)}
    site_hook_codes = torch.tensor([hook_code[site.hook_key] for site in sites], dtype=torch.long, device=device)
    layout = _hook_layout(sites)
    readout_weights = readout.weights.to(device=device, dtype=torch.float32)
    by_base: dict[str, list[int]] = {}
    for record_index, spec in enumerate(specs):
        by_base.setdefault(spec.base_id, []).append(record_index)

    for base_id, record_indices_list in sorted(by_base.items()):
        record_indices = torch.tensor(record_indices_list, dtype=torch.long, device=device)
        bucket_records = len(record_indices_list)
        task_count = len(configs) * bucket_records
        for task_start in range(0, task_count, int(max_batch_size)):
            task_stop = min(task_count, task_start + int(max_batch_size))
            flat = torch.arange(task_start, task_stop, dtype=torch.long, device=device)
            config_indices = torch.div(flat, bucket_records, rounding_mode="floor")
            bucket_indices = flat.remainder(bucket_records)
            global_records = record_indices[bucket_indices]
            batch_size = int(flat.shape[0])
            selected_members = members[config_indices]
            valid = selected_members >= 0
            rows = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(selected_members)
            patch_rows = rows[valid]
            patch_sites = selected_members[valid]
            patch_records = global_records.unsqueeze(1).expand_as(selected_members)[valid]
            source_flags = use_source[config_indices][valid]
            desired = torch.where(
                source_flags,
                source_features[patch_records, patch_sites],
                base_features[patch_records, patch_sites],
            )
            patch_coefficients = coefficients[config_indices][valid]
            patch_hook_codes = site_hook_codes[patch_sites]
            patch_channels = site_channels[patch_sites]
            feature_matrix = torch.empty((batch_size, len(sites)), dtype=torch.float32, device=device)
            seen_hooks: set[str] = set()
            interventions: dict[str, Any] = {}
            for hook_name in hook_names:
                code = hook_code[hook_name]
                mask = patch_hook_codes == code
                hook_rows = patch_rows[mask]
                hook_channels = patch_channels[mask]
                hook_desired = desired[mask]
                hook_coefficients = patch_coefficients[mask]
                capture_indices = layout[hook_name]["site_indices"].to(device=device)
                capture_channels = layout[hook_name]["channels"].to(device=device)

                def _intervene(
                    tensor: torch.Tensor,
                    *,
                    hook_name: str = hook_name,
                    hook_rows: torch.Tensor = hook_rows,
                    hook_channels: torch.Tensor = hook_channels,
                    hook_desired: torch.Tensor = hook_desired,
                    hook_coefficients: torch.Tensor = hook_coefficients,
                    capture_indices: torch.Tensor = capture_indices,
                    capture_channels: torch.Tensor = capture_channels,
                ) -> torch.Tensor:
                    seen_hooks.add(hook_name)
                    if hook_rows.numel():
                        tensor = tensor.clone()
                        current = tensor[hook_rows, -1, hook_channels]
                        target = hook_desired.to(device=tensor.device, dtype=tensor.dtype)
                        alpha = hook_coefficients.to(device=tensor.device, dtype=tensor.dtype)
                        tensor[hook_rows, -1, hook_channels] = current + alpha * (target - current)
                    feature_matrix[:, capture_indices] = tensor[:, -1, capture_channels].to(torch.float32)
                    return tensor

                interventions[hook_name] = _intervene
            with torch.no_grad():
                with hook_recorder(regex="^$", interventions=interventions):
                    margins = bracket_incremental_final_forward(
                        model,
                        prefix_cache_bank[base_id],
                        batch_size=batch_size,
                        single_close_token_id=single_close_token_id,
                        double_close_token_id=double_close_token_id,
                        include_margin=True,
                    )
            if set(hook_names) - seen_hooks:
                raise RuntimeError("not all localized hooks were observed")
            if margins is None:
                raise AssertionError("crossed forward did not return margins")
            delta_phi = feature_matrix @ readout_weights - base_phi[global_records]
            config_cpu = config_indices.detach().cpu().numpy()
            record_cpu = global_records.detach().cpu().numpy()
            output_phi[config_cpu, record_cpu] = delta_phi.detach().cpu().numpy()
            output_margin[config_cpu, record_cpu] = margins.detach().to(torch.float32).cpu().numpy()
    return output_phi, output_margin


def _close_from_margin(value: float) -> int:
    return 2 if float(value) > 0.0 else 1


def _blocking_records(
    *,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, Any],
    mid_phi: np.ndarray,
    mid_margins: np.ndarray,
    late_phi: np.ndarray,
    late_margins: np.ndarray,
    blocked_phi: np.ndarray,
    blocked_margins: np.ndarray,
) -> list[dict[str, Any]]:
    records = []
    for index, spec in enumerate(specs):
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        if int(base.close_count) == int(source.close_count):
            continue
        base_run = clean_runs[spec.base_id]
        base_t2_value = float(base_run.depth_phi[1])
        mid_t2 = int(base_t2_value + float(mid_phi[index, 1]) >= 0.5)
        late_t2 = int(base_t2_value + float(late_phi[index, 1]) >= 0.5)
        blocked_t2 = int(base_t2_value + float(blocked_phi[index, 1]) >= 0.5)
        mid_close = _close_from_margin(float(mid_margins[index]))
        late_close = _close_from_margin(float(late_margins[index]))
        blocked_close = _close_from_margin(float(blocked_margins[index]))
        records.append(
            {
                "base_id": spec.base_id,
                "source_id": spec.source_id,
                "base_content": str(base.numeric_content),
                "source_content": str(source.numeric_content),
                "direction": direction(base.close_count, source.close_count),
                "mid_output_source": mid_close == int(source.close_count),
                "mid_T2_source": mid_t2 == int(source.depth >= 2),
                "late_output_source": late_close == int(source.close_count),
                "late_T2_base": late_t2 == int(base.depth >= 2),
                "blocked_output_base": blocked_close == int(base.close_count),
                "blocked_output_source": blocked_close == int(source.close_count),
                "blocked_T2_source": blocked_t2 == int(source.depth >= 2),
                "CDE_fraction": signed_controlled_direct_fraction(
                    base=float(base_run.margin),
                    mid=float(mid_margins[index]),
                    blocked=float(blocked_margins[index]),
                ),
            }
        )
    return records


def _factorial_records(
    *,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, Any],
    mid_phi: np.ndarray,
    mid_margins: np.ndarray,
    late_phi: np.ndarray,
    late_margins: np.ndarray,
    blocked_phi: np.ndarray,
    blocked_margins: np.ndarray,
    both_phi: np.ndarray,
    both_margins: np.ndarray,
) -> list[dict[str, Any]]:
    records = _blocking_records(
        specs=specs,
        examples=examples,
        clean_runs=clean_runs,
        mid_phi=mid_phi,
        mid_margins=mid_margins,
        late_phi=late_phi,
        late_margins=late_margins,
        blocked_phi=blocked_phi,
        blocked_margins=blocked_margins,
    )
    differing_indices = [
        index
        for index, spec in enumerate(specs)
        if int(examples[spec.base_id].close_count) != int(examples[spec.source_id].close_count)
    ]
    if len(records) != len(differing_indices):
        raise AssertionError("factorial record alignment failed")
    for row, index in zip(records, differing_indices):
        spec = specs[index]
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        base_t2_value = float(clean_runs[spec.base_id].depth_phi[1])
        row["both_source_output_source"] = _close_from_margin(float(both_margins[index])) == int(source.close_count)
        row["both_source_T2_source"] = int(base_t2_value + float(both_phi[index, 1]) >= 0.5) == int(
            source.depth >= 2
        )
    return records


def _report(path: Path, payload: Mapping[str, Any]) -> None:
    selected = payload["selection"]
    factorial = payload["factorial"]
    metrics = factorial["metrics"]
    intervals = factorial["CDE_bootstrap_95"]
    lines = [
        "# R_mid Branching Factorial Test",
        "",
        "## Models",
        "",
        "```text",
        "Pure chain: X -> R_mid -> R_late -> Y",
        "Branching:  X -> R_mid -> R_late -> Y and R_mid -> Y",
        "```",
        "",
        "## Frozen And Calibrated Handles",
        "",
        f"- frozen R_mid: `{' + '.join(payload['rmid']['site_ids'])}`",
        f"- selected R_late: `{' + '.join(selected['site_ids'])}`",
        f"- epsilon / beta: `{selected['epsilon']}` / `{selected['beta']}`",
        f"- K / lambda: `{selected['k']}` / `{selected['strength']}`",
        f"- selected on Dcal before Dte: `{payload['selection_completed_before_Dte']}`",
        f"- bootstrap unit: `{factorial['bootstrap_unit']}`",
        "",
        "## Fresh Heldout Factorial Test",
        "",
        "| direction | R_mid alone -> source | R_late alone -> source | R_late preserves base T2 | both source -> source | R_mid source + R_late base -> base | mean direct fraction | 95% bootstrap interval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for value in ("one_to_two", "two_to_one"):
        interval = intervals[value]
        lines.append(
            f"| {value} | {metrics[f'mid_output_source_{value}']:.3f} | "
            f"{metrics[f'late_output_source_{value}']:.3f} | "
            f"{metrics[f'late_T2_base_{value}']:.3f} | "
            f"{metrics[f'both_source_output_source_{value}']:.3f} | "
            f"{metrics[f'blocked_output_base_{value}']:.3f} | "
            f"{metrics[f'mean_CDE_fraction_{value}']:.3f} | "
            f"[{interval['low']:.3f}, {interval['high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- pure chain validated: `{factorial['pure_chain_validated']}`",
            f"- symmetric branching model validated: `{factorial['symmetric_branch_validated']}`",
            f"- result: `{factorial['conclusion']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = not bool(args.no_resume)
    device = "cuda" if args.cuda else "cpu"
    depths = parse_depths(args.depths)
    k_grid = _int_grid(args.k_grid)
    strength_grid = _float_grid(args.strength_grid)
    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv, expected_node_count=int(args.expected_node_count)
    )
    sites = universe.sites
    site_ids = [site.site_id for site in sites]
    prior_candidate = json.loads((args.prior_run_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    if int(prior_candidate["candidate_count"]) != len(sites) or not prior_candidate["no_filtering_applied"]:
        raise ValueError("prior run was not the full unfiltered all-133 experiment")
    if list(prior_candidate["all_candidate_node_ids"]) != site_ids:
        raise ValueError("prior candidate order differs")
    if prior_candidate["candidate_csv_sha256"] != universe.csv_sha256:
        raise ValueError("prior candidate CSV hash differs")
    prior_result = json.loads((args.prior_run_dir / "joint_rmid_rlate_plot.json").read_text(encoding="utf-8"))
    coupling_payload = json.loads((args.prior_run_dir / "joint_couplings.json").read_text(encoding="utf-8"))
    rmid = dict(prior_result["selection"]["best"]["selected"]["R_mid"])
    candidates = build_rlate_calibration_candidates(
        coupling_payload,
        k_grid=k_grid,
        strength_grid=strength_grid,
        rmid_site_ids=rmid["site_ids"],
    )
    manifest = {
        "schema_version": 1,
        "models": ["X -> R_mid -> R_late -> Y", "X -> R_mid -> {R_late,Y}; R_late -> Y"],
        "candidate_count": len(sites),
        "candidate_csv_sha256": universe.csv_sha256,
        "no_filtering_applied_in_matching": True,
        "matching_source": "read-only 2x133 singleton coupling from prior joint run",
        "pair_or_triple_candidates": False,
        "rmid_frozen_before_calibration": rmid,
        "rlate_calibration_candidate_count": len(candidates),
        "epsilon_values": sorted({float(row["epsilon"]) for row in candidates}),
        "beta_values": sorted({float(row["beta"]) for row in candidates}),
        "k_grid": list(k_grid),
        "strength_grid": list(strength_grid),
        "fresh_content_offset": int(args.fresh_content_offset),
        "record_balancing": "deterministic balance over ordered depth transition and base numeric content",
        "bootstrap_unit": "base numeric content",
        "Dte_policy": "no Dte clean run or intervention before selection_manifest.json",
    }
    _atomic_json(out_dir / "run_manifest.json", manifest)
    _atomic_json(out_dir / "rlate_candidate_handles.json", candidates)

    encoding = make_tinypython_encoding(args.circuit_home)
    readout = _load_frozen_readout(args.parent_run_dir / "frozen_depth_readout.json", site_ids)
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
        split: build_content_transition_balanced_specs(
            fresh_examples, split=split, records_per_relation=int(args.records_per_relation)
        )
        for split in ("Dcal", "Dte")
    }
    _atomic_json(out_dir / "fresh_bank_manifest.json", _fresh_manifest(fresh_examples))
    _atomic_json(
        out_dir / "record_manifest.json",
        {split: [spec_key(spec) for spec in rows] for split, rows in fresh_specs.items()},
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
    _attach_phi(readout, cal_runs)
    if not all(row.correct for row in cal_runs.values()):
        raise RuntimeError("fresh Dcal clean accuracy is below 1.0")
    if _clean_t2_accuracy(cal_examples, cal_runs) < float(args.acceptance_threshold):
        raise RuntimeError("frozen T2 readout fails fresh Dcal")
    cal_prefix = build_prefix_cache_bank(
        model,
        {base_id: cal_runs[base_id].token_ids for base_id in sorted({row.base_id for row in fresh_specs["Dcal"]})},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _save_prefix_cache_bank(out_dir / "Dcal_prefix_cache.pt", cal_prefix)
    mid_config = _group_config(rmid, sites)
    mid_phi, mid_margins = batched_patch_outputs(
        model,
        configs=[mid_config],
        specs=fresh_specs["Dcal"],
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
    direct_configs = [_group_config(row, sites) for row in candidates]
    late_phi, late_margins = batched_patch_outputs(
        model,
        configs=direct_configs,
        specs=fresh_specs["Dcal"],
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
    if mid_margins is None or late_margins is None:
        raise AssertionError("calibration interventions did not return margins")
    eligible_indices = [index for index, row in enumerate(candidates) if row["chain_eligible"]]
    crossed_configs = [
        _crossed_config(
            config_id=f"block:{candidates[index]['config_id']}",
            mode="mid_source_late_base",
            rmid=rmid,
            rlate=candidates[index],
            sites=sites,
        )
        for index in eligible_indices
    ]
    blocked_phi, blocked_margins = batched_crossed_outputs(
        model,
        configs=crossed_configs,
        specs=fresh_specs["Dcal"],
        clean_runs=cal_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=cal_prefix,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
    )
    blocked_by_candidate = {candidate_index: row for row, candidate_index in enumerate(eligible_indices)}
    cal_grid = []
    cal_records_by_id: dict[str, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        direct_records = validation_records(
            target="R_late",
            specs=fresh_specs["Dcal"],
            examples=fresh_by_id,
            base_phi=_base_phi(fresh_specs["Dcal"], cal_runs),
            patched_delta_phi=late_phi[index].tolist(),
            patched_margins=late_margins[index].tolist(),
        )
        direct_summary = summarize_validation(direct_records)
        direct_acceptance = validation_acceptance(direct_summary, threshold=float(args.acceptance_threshold))
        if index in blocked_by_candidate:
            block_index = blocked_by_candidate[index]
            block_records = _blocking_records(
                specs=fresh_specs["Dcal"],
                examples=fresh_by_id,
                clean_runs=cal_runs,
                mid_phi=mid_phi[0],
                mid_margins=mid_margins[0],
                late_phi=late_phi[index],
                late_margins=late_margins[index],
                blocked_phi=blocked_phi[block_index],
                blocked_margins=blocked_margins[block_index],
            )
            block_summary = summarize_blocking_records(
                block_records, threshold=float(args.acceptance_threshold)
            )
        else:
            block_records = []
            block_summary = {"validated": False, "score": 0.0, "worst_gate": 0.0, "metrics": {}}
        row = {
            **candidate,
            "direct_validated": bool(direct_acceptance["validated"]),
            "blocking_validated": bool(block_summary["validated"]),
            "direct_summary": direct_summary,
            "blocking_summary": block_summary,
            "score": (float(direct_summary["score"]) + float(block_summary["score"])) / 2.0,
            "worst_gate": min(float(direct_summary["worst_gate"]), float(block_summary["worst_gate"])),
        }
        cal_grid.append(row)
        cal_records_by_id[str(candidate["config_id"])] = block_records
    selected = dict(select_mediation_aware_candidate(cal_grid))
    _atomic_jsonl(out_dir / "calibration" / "grid.jsonl", cal_grid)
    _atomic_jsonl(
        out_dir / "calibration" / "selected_blocking_records.jsonl",
        cal_records_by_id[str(selected["config_id"])],
    )
    selection_manifest = {
        "selection_completed_before_Dte": True,
        "Dte_inputs_used": [],
        "selection_inputs": ["prior all-133 singleton coupling", "fresh Dcal direct interventions", "fresh Dcal blocking"],
        "selected": selected,
    }
    _atomic_json(out_dir / "selection_manifest.json", selection_manifest)

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
    _attach_phi(readout, test_runs)
    if not all(row.correct for row in test_runs.values()):
        raise RuntimeError("fresh Dte clean accuracy is below 1.0")
    if _clean_t2_accuracy(test_examples, test_runs) < float(args.acceptance_threshold):
        raise RuntimeError("frozen T2 readout fails fresh Dte")
    test_prefix = build_prefix_cache_bank(
        model,
        {base_id: test_runs[base_id].token_ids for base_id in sorted({row.base_id for row in fresh_specs["Dte"]})},
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    selected_late_config = _group_config(selected, sites)
    direct_test_phi, direct_test_margins = batched_patch_outputs(
        model,
        configs=[mid_config, selected_late_config],
        specs=fresh_specs["Dte"],
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
    if direct_test_margins is None:
        raise AssertionError("Dte direct interventions did not return margins")
    crossed_test_configs = [
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
    ]
    crossed_test_phi, crossed_test_margins = batched_crossed_outputs(
        model,
        configs=crossed_test_configs,
        specs=fresh_specs["Dte"],
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
        specs=fresh_specs["Dte"],
        examples=fresh_by_id,
        clean_runs=test_runs,
        mid_phi=direct_test_phi[0],
        mid_margins=direct_test_margins[0],
        late_phi=direct_test_phi[1],
        late_margins=direct_test_margins[1],
        blocked_phi=crossed_test_phi[0],
        blocked_margins=crossed_test_margins[0],
        both_phi=crossed_test_phi[1],
        both_margins=crossed_test_margins[1],
    )
    factorial = summarize_factorial_records(
        factorial_records,
        threshold=float(args.acceptance_threshold),
        direct_fraction_tolerance=float(args.direct_fraction_tolerance),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    _atomic_jsonl(out_dir / "heldout" / "factorial_records.jsonl", factorial_records)
    result = {
        "rmid": rmid,
        "selection": selected,
        "selection_completed_before_Dte": True,
        "clean": {"Dcal_accuracy": 1.0, "Dte_accuracy": 1.0},
        "factorial": factorial,
    }
    _atomic_json(out_dir / "bracket_rmid_branching_factorial.json", result)
    _report(out_dir / "bracket_rmid_branching_factorial.md", result)
    print(json.dumps({"status": "complete", "conclusion": factorial["conclusion"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
