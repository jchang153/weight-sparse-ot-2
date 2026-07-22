from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .activation import ChannelSite
from .bracket_d_large_bank_frozen_readout import (
    build_transition_balanced_specs,
    calibration_score as depth_calibration_score,
    canonical_sha256,
    depth_acceptance,
    depth_from_phi,
    spec_key,
    summarize_validation_records,
)
from .bracket_d_rich_signatures import FrozenDepthReadout, load_full_localized_candidate_universe
from .bracket_group_depth_plot import (
    GroupCandidate,
    abstract_delta_matrix,
    build_single_pair_groups,
    select_pass_first,
    target_signature,
)
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .bracket_progressive_model_discovery import build_discovery_bank
from .bracket_threshold_components import (
    TARGET_COMPONENTS,
    component_acceptance,
    component_calibration_score,
    expected_component_record,
    individual_spec_eligible,
    summarize_component_records,
)
from .plot_matching import sinkhorn_one_sided_uot
from .run_bracket_d_large_bank_frozen_readout_plot import (
    _atomic_json,
    _collect_or_resume_clean_runs,
    _load_jsonl,
)
from .run_bracket_d_rich_signature_experiments import CleanRun
from .run_bracket_threshold_component_plot import _load_frozen_readout
from .runtime import load_sparse_gpt_model, make_tinypython_encoding
from .sparse_inference_runtime import (
    PrefixKVCache,
    bracket_incremental_final_forward,
    bracket_forward_without_full_logits,
    build_prefix_cache_bank,
    convert_transformer_linears_to_sparse,
)


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_PARENT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_group_depth_plot_20260711")
TARGETS = ("D", "T2", "T3", "T4")


@dataclass(frozen=True)
class PatchConfiguration:
    config_id: str
    group: GroupCandidate
    strength: float
    ranking_position: int
    coefficients: tuple[float, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical full-swap singleton/pair PLOT for bracket D and T2/T3/T4."
    )
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--calibrate-top-groups", type=int, default=32)
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,3.0,4.0,8.0")
    parser.add_argument("--acceptance-threshold", type=float, default=0.90)
    parser.add_argument("--group-chunk-size", type=int, default=128)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--signature-start", type=int, default=0)
    parser.add_argument("--signature-stop", type=int, default=None)
    parser.add_argument("--equivalence-base-limit", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--dense-kernels", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--signatures-only", action="store_true")
    return parser.parse_args()


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("empty strength grid")
    return values


def _load_parent(
    parent_dir: Path,
    *,
    candidate_sites: Sequence[ChannelSite],
    candidate_csv_sha256: str,
    encoding: Any,
    depths: Sequence[int],
    records_per_relation: int,
) -> dict[str, Any]:
    candidate_manifest = json.loads((parent_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    bank = json.loads((parent_dir / "bank_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_ids = [site.site_id for site in candidate_sites]
    if int(candidate_manifest.get("candidate_count", -1)) != len(candidate_sites):
        raise ValueError("parent candidate count differs from current universe")
    if not bool(candidate_manifest.get("no_filtering_applied")):
        raise ValueError("parent run did not certify no filtering")
    if candidate_manifest.get("candidate_csv_sha256") != candidate_csv_sha256:
        raise ValueError("parent candidate CSV hash differs")
    if list(candidate_manifest.get("all_candidate_node_ids", ())) != expected_ids:
        raise ValueError("parent candidate IDs/order differ")

    split_payload = bank["splits"]
    fit_contents = int(split_payload["Dfit"]["content_count"])
    cal_contents = int(split_payload["Dcal"]["content_count"])
    test_contents = int(split_payload["Dte"]["content_count"])
    examples = build_discovery_bank(
        encoding,
        contents=fit_contents + cal_contents + test_contents,
        fit_contents=fit_contents,
        cal_contents=cal_contents,
        test_contents=test_contents,
        depths=depths,
    )
    examples_by_id = {example.example_id: example for example in examples}
    specs = {
        split: build_transition_balanced_specs(
            examples,
            split=split,
            records_per_relation=int(records_per_relation),
        )
        for split in ("Dfit", "Dcal", "Dte")
    }
    expected_keys = run_manifest.get("record_keys", {})
    for split in specs:
        if expected_keys and [spec_key(spec) for spec in specs[split]] != list(expected_keys[split]):
            raise ValueError(f"reconstructed {split} records differ from parent manifest")
    if len(specs["Dfit"]) != 600:
        raise ValueError(f"expected 600 Dfit records, got {len(specs['Dfit'])}")

    readout = _load_frozen_readout(parent_dir / "frozen_depth_readout.json", expected_ids)
    singleton_rows = _load_jsonl(parent_dir / "signatures" / "C_D_frozen_readout.jsonl")
    singleton_signatures = {
        str(row["site_id"]): np.asarray(row["signature"], dtype=np.float32).reshape(len(specs["Dfit"]), 4)
        for row in singleton_rows
    }
    if set(singleton_signatures) != set(expected_ids):
        raise ValueError("parent singleton signatures do not cover all 133 sites")
    return {
        "candidate_manifest": candidate_manifest,
        "bank_manifest": bank,
        "run_manifest": run_manifest,
        "examples": examples,
        "examples_by_id": examples_by_id,
        "specs": specs,
        "readout": readout,
        "singleton_signatures": singleton_signatures,
    }


def _attach_phi(readout: FrozenDepthReadout, runs: Mapping[str, CleanRun]) -> None:
    for run in runs.values():
        run.depth_phi = tuple(readout.predict(run.feature_vector))


def _save_prefix_cache_bank(path: Path, bank: Mapping[str, PrefixKVCache]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        base_id: {
            "token_ids": cache.token_ids.detach().cpu(),
            "queries": [tensor.detach().cpu() for tensor in cache.queries],
            "keys": [tensor.detach().cpu() for tensor in cache.keys],
            "values": [tensor.detach().cpu() for tensor in cache.values],
        }
        for base_id, cache in bank.items()
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_prefix_cache_bank(path: Path, *, device: str) -> dict[str, PrefixKVCache]:
    payload = torch.load(path, map_location=device, weights_only=True)
    return {
        str(base_id): PrefixKVCache(
            token_ids=row["token_ids"].to(device=device),
            queries=tuple(tensor.to(device=device) for tensor in row["queries"]),
            keys=tuple(tensor.to(device=device) for tensor in row["keys"]),
            values=tuple(tensor.to(device=device) for tensor in row["values"]),
        )
        for base_id, row in payload.items()
    }


def _hook_layout(sites: Sequence[ChannelSite]) -> dict[str, dict[str, torch.Tensor]]:
    by_hook: dict[str, list[tuple[int, ChannelSite]]] = {}
    for site_index, site in enumerate(sites):
        by_hook.setdefault(site.hook_key, []).append((site_index, site))
    return {
        hook: {
            "site_indices": torch.tensor([index for index, _site in rows], dtype=torch.long),
            "channels": torch.tensor([site.channel for _index, site in rows], dtype=torch.long),
        }
        for hook, rows in by_hook.items()
    }


def _verify_incremental_execution(
    model: Any,
    *,
    base_ids: Sequence[str],
    clean_runs: Mapping[str, CleanRun],
    sites: Sequence[ChannelSite],
    prefix_cache_bank: Mapping[str, PrefixKVCache],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    hooks = sorted({site.hook_key for site in sites})
    regex = "^(?:" + "|".join(re.escape(hook) for hook in hooks) + ")$"
    rows = []
    for base_id in base_ids:
        with torch.no_grad():
            with hook_recorder(regex=regex) as full_context:
                full_margin = bracket_forward_without_full_logits(
                    model,
                    clean_runs[base_id].token_ids,
                    single_close_token_id=single_close_token_id,
                    double_close_token_id=double_close_token_id,
                    include_margin=True,
                )
            with hook_recorder(regex=regex) as incremental_context:
                incremental_margin = bracket_incremental_final_forward(
                    model,
                    prefix_cache_bank[base_id],
                    batch_size=1,
                    single_close_token_id=single_close_token_id,
                    double_close_token_id=double_close_token_id,
                    include_margin=True,
                )
        feature_errors = [
            abs(
                float(full_context[site.hook_key][0, -1, site.channel])
                - float(incremental_context[site.hook_key][0, -1, site.channel])
            )
            for site in sites
        ]
        if full_margin is None or incremental_margin is None:
            raise AssertionError("execution verification did not produce margins")
        rows.append(
            {
                "base_id": base_id,
                "max_candidate_feature_error": max(feature_errors),
                "mean_candidate_feature_error": sum(feature_errors) / len(feature_errors),
                "margin_error": abs(float(full_margin[0]) - float(incremental_margin[0])),
            }
        )
    return {
        "rows": rows,
        "max_candidate_feature_error": max(float(row["max_candidate_feature_error"]) for row in rows),
        "max_margin_error": max(float(row["margin_error"]) for row in rows),
    }


def _member_matrix(configs: Sequence[PatchConfiguration], *, device: str) -> torch.Tensor:
    width = max(config.group.group_size for config in configs)
    matrix = torch.full((len(configs), width), -1, dtype=torch.long, device=device)
    for row, config in enumerate(configs):
        matrix[row, : config.group.group_size] = torch.tensor(
            config.group.site_indices,
            dtype=torch.long,
            device=device,
        )
    return matrix


def _coefficient_matrix(configs: Sequence[PatchConfiguration], *, device: str) -> torch.Tensor:
    width = max(config.group.group_size for config in configs)
    matrix = torch.zeros((len(configs), width), dtype=torch.float32, device=device)
    for row, config in enumerate(configs):
        coefficients = (
            tuple(float(config.strength) for _ in config.group.site_ids)
            if config.coefficients is None
            else tuple(float(value) for value in config.coefficients)
        )
        if len(coefficients) != config.group.group_size:
            raise ValueError(f"coefficient count differs from group size for {config.config_id}")
        matrix[row, : config.group.group_size] = torch.tensor(coefficients, dtype=torch.float32, device=device)
    return matrix


def batched_patch_outputs(
    model: Any,
    *,
    configs: Sequence[PatchConfiguration],
    specs: Sequence[MultiDepthResamplingSpec],
    clean_runs: Mapping[str, CleanRun],
    sites: Sequence[ChannelSite],
    readout: FrozenDepthReadout,
    prefix_cache_bank: Mapping[str, PrefixKVCache],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
    max_batch_size: int,
    include_margin: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Evaluate a cross-product of patch configurations and source/base records.

    Each member of a group receives the full source-minus-base swap, multiplied
    only by the configuration strength. No group-size normalization is applied.
    """

    from circuit_sparsity.inference.hook_utils import hook_recorder

    if not configs or not specs:
        raise ValueError("configs and specs must both be nonempty")
    if int(max_batch_size) <= 0:
        raise ValueError("max_batch_size must be positive")

    record_count = len(specs)
    config_count = len(configs)
    output_phi = np.empty((config_count, record_count, 4), dtype=np.float32)
    output_margin = np.empty((config_count, record_count), dtype=np.float32) if include_margin else None
    source_features = torch.tensor(
        [clean_runs[spec.source_id].feature_vector for spec in specs],
        dtype=torch.float32,
        device=device,
    )
    base_phi = torch.tensor(
        [clean_runs[spec.base_id].depth_phi for spec in specs],
        dtype=torch.float32,
        device=device,
    )
    members = _member_matrix(configs, device=device)
    coefficients = _coefficient_matrix(configs, device=device)
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
        if base_id not in prefix_cache_bank:
            raise KeyError(f"missing prefix cache for base {base_id}")
        record_indices = torch.tensor(record_indices_list, dtype=torch.long, device=device)
        bucket_records = len(record_indices_list)
        task_count = config_count * bucket_records
        for task_start in range(0, task_count, int(max_batch_size)):
            task_stop = min(task_count, task_start + int(max_batch_size))
            flat = torch.arange(task_start, task_stop, dtype=torch.long, device=device)
            config_indices = torch.div(flat, bucket_records, rounding_mode="floor")
            bucket_indices = flat.remainder(bucket_records)
            global_records = record_indices[bucket_indices]
            batch_size = int(flat.shape[0])

            selected_members = members[config_indices]
            rows = torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1).expand_as(selected_members)
            valid = selected_members >= 0
            patch_rows = rows[valid]
            patch_sites = selected_members[valid]
            patch_records = global_records.unsqueeze(1).expand_as(selected_members)[valid]
            patch_source_values = source_features[patch_records, patch_sites]
            patch_strengths = coefficients[config_indices][valid]
            patch_hook_codes = site_hook_codes[patch_sites]
            patch_channels = site_channels[patch_sites]

            feature_matrix = torch.empty((batch_size, len(sites)), dtype=torch.float32, device=device)
            seen_hooks: set[str] = set()
            interventions: dict[str, Any] = {}
            for hook_name in hook_names:
                code = hook_code[hook_name]
                patch_mask = patch_hook_codes == code
                hook_patch_rows = patch_rows[patch_mask]
                hook_patch_channels = patch_channels[patch_mask]
                hook_source_values = patch_source_values[patch_mask]
                hook_strengths = patch_strengths[patch_mask]
                capture_indices = layout[hook_name]["site_indices"].to(device=device)
                capture_channels = layout[hook_name]["channels"].to(device=device)

                def _intervene(
                    tensor: torch.Tensor,
                    *,
                    hook_name: str = hook_name,
                    hook_patch_rows: torch.Tensor = hook_patch_rows,
                    hook_patch_channels: torch.Tensor = hook_patch_channels,
                    hook_source_values: torch.Tensor = hook_source_values,
                    hook_strengths: torch.Tensor = hook_strengths,
                    capture_indices: torch.Tensor = capture_indices,
                    capture_channels: torch.Tensor = capture_channels,
                ) -> torch.Tensor:
                    seen_hooks.add(hook_name)
                    if hook_patch_rows.numel():
                        tensor = tensor.clone()
                        base_values = tensor[hook_patch_rows, -1, hook_patch_channels]
                        source_values = hook_source_values.to(device=tensor.device, dtype=tensor.dtype)
                        alpha = hook_strengths.to(device=tensor.device, dtype=tensor.dtype)
                        tensor[hook_patch_rows, -1, hook_patch_channels] = base_values + alpha * (
                            source_values - base_values
                        )
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
                        include_margin=include_margin,
                    )
            missing_hooks = set(hook_names) - seen_hooks
            if missing_hooks:
                raise RuntimeError(f"candidate hooks were not observed: {sorted(missing_hooks)}")
            patched_phi = feature_matrix @ readout_weights
            delta_phi = patched_phi - base_phi[global_records]
            config_cpu = config_indices.detach().cpu().numpy()
            record_cpu = global_records.detach().cpu().numpy()
            output_phi[config_cpu, record_cpu, :] = delta_phi.detach().cpu().numpy()
            if include_margin and output_margin is not None:
                if margins is None:
                    raise AssertionError("margin-producing forward returned no margins")
                output_margin[config_cpu, record_cpu] = margins.detach().to(torch.float32).cpu().numpy()
    return output_phi, output_margin


def _open_signature_checkpoint(
    out_dir: Path,
    *,
    group_count: int,
    record_count: int,
    manifest_hash: str,
    resume: bool,
) -> tuple[np.memmap, np.memmap]:
    signature_path = out_dir / "signatures" / "group_delta_phi.npy"
    complete_path = out_dir / "signatures" / "group_complete.npy"
    metadata_path = out_dir / "signatures" / "checkpoint_manifest.json"
    metadata = {
        "manifest_hash": manifest_hash,
        "shape": [int(group_count), int(record_count), 4],
        "dtype": "float32",
        "semantics": "phi_D(y_swap_group)-phi_D(y_base); every group member fully swapped at lambda=1",
    }
    signature_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("signature checkpoint manifest differs")
        signatures = np.lib.format.open_memmap(signature_path, mode="r+")
        complete = np.lib.format.open_memmap(complete_path, mode="r+")
    else:
        _atomic_json(metadata_path, metadata)
        signatures = np.lib.format.open_memmap(
            signature_path,
            mode="w+",
            dtype=np.float32,
            shape=(group_count, record_count, 4),
        )
        complete = np.lib.format.open_memmap(
            complete_path,
            mode="w+",
            dtype=np.bool_,
            shape=(group_count,),
        )
        complete[:] = False
        complete.flush()
    return signatures, complete


def _scan_group_signatures(
    model: Any,
    *,
    groups: Sequence[GroupCandidate],
    specs: Sequence[MultiDepthResamplingSpec],
    clean_runs: Mapping[str, CleanRun],
    sites: Sequence[ChannelSite],
    readout: FrozenDepthReadout,
    prefix_cache_bank: Mapping[str, PrefixKVCache],
    exact_singleton_signatures: Mapping[str, np.ndarray] | None,
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
    max_batch_size: int,
    group_chunk_size: int,
    out_dir: Path,
    manifest_hash: str,
    resume: bool,
    start: int,
    stop: int | None,
) -> np.memmap:
    signatures, complete = _open_signature_checkpoint(
        out_dir,
        group_count=len(groups),
        record_count=len(specs),
        manifest_hash=manifest_hash,
        resume=resume,
    )
    if exact_singleton_signatures is not None:
        seeded = 0
        for group_index, group in enumerate(groups):
            if group.group_size != 1:
                break
            site_id = group.site_ids[0]
            if site_id not in exact_singleton_signatures:
                raise KeyError(f"missing exact parent singleton signature for {site_id}")
            expected = np.asarray(exact_singleton_signatures[site_id], dtype=np.float32)
            if expected.shape != (len(specs), 4):
                raise ValueError(f"unexpected singleton signature shape for {site_id}: {expected.shape}")
            if bool(complete[group_index]):
                if not np.array_equal(np.asarray(signatures[group_index]), expected):
                    signatures[group_index, :, :] = expected
            else:
                signatures[group_index, :, :] = expected
            complete[group_index] = True
            seeded += 1
        signatures.flush()
        complete.flush()
        if seeded != len(sites):
            raise ValueError(f"seeded {seeded} exact singletons, expected {len(sites)}")
    first = max(0, int(start))
    last = len(groups) if stop is None else min(len(groups), int(stop))
    started = time.time()
    initial_complete = int(np.count_nonzero(complete[first:last]))
    for chunk_start in range(first, last, int(group_chunk_size)):
        chunk_stop = min(last, chunk_start + int(group_chunk_size))
        pending_indices = [index for index in range(chunk_start, chunk_stop) if not bool(complete[index])]
        if not pending_indices:
            continue
        configs = [
            PatchConfiguration(
                config_id=groups[index].group_id,
                group=groups[index],
                strength=1.0,
                ranking_position=index + 1,
            )
            for index in pending_indices
        ]
        delta_phi, _margins = batched_patch_outputs(
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
            max_batch_size=max_batch_size,
            include_margin=False,
        )
        signatures[pending_indices, :, :] = delta_phi
        signatures.flush()
        complete[pending_indices] = True
        complete.flush()
        done = int(np.count_nonzero(complete[first:last]))
        elapsed = max(time.time() - started, 1e-6)
        new_done = max(done - initial_complete, 1)
        remaining = (last - first) - done
        eta_minutes = remaining * elapsed / new_done / 60.0
        print(
            f"group signatures {done}/{last-first}; rows {pending_indices[0]}..{pending_indices[-1]}; "
            f"elapsed={elapsed/60.0:.1f}m eta={eta_minutes:.1f}m",
            flush=True,
        )
    return signatures


def _selector_for_target(
    *,
    target: str,
    abstract_matrix: torch.Tensor,
    signatures: np.ndarray,
    groups: Sequence[GroupCandidate],
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    abstract = target_signature(abstract_matrix, target).detach().cpu().numpy().astype(np.float32, copy=False)
    if target == "D":
        neural = np.asarray(signatures, dtype=np.float32).reshape(len(groups), -1)
    else:
        component = TARGET_COMPONENTS[target]
        neural = np.asarray(signatures[:, :, component], dtype=np.float32)
    if neural.shape[1] != abstract.shape[0]:
        raise ValueError("abstract and neural signature widths differ")
    abstract_norm = max(float(np.linalg.norm(abstract)), 1e-12)
    neural_norm = np.maximum(np.linalg.norm(neural, axis=1), 1e-12)
    dots = neural @ abstract
    similarity = dots / (neural_norm * abstract_norm)
    similarity = np.nan_to_num(similarity, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)
    cosine_cost = (1.0 - similarity).astype(np.float32)
    squared_cost = (
        np.sum(neural * neural, axis=1)
        + float(np.sum(abstract * abstract))
        - 2.0 * dots
    ).astype(np.float32)
    main_uot = sinkhorn_one_sided_uot(
        torch.from_numpy(cosine_cost).reshape(1, -1),
        epsilon=float(epsilon),
        beta_neural=float(beta),
        n_iter=300,
    )[0].numpy()
    squared_uot = sinkhorn_one_sided_uot(
        torch.from_numpy(squared_cost).reshape(1, -1),
        epsilon=float(epsilon),
        beta_neural=float(beta),
        n_iter=300,
    )[0].numpy()
    rows = [
        {
            "group_index": index,
            "group_id": group.group_id,
            "site_ids": list(group.site_ids),
            "group_size": group.group_size,
            "raw_cosine_uot_weight": float(main_uot[index]),
            "cosine_similarity": float(similarity[index]),
            "cosine_cost": float(cosine_cost[index]),
            "raw_squared_uot_weight": float(squared_uot[index]),
            "squared_cost": float(squared_cost[index]),
        }
        for index, group in enumerate(groups)
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["raw_cosine_uot_weight"]),
            float(row["cosine_cost"]),
            int(row["group_size"]),
            str(row["group_id"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return {
        "target": target,
        "candidate_count": len(groups),
        "main_selector": "raw_cosine_uot",
        "epsilon": float(epsilon),
        "beta": float(beta),
        "signature_semantics": "phi(y_swap_group)-phi(y_base)",
        "group_semantics": "all members receive the full source-minus-base swap at lambda=1",
        "ranked_groups": ranked,
    }


def _vector_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def _depth_record_from_outputs(
    *,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    patched_phi: Sequence[float],
    patched_margin: float,
) -> dict[str, Any]:
    base_example = examples[spec.base_id]
    source_example = examples[spec.source_id]
    base_phi = clean_runs[spec.base_id].depth_phi
    source_phi = clean_runs[spec.source_id].depth_phi
    base_distance = _vector_distance(base_phi, source_phi)
    patched_distance = _vector_distance(patched_phi, source_phi)
    effect_fraction = _vector_distance(patched_phi, base_phi) / base_distance if base_distance > 1e-6 else float("nan")
    close_count = 2 if float(patched_margin) > 0.0 else 1
    predicted_depth = depth_from_phi(patched_phi)
    return {
        "spec_key": spec_key(spec),
        "relation": spec.relation,
        "wrong_variable": spec.wrong_variable,
        "base_id": spec.base_id,
        "source_id": spec.source_id,
        "base_depth": int(base_example.depth),
        "source_depth": int(source_example.depth),
        "base_close_count": int(base_example.close_count),
        "source_close_count": int(source_example.close_count),
        "patched_close_count": close_count,
        "patched_margin": float(patched_margin),
        "base_depth_phi": list(base_phi),
        "source_depth_phi": list(source_phi),
        "patched_depth_phi": [float(value) for value in patched_phi],
        "predicted_depth": predicted_depth,
        "depth_matches_source": predicted_depth == int(source_example.depth),
        "depth_matches_base": predicted_depth == int(base_example.depth),
        "depth_moves_toward_source": patched_distance + 1e-6 < base_distance,
        "depth_effect_fraction": float(effect_fraction),
        "output_matches_source": close_count == int(source_example.close_count),
        "output_preserves_base": close_count == int(base_example.close_count),
    }


def _worst_component_gate(summary: Mapping[str, Any]) -> float:
    values = [float(value) for value in summary["metrics"].values()]
    for row in summary["directional"].values():
        values.extend(float(row[name]) for name in ("target_source_match", "non_target_base_preserve", "expected_output_success"))
    return min(values)


def _worst_depth_gate(summary: Mapping[str, Any]) -> float:
    values = [float(value) for value in summary["metrics"].values()]
    for transitions in summary["ordered_transitions"].values():
        for row in transitions.values():
            values.extend((float(row["depth_source_match"]), float(row["expected_output_success"])))
    return min(values)


def _calibrate_target(
    model: Any,
    *,
    target: str,
    selector: Mapping[str, Any],
    groups: Sequence[GroupCandidate],
    strengths: Sequence[float],
    top_groups: int,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    sites: Sequence[ChannelSite],
    readout: FrozenDepthReadout,
    prefix_cache_bank: Mapping[str, PrefixKVCache],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
    max_batch_size: int,
    acceptance_threshold: float,
) -> dict[str, Any]:
    ranked = list(selector["ranked_groups"][: int(top_groups)])
    if target == "D":
        eligible_specs = list(specs)
    else:
        eligible_specs = [spec for spec in specs if individual_spec_eligible(target, spec, examples)]
    configs: list[PatchConfiguration] = []
    for row in ranked:
        group = groups[int(row["group_index"])]
        for strength in strengths:
            configs.append(
                PatchConfiguration(
                    config_id=f"{target}:rank{row['rank']}:lambda{float(strength):g}",
                    group=group,
                    strength=float(strength),
                    ranking_position=int(row["rank"]),
                )
            )
    delta_phi, margins = batched_patch_outputs(
        model,
        configs=configs,
        specs=eligible_specs,
        clean_runs=clean_runs,
        sites=sites,
        readout=readout,
        prefix_cache_bank=prefix_cache_bank,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=max_batch_size,
        include_margin=True,
    )
    if margins is None:
        raise AssertionError("calibration requested margins")

    rows: list[dict[str, Any]] = []
    records_by_config: dict[str, list[dict[str, Any]]] = {}
    for config_index, config in enumerate(configs):
        records: list[dict[str, Any]] = []
        for record_index, spec in enumerate(eligible_specs):
            patched_phi = np.asarray(clean_runs[spec.base_id].depth_phi, dtype=np.float32) + delta_phi[
                config_index, record_index
            ]
            if target == "D":
                record = _depth_record_from_outputs(
                    spec=spec,
                    examples=examples,
                    clean_runs=clean_runs,
                    patched_phi=patched_phi,
                    patched_margin=float(margins[config_index, record_index]),
                )
            else:
                record = expected_component_record(
                    target=target,
                    spec=spec,
                    examples=examples,
                    patched_phi=patched_phi,
                    patched_close_count=2 if float(margins[config_index, record_index]) > 0.0 else 1,
                )
                record["spec_key"] = spec_key(spec)
                record["patched_margin"] = float(margins[config_index, record_index])
            records.append(record)
        if target == "D":
            summary = summarize_validation_records(records)
            acceptance = depth_acceptance(summary, threshold=float(acceptance_threshold))
            score = depth_calibration_score(summary)
            worst_gate = _worst_depth_gate(summary)
            validated = bool(acceptance["D_validated"])
        else:
            summary = summarize_component_records(records)
            acceptance = component_acceptance(target, summary, threshold=float(acceptance_threshold))
            score = component_calibration_score(summary)
            worst_gate = _worst_component_gate(summary)
            validated = bool(acceptance["validated"])
        row = {
            "config_id": config.config_id,
            "group_id": config.group.group_id,
            "site_ids": list(config.group.site_ids),
            "distinct_site_count": config.group.group_size,
            "group_size": config.group.group_size,
            "strength": config.strength,
            "ranking_position": config.ranking_position,
            "calibration_score": float(score),
            "worst_gate": float(worst_gate),
            "validated": validated,
            "acceptance": acceptance,
            "summary": summary,
        }
        rows.append(row)
        records_by_config[config.config_id] = records
    best = dict(select_pass_first(rows))
    return {
        "target": target,
        "selection_rule": "pass first; then smallest support, worst gate, average score, lambda closest to 1, rank",
        "top_groups_calibrated": int(top_groups),
        "grid": rows,
        "best": best,
        "best_records": records_by_config[str(best["config_id"])],
    }


def _posthoc_group_ranks(selector: Mapping[str, Any]) -> dict[str, Any]:
    known = (
        "2.attn.resid_delta:1249",
        "4.attn.resid_delta:1079",
        "7.mlp.post_act:4133",
        "7.mlp.resid_delta:2041",
    )
    singleton = {site_id: None for site_id in known}
    best_containing = {site_id: None for site_id in known}
    for row in selector["ranked_groups"]:
        members = set(row["site_ids"])
        for site_id in known:
            if site_id in members and best_containing[site_id] is None:
                best_containing[site_id] = int(row["rank"])
            if members == {site_id}:
                singleton[site_id] = int(row["rank"])
    return {"singleton_rank": singleton, "best_group_containing_rank": best_containing}


def main() -> None:
    args = parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is unavailable")
    resume = not bool(args.no_resume)
    device = "cuda" if args.cuda else "cpu"
    depths = parse_depths(args.depths)
    if depths != (1, 2, 3, 4):
        raise ValueError("this experiment is preregistered for depths 1,2,3,4")
    strengths = _parse_float_grid(args.strength_grid)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv,
        expected_node_count=int(args.expected_node_count),
    )
    candidate_sites = tuple(universe.sites)
    if len(candidate_sites) != 133:
        raise ValueError("primary universe must contain exactly all 133 localized nodes")
    groups = build_single_pair_groups(candidate_sites)
    if len(groups) != 8911 or sum(group.group_size == 2 for group in groups) != 8778:
        raise AssertionError("singleton/pair universe is incomplete")
    encoding = make_tinypython_encoding(args.circuit_home)
    parent = _load_parent(
        args.parent_run_dir,
        candidate_sites=candidate_sites,
        candidate_csv_sha256=universe.csv_sha256,
        encoding=encoding,
        depths=depths,
        records_per_relation=int(args.records_per_relation),
    )

    group_manifest = {
        "candidate_csv": str(args.candidate_node_csv),
        "candidate_csv_sha256": universe.csv_sha256,
        "candidate_count": len(candidate_sites),
        "all_candidate_node_ids": [site.site_id for site in candidate_sites],
        "no_filtering_applied": True,
        "group_count": len(groups),
        "singleton_count": sum(group.group_size == 1 for group in groups),
        "pair_count": sum(group.group_size == 2 for group in groups),
        "group_construction": "all singletons followed by all unordered pairs in CSV order",
        "group_patch_semantics": "every member fully swaps from source into base; no group-size normalization",
        "known_site_policy": "known sites are used only after ranking for recovery reporting",
        "groups": [
            {
                "group_index": index,
                "group_id": group.group_id,
                "site_indices": list(group.site_indices),
                "site_ids": list(group.site_ids),
            }
            for index, group in enumerate(groups)
        ],
    }
    _atomic_json(out_dir / "candidate_group_manifest.json", group_manifest)
    run_manifest = {
        "schema_version": 1,
        "model": args.model,
        "parent_run_dir": str(args.parent_run_dir),
        "parent_candidate_manifest_sha256": canonical_sha256(parent["candidate_manifest"]),
        "candidate_group_manifest_sha256": canonical_sha256(group_manifest),
        "targets": list(TARGETS),
        "Dfit_record_count": len(parent["specs"]["Dfit"]),
        "Dcal_record_count": len(parent["specs"]["Dcal"]),
        "Dte_policy": "not used by this development selector/calibration runner",
        "effect_signature": "phi(y_swap_group)-phi(y_base)",
        "phi": ["norm_D", "T2", "T3", "T4"],
        "selector": "raw cosine-cost one-sided UOT",
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "calibrate_top_groups": int(args.calibrate_top_groups),
        "strength_grid": list(strengths),
        "acceptance_threshold": float(args.acceptance_threshold),
        "group_chunk_size": int(args.group_chunk_size),
        "max_batch_size": int(args.max_batch_size),
        "linear_kernel": "dense" if args.dense_kernels else "exact_sparse_csr",
        "full_logits_policy": "not materialized; only the exact ]/]] margin is computed when required",
        "prefix_execution": "cache causally fixed clean prefix K/V and recompute only the final token",
        "equivalence_base_limit": int(args.equivalence_base_limit),
    }
    _atomic_json(out_dir / "run_manifest.json", run_manifest)
    manifest_hash = canonical_sha256(run_manifest)

    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not bool(args.no_flash),
        grad_checkpointing=False,
    )
    sparse_records = () if args.dense_kernels else convert_transformer_linears_to_sparse(model)
    if args.cuda:
        torch.cuda.empty_cache()
    model_info["execution_linear_kernel"] = "dense" if args.dense_kernels else "exact_sparse_csr"
    model_info["sparse_conversion"] = [record.to_json() for record in sparse_records]
    _atomic_json(out_dir / "model_info.json", model_info)
    single_close_token_id = int(encoding.encode("]\n")[0])
    double_close_token_id = int(encoding.encode("]]\n")[0])
    clean_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=parent["examples"],
        candidate_sites=candidate_sites,
        checkpoint_path=args.parent_run_dir / "clean_runs.jsonl",
        resume=True,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    _attach_phi(parent["readout"], clean_runs)
    if not all(clean_runs[example.example_id].correct for example in parent["examples"]):
        raise ValueError("clean model accuracy is below 1.0")
    cache_splits = ("Dfit",) if args.signatures_only else ("Dfit", "Dcal")
    prefix_base_ids = sorted(
        {
            spec.base_id
            for split in cache_splits
            for spec in parent["specs"][split]
        }
    )
    prefix_cache_path = out_dir / "prefix_cache.pt"
    if resume and prefix_cache_path.exists():
        prefix_cache_bank = _load_prefix_cache_bank(prefix_cache_path, device=device)
        if set(prefix_cache_bank) != set(prefix_base_ids):
            raise ValueError("persisted prefix cache base IDs differ from the current run")
    else:
        prefix_cache_bank = build_prefix_cache_bank(
            model,
            {base_id: clean_runs[base_id].token_ids for base_id in prefix_base_ids},
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        _save_prefix_cache_bank(prefix_cache_path, prefix_cache_bank)
    equivalence_base_ids = (
        prefix_base_ids
        if int(args.equivalence_base_limit) <= 0
        else prefix_base_ids[: int(args.equivalence_base_limit)]
    )
    equivalence_path = out_dir / "incremental_execution_equivalence.json"
    if resume and equivalence_path.exists():
        execution_equivalence = json.loads(equivalence_path.read_text(encoding="utf-8"))
    else:
        execution_equivalence = _verify_incremental_execution(
            model,
            base_ids=equivalence_base_ids,
            clean_runs=clean_runs,
            sites=candidate_sites,
            prefix_cache_bank=prefix_cache_bank,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        execution_equivalence.update(
            {
                "candidate_feature_tolerance": 2e-3,
                "margin_tolerance": 2e-3,
            }
        )
        execution_equivalence["passed"] = (
            float(execution_equivalence["max_candidate_feature_error"]) <= 2e-3
            and float(execution_equivalence["max_margin_error"]) <= 2e-3
        )
        _atomic_json(equivalence_path, execution_equivalence)
    if not bool(execution_equivalence["passed"]):
        raise ValueError(f"incremental final-token execution failed equivalence: {execution_equivalence}")

    signatures = _scan_group_signatures(
        model,
        groups=groups,
        specs=parent["specs"]["Dfit"],
        clean_runs=clean_runs,
        sites=candidate_sites,
        readout=parent["readout"],
        prefix_cache_bank=prefix_cache_bank,
        exact_singleton_signatures=parent["singleton_signatures"],
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
        max_batch_size=int(args.max_batch_size),
        group_chunk_size=int(args.group_chunk_size),
        out_dir=out_dir,
        manifest_hash=manifest_hash,
        resume=resume,
        start=int(args.signature_start),
        stop=args.signature_stop,
    )

    complete = np.lib.format.open_memmap(out_dir / "signatures" / "group_complete.npy", mode="r")
    if not bool(np.all(complete)):
        print(
            json.dumps(
                {
                    "status": "partial_signature_scan",
                    "complete": int(np.count_nonzero(complete)),
                    "total": len(groups),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    singleton_errors = {}
    singleton_p99_errors = {}
    singleton_mean_errors = {}
    for index, site in enumerate(candidate_sites):
        absolute_error = np.abs(np.asarray(signatures[index]) - parent["singleton_signatures"][site.site_id])
        singleton_errors[site.site_id] = float(np.max(absolute_error))
        singleton_p99_errors[site.site_id] = float(np.quantile(absolute_error, 0.99))
        singleton_mean_errors[site.site_id] = float(np.mean(absolute_error))
    max_singleton_error = max(singleton_errors.values())
    max_singleton_p99_error = max(singleton_p99_errors.values())
    tolerance = 2e-3
    p99_tolerance = 1e-4
    _atomic_json(
        out_dir / "signatures" / "singleton_equivalence.json",
        {
            "max_absolute_error": max_singleton_error,
            "max_per_site_p99_absolute_error": max_singleton_p99_error,
            "per_site_max_absolute_error": singleton_errors,
            "per_site_p99_absolute_error": singleton_p99_errors,
            "per_site_mean_absolute_error": singleton_mean_errors,
            "max_tolerance": tolerance,
            "p99_tolerance": p99_tolerance,
            "passed": max_singleton_error <= tolerance and max_singleton_p99_error <= p99_tolerance,
        },
    )
    if max_singleton_error > tolerance or max_singleton_p99_error > p99_tolerance:
        raise ValueError(
            "batched full-swap semantics disagree with parent singletons: "
            f"max={max_singleton_error}, max_site_p99={max_singleton_p99_error}"
        )
    if args.signatures_only:
        print(json.dumps({"status": "signatures_complete", "max_singleton_error": max_singleton_error}, indent=2))
        return

    abstract_matrix = abstract_delta_matrix(
        parent["specs"]["Dfit"],
        parent["examples_by_id"],
        max_depth=max(depths),
    )
    selectors: dict[str, Any] = {}
    calibrations: dict[str, Any] = {}
    for target in TARGETS:
        selector = _selector_for_target(
            target=target,
            abstract_matrix=abstract_matrix,
            signatures=signatures,
            groups=groups,
            epsilon=float(args.selector_epsilon),
            beta=float(args.selector_beta),
        )
        selector["posthoc_recovery"] = _posthoc_group_ranks(selector)
        _atomic_json(out_dir / "selectors" / f"{target}.json", selector)
        selectors[target] = selector
        calibration = _calibrate_target(
            model,
            target=target,
            selector=selector,
            groups=groups,
            strengths=strengths,
            top_groups=int(args.calibrate_top_groups),
            specs=parent["specs"]["Dcal"],
            examples=parent["examples_by_id"],
            clean_runs=clean_runs,
            sites=candidate_sites,
            readout=parent["readout"],
            prefix_cache_bank=prefix_cache_bank,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
            device=device,
            max_batch_size=int(args.max_batch_size),
            acceptance_threshold=float(args.acceptance_threshold),
        )
        _atomic_json(out_dir / "calibration" / f"{target}.json", calibration)
        calibrations[target] = calibration
        print(
            json.dumps(
                {
                    "target": target,
                    "top_group": selector["ranked_groups"][0]["group_id"],
                    "selected": calibration["best"]["group_id"],
                    "strength": calibration["best"]["strength"],
                    "validated_on_Dcal": calibration["best"]["validated"],
                    "worst_gate": calibration["best"]["worst_gate"],
                }
            ),
            flush=True,
        )

    summary = {
        "status": "development_selection_complete",
        "candidate_count": len(candidate_sites),
        "group_count": len(groups),
        "no_filtering_applied": True,
        "max_singleton_equivalence_error": max_singleton_error,
        "Dte_used": False,
        "targets": {
            target: {
                "raw_top_group": selectors[target]["ranked_groups"][0],
                "posthoc_recovery": selectors[target]["posthoc_recovery"],
                "calibrated_best": calibrations[target]["best"],
            }
            for target in TARGETS
        },
    }
    _atomic_json(out_dir / "development_group_plot_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
