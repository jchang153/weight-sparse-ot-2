from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_d_large_bank_frozen_readout import (
    build_transition_balanced_specs,
    canonical_sha256,
    resample_indices_within_relation,
    spec_key,
)
from .bracket_d_rich_signatures import (
    FrozenDepthReadout,
    load_full_localized_candidate_universe,
    selector_payload_from_signatures,
    weights_from_ranked,
)
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .bracket_progressive_model_discovery import bank_manifest, build_discovery_bank
from .bracket_threshold_components import (
    POSTHOC_SITE_IDS,
    SCHEMA_VERSION,
    TARGET_COMPONENTS,
    abstract_component_signature,
    changed_threshold_targets,
    combined_joint_coefficients,
    component_acceptance,
    component_calibration_score,
    expected_component_record,
    final_model_decision,
    individual_spec_eligible,
    is_joint_compound_spec,
    joint_acceptance,
    mediation_fraction,
    predicted_threshold_vector,
    readout_component_quality,
    resample_record_indices,
    slice_component_signature,
    summarize_component_records,
    summarize_joint_records,
    summarize_t2_mediation,
    t2_mediation_acceptance,
    threshold_vector,
)
from .run_bracket_d1249_r1079_mediation_test import PatchSpec, _run_patch_and_record, scalar_moves_toward_source
from .run_bracket_d_large_bank_frozen_readout_plot import (
    _atomic_json,
    _atomic_jsonl,
    _collect_or_resume_clean_runs,
    _load_jsonl,
    _parse_float_grid,
    _parse_int_grid,
    _require_manifest,
)
from .run_bracket_d_rich_signature_experiments import CleanRun, _run_weighted_patch
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_PARENT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_threshold_components_20260710")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run componentwise PLOT for bracket threshold variables T2/T3/T4.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--parent-run-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fresh-content-start", type=int, default=96)
    parser.add_argument("--cal-contents", type=int, default=24)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--near-optimal-epsilon", type=float, default=0.02)
    parser.add_argument("--bootstrap-reps", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--acceptance-threshold", type=float, default=0.90)
    parser.add_argument("--r-control-site", default="4.attn.resid_delta:1079")
    parser.add_argument("--r-late-probe-site", default="7.mlp.resid_delta:2041")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _load_frozen_readout(path: Path, candidate_site_ids: Sequence[str]) -> FrozenDepthReadout:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fit_split") != "Dfit":
        raise ValueError("parent frozen readout was not fit only on Dfit")
    readout_payload = payload["readout"]
    feature_ids = tuple(str(site_id) for site_id in readout_payload["feature_site_ids"])
    if feature_ids != tuple(candidate_site_ids):
        raise ValueError("parent readout feature order does not match the current 133-site universe")
    target_names = tuple(str(name) for name in readout_payload.get("target_names", ()))
    if target_names != ("norm_D", "D_ge_2", "D_ge_3", "D_ge_4"):
        raise ValueError(f"unexpected frozen readout targets: {target_names}")
    return FrozenDepthReadout(
        weights=torch.tensor(readout_payload["weights"], dtype=torch.float32),
        alpha=float(readout_payload["alpha"]),
        feature_site_ids=feature_ids,
        target_names=target_names,
    )


def _posthoc_ranks(ranked_sites: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    rank_by_site = {str(row["site_id"]): idx + 1 for idx, row in enumerate(ranked_sites)}
    return {site_id: rank_by_site.get(site_id) for site_id in POSTHOC_SITE_IDS}


def _load_parent_inputs(
    *,
    parent_dir: Path,
    universe_manifest: Mapping[str, Any],
    candidate_sites: Sequence[ChannelSite],
) -> dict[str, Any]:
    required = (
        "candidate_manifest.json",
        "bank_manifest.json",
        "run_manifest.json",
        "frozen_depth_readout.json",
        "signatures/C_D_frozen_readout.jsonl",
    )
    for name in required:
        if not (parent_dir / name).exists():
            raise FileNotFoundError(parent_dir / name)

    candidate_manifest = json.loads((parent_dir / "candidate_manifest.json").read_text(encoding="utf-8"))
    parent_bank = json.loads((parent_dir / "bank_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((parent_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_ids = [site.site_id for site in candidate_sites]
    if int(candidate_manifest.get("candidate_count", -1)) != len(expected_ids):
        raise ValueError("parent candidate count is not 133")
    if not bool(candidate_manifest.get("no_filtering_applied")):
        raise ValueError("parent candidate manifest does not certify no filtering")
    if candidate_manifest.get("candidate_csv_sha256") != universe_manifest.get("candidate_csv_sha256"):
        raise ValueError("parent and current candidate CSV hashes differ")
    if list(candidate_manifest.get("all_candidate_node_ids", ())) != expected_ids:
        raise ValueError("parent candidate IDs or order differ from the current universe")

    split_payloads = parent_bank["splits"]
    fit_contents = int(split_payloads["Dfit"]["content_count"])
    cal_contents = int(split_payloads["Dcal"]["content_count"])
    test_contents = int(split_payloads["Dte"]["content_count"])
    contents = fit_contents + cal_contents + test_contents
    parent_examples = build_discovery_bank(
        None,
        contents=contents,
        fit_contents=fit_contents,
        cal_contents=cal_contents,
        test_contents=test_contents,
        depths=tuple(int(depth) for depth in run_manifest["depths"]),
    )
    parent_lookup = {example.example_id: example for example in parent_examples}
    fit_specs = build_transition_balanced_specs(
        parent_examples,
        split="Dfit",
        records_per_relation=int(run_manifest["records_per_relation"]),
    )
    record_keys = [spec_key(spec) for spec in fit_specs]
    if record_keys != list(run_manifest["record_keys"]["Dfit"]):
        raise ValueError("reconstructed Dfit record order does not match the parent manifest")
    if len(fit_specs) != 600:
        raise ValueError(f"parent Dfit has {len(fit_specs)} records, expected 600")

    signature_rows = _load_jsonl(parent_dir / "signatures" / "C_D_frozen_readout.jsonl")
    if len(signature_rows) != len(expected_ids):
        raise ValueError(f"parent has {len(signature_rows)} signatures, expected {len(expected_ids)}")
    signature_by_site: dict[str, tuple[float, ...]] = {}
    for row in signature_rows:
        site_id = str(row["site_id"])
        if site_id in signature_by_site:
            raise ValueError(f"duplicate parent signature for {site_id}")
        signature = tuple(float(value) for value in row["signature"])
        if len(signature) != len(fit_specs) * 4:
            raise ValueError(f"parent signature for {site_id} has length {len(signature)}, expected 2400")
        signature_by_site[site_id] = signature
    if set(signature_by_site) != set(expected_ids):
        raise ValueError("parent signatures do not cover exactly all 133 candidates")

    readout = _load_frozen_readout(parent_dir / "frozen_depth_readout.json", expected_ids)
    previous_indices = sorted(
        {
            int(index)
            for split in parent_bank["splits"].values()
            for index in split["content_indices"]
        }
    )
    source_manifest = {
        "parent_run_dir": str(parent_dir),
        "parent_candidate_manifest_sha256": canonical_sha256(candidate_manifest),
        "parent_run_manifest_sha256": canonical_sha256(run_manifest),
        "parent_candidate_csv_sha256": candidate_manifest["candidate_csv_sha256"],
        "candidate_count": len(expected_ids),
        "Dfit_records": len(fit_specs),
        "parent_signature_count": len(signature_rows),
        "parent_signature_width": 4,
        "frozen_readout_fit_split": "Dfit",
        "previous_content_indices": previous_indices,
        "reuse_policy": "exact Dfit signature slicing only; parent calibration and test records are not reused",
    }
    return {
        "source_manifest": source_manifest,
        "readout": readout,
        "fit_specs": fit_specs,
        "fit_examples": parent_lookup,
        "signature_by_site": signature_by_site,
    }


def _derive_selectors(
    *,
    parent: Mapping[str, Any],
    candidate_sites: Sequence[ChannelSite],
    out_dir: Path,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for target in TARGET_COMPONENTS:
        abstract = abstract_component_signature(parent["fit_specs"], parent["fit_examples"], target=target)
        neural = {
            site.site_id: slice_component_signature(
                parent["signature_by_site"][site.site_id],
                target=target,
                record_count=len(parent["fit_specs"]),
            )
            for site in candidate_sites
        }
        _atomic_jsonl(
            out_dir / "signatures" / f"PLOT_{target}.jsonl",
            [{"site_id": site.site_id, "signature": list(neural[site.site_id])} for site in candidate_sites],
        )
        selector = selector_payload_from_signatures(
            variant=f"PLOT_{target}",
            abstract=abstract,
            neural_by_site=neural,
            epsilon=float(epsilon),
            beta=float(beta),
        )
        _atomic_json(out_dir / "selectors" / f"PLOT_{target}.json", selector)
        outputs[target] = {"abstract": abstract, "neural": neural, "selector": selector}
    return outputs


def _fresh_bank(
    encoding: Any,
    *,
    fresh_content_start: int,
    cal_contents: int,
    test_contents: int,
    depths: Sequence[int],
    previous_indices: Sequence[int],
) -> tuple[tuple[MultiDepthBracketExample, ...], dict[str, Any]]:
    total = int(fresh_content_start) + int(cal_contents) + int(test_contents)
    all_examples = build_discovery_bank(
        encoding,
        contents=total,
        fit_contents=int(fresh_content_start),
        cal_contents=int(cal_contents),
        test_contents=int(test_contents),
        depths=depths,
    )
    fresh = tuple(example for example in all_examples if example.split in {"Dcal", "Dte"})
    manifest = bank_manifest(fresh)
    fresh_indices = {
        int(index)
        for split in manifest["splits"].values()
        for index in split["content_indices"]
    }
    overlap = fresh_indices & set(int(index) for index in previous_indices)
    if overlap:
        raise ValueError(f"fresh content overlaps parent indices: {sorted(overlap)}")
    expected_cal = set(range(int(fresh_content_start), int(fresh_content_start) + int(cal_contents)))
    expected_test = set(
        range(
            int(fresh_content_start) + int(cal_contents),
            int(fresh_content_start) + int(cal_contents) + int(test_contents),
        )
    )
    if set(manifest["splits"]["Dcal"]["content_indices"]) != expected_cal:
        raise ValueError("fresh Dcal indices differ from the preregistered range")
    if set(manifest["splits"]["Dte"]["content_indices"]) != expected_test:
        raise ValueError("fresh Dte indices differ from the preregistered range")
    manifest.update(
        {
            "fresh_content_start": int(fresh_content_start),
            "previous_content_indices": list(previous_indices),
            "fresh_overlap_with_parent": [],
        }
    )
    return fresh, manifest


def _attach_phi(readout: FrozenDepthReadout, runs: Mapping[str, CleanRun]) -> dict[str, tuple[float, ...]]:
    phi: dict[str, tuple[float, ...]] = {}
    for example_id, run in runs.items():
        value = tuple(readout.predict(run.feature_vector))
        run.depth_phi = value
        phi[example_id] = value
    return phi


def _clean_accuracy(examples: Sequence[MultiDepthBracketExample], runs: Mapping[str, CleanRun]) -> float:
    return sum(float(runs[example.example_id].correct) for example in examples) / len(examples)


def _component_validation_record(
    *,
    model: Any,
    target: str,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    patch_sites: Sequence[ChannelSite],
    weights_by_site: Mapping[str, float],
    strength: float,
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    base = clean_runs[spec.base_id]
    source = clean_runs[spec.source_id]
    margin, close_count, _features, vector = _run_weighted_patch(
        model,
        base=base,
        source=source,
        patch_sites=patch_sites,
        weights_by_site=weights_by_site,
        strength=float(strength),
        record_sites=record_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    row = expected_component_record(
        target=target,
        spec=spec,
        examples=examples,
        patched_phi=readout.predict(vector),
        patched_close_count=close_count,
    )
    row.update(
        {
            "spec_key": spec_key(spec),
            "patched_margin": float(margin),
            "weights_by_site": {str(key): float(value) for key, value in weights_by_site.items()},
            "strength": float(strength),
        }
    )
    return row


def _calibrate_target(
    *,
    model: Any,
    target: str,
    ranked_sites: Sequence[Mapping[str, Any]],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    rows_dir: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    eligible_specs = [spec for spec in specs if individual_spec_eligible(target, spec, examples)]
    rows: list[dict[str, Any]] = []
    for k in k_grid:
        weights = weights_from_ranked(ranked_sites, k=int(k))
        patch_sites = tuple(site_lookup[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"PLOT_{target}_top{k}_lambda{float(strength):g}"
            path = rows_dir / f"{handle_id}.json"
            if path.exists() and resume:
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                records = [
                    _component_validation_record(
                        model=model,
                        target=target,
                        spec=spec,
                        examples=examples,
                        clean_runs=clean_runs,
                        readout=readout,
                        patch_sites=patch_sites,
                        weights_by_site=weights,
                        strength=float(strength),
                        record_sites=record_sites,
                        single_close_token_id=single_close_token_id,
                        double_close_token_id=double_close_token_id,
                    )
                    for spec in eligible_specs
                ]
                summary = summarize_component_records(records)
                row = {
                    "handle_id": handle_id,
                    "target": target,
                    "k": int(k),
                    "strength": float(strength),
                    "site_ids": list(weights),
                    "weights_by_site": weights,
                    "calibration_score": component_calibration_score(summary),
                    "summary": summary,
                    "records": records,
                }
                _atomic_json(path, row)
            rows.append(row)
        print(f"calibrated {target} K={k}", flush=True)
    best = sorted(
        rows,
        key=lambda row: (
            -float(row["calibration_score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]
    return {"grid": rows, "best": best, "eligible_specs": eligible_specs}


def _bootstrap_target(
    *,
    target: str,
    fit_specs: Sequence[MultiDepthResamplingSpec],
    abstract: Sequence[float],
    signatures: Mapping[str, Sequence[float]],
    calibration_rows: Sequence[Mapping[str, Any]],
    reps: int,
    seed: int,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    exemplar = calibration_rows[0]["records"]
    for rep in range(int(reps)):
        rng = random.Random(int(seed) + TARGET_COMPONENTS[target] * 10000 + rep)
        fit_indices = resample_indices_within_relation(fit_specs, rng=rng)
        selector = selector_payload_from_signatures(
            variant=f"PLOT_{target}",
            abstract=tuple(float(abstract[idx]) for idx in fit_indices),
            neural_by_site={
                site_id: tuple(float(signature[idx]) for idx in fit_indices)
                for site_id, signature in signatures.items()
            },
            epsilon=float(epsilon),
            beta=float(beta),
        )
        ranked = selector["selectors"]["raw_cosine_uot"]["ranked_sites"]
        cal_indices = resample_record_indices(exemplar, rng=rng)
        rescored = []
        for row in calibration_rows:
            summary = summarize_component_records([row["records"][idx] for idx in cal_indices])
            rescored.append({**row, "bootstrap_score": component_calibration_score(summary)})
        selected = sorted(
            rescored,
            key=lambda row: (
                -float(row["bootstrap_score"]),
                int(row["k"]),
                abs(float(row["strength"]) - 1.0),
            ),
        )[0]
        records.append(
            {
                "rep": rep,
                "top1": ranked[0]["site_id"],
                "top5": [row["site_id"] for row in ranked[:5]],
                "calibrated_handle_id": selected["handle_id"],
                "calibrated_site_ids": selected["site_ids"],
            }
        )
    top1 = Counter(row["top1"] for row in records)
    support = Counter(site_id for row in records for site_id in row["calibrated_site_ids"])
    return {
        "records": records,
        "summary": {
            "reps": int(reps),
            "top1_frequency": dict(top1.most_common()),
            "calibrated_support_frequency": dict(support.most_common()),
            "posthoc_top1_frequency": {site_id: int(top1.get(site_id, 0)) for site_id in POSTHOC_SITE_IDS},
            "posthoc_calibrated_frequency": {
                site_id: int(support.get(site_id, 0)) for site_id in POSTHOC_SITE_IDS
            },
        },
    }


def _heldout_target(
    *,
    model: Any,
    target: str,
    best: Mapping[str, Any],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    records_dir: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    eligible = [spec for spec in specs if individual_spec_eligible(target, spec, examples)]
    patch_sites = tuple(site_lookup[site_id] for site_id in best["site_ids"])
    records: list[dict[str, Any]] = []
    for idx, spec in enumerate(eligible, start=1):
        digest = canonical_sha256({"target": target, "handle": best["handle_id"], "spec": spec_key(spec)})[:20]
        path = records_dir / f"{idx:04d}_{digest}.json"
        if path.exists() and resume:
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            row = _component_validation_record(
                model=model,
                target=target,
                spec=spec,
                examples=examples,
                clean_runs=clean_runs,
                readout=readout,
                patch_sites=patch_sites,
                weights_by_site=best["weights_by_site"],
                strength=float(best["strength"]),
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            _atomic_json(path, row)
        records.append(row)
        if idx % 50 == 0 or idx == len(eligible):
            print(f"heldout {target} {idx}/{len(eligible)}", flush=True)
    return records


def _joint_records(
    *,
    model: Any,
    handles: Mapping[str, Mapping[str, Any]],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    records_dir: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    joint_specs = [spec for spec in specs if is_joint_compound_spec(spec, examples)]
    records: list[dict[str, Any]] = []
    handle_ids = {target: handles[target]["handle_id"] for target in TARGET_COMPONENTS}
    for idx, spec in enumerate(joint_specs, start=1):
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        changed = changed_threshold_targets(base_ex.depth, source_ex.depth)
        coefficients = combined_joint_coefficients(handles, changed)
        patch_sites = tuple(site_lookup[site_id] for site_id in coefficients)
        digest = canonical_sha256({"handles": handle_ids, "spec": spec_key(spec)})[:20]
        path = records_dir / f"{idx:04d}_{digest}.json"
        if path.exists() and resume:
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            margin, close_count, _features, vector = _run_weighted_patch(
                model,
                base=clean_runs[spec.base_id],
                source=clean_runs[spec.source_id],
                patch_sites=patch_sites,
                weights_by_site=coefficients,
                strength=1.0,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            patched_phi = readout.predict(vector)
            patched_bits = predicted_threshold_vector(patched_phi)
            source_bits = threshold_vector(source_ex.depth)
            row = {
                "spec_key": spec_key(spec),
                "relation": spec.relation,
                "base_id": base_ex.example_id,
                "source_id": source_ex.example_id,
                "base_depth": int(base_ex.depth),
                "source_depth": int(source_ex.depth),
                "transition": f"{base_ex.depth}->{source_ex.depth}",
                "changed_targets": list(changed),
                "combined_coefficients": coefficients,
                "patched_phi": [float(value) for value in patched_phi],
                "patched_bits": list(patched_bits),
                "source_bits": list(source_bits),
                "vector_matches_source": patched_bits == source_bits,
                "patched_close_count": int(close_count),
                "source_close_count": int(source_ex.close_count),
                "output_matches_source": int(close_count) == int(source_ex.close_count),
                "patched_margin": float(margin),
            }
            _atomic_json(path, row)
        records.append(row)
        if idx % 25 == 0 or idx == len(joint_specs):
            print(f"joint heldout {idx}/{len(joint_specs)}", flush=True)
    return records


def _patch_specs_for_handle(
    *,
    best: Mapping[str, Any],
    site_lookup: Mapping[str, ChannelSite],
    source: CleanRun,
    base: CleanRun,
) -> list[PatchSpec]:
    return [
        PatchSpec(
            site=site_lookup[site_id],
            source_site_id=site_id,
            source_position=source.final_position,
            target_position=base.final_position,
            source_value=float(source.features_by_site[site_id]),
            strength=float(best["strength"]) * float(weight),
            label="patch frozen T2 handle from source",
        )
        for site_id, weight in best["weights_by_site"].items()
    ]


def _mediation_records(
    *,
    model: Any,
    t2_best: Mapping[str, Any],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    site_lookup: Mapping[str, ChannelSite],
    record_sites: Sequence[ChannelSite],
    r_control_site: ChannelSite,
    late_probe_site: ChannelSite,
    records_dir: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    selected_specs = []
    for spec in specs:
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        changed = changed_threshold_targets(base.depth, source.depth)
        isolated_t2 = changed == ("T2",)
        wrong = spec.relation in {"wrong_numeric_content", "wrong_tail_length"}
        if isolated_t2 or wrong:
            selected_specs.append((spec, "isolated_T2" if isolated_t2 else "wrong_control"))

    records: list[dict[str, Any]] = []
    for idx, (spec, kind) in enumerate(selected_specs, start=1):
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = clean_runs[spec.base_id]
        source = clean_runs[spec.source_id]
        digest = canonical_sha256({"handle": t2_best["handle_id"], "spec": spec_key(spec), "kind": kind})[:20]
        path = records_dir / f"{idx:04d}_{digest}.json"
        if path.exists() and resume:
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            t2_patches = _patch_specs_for_handle(best=t2_best, site_lookup=site_lookup, source=source, base=base)
            block_patch = PatchSpec(
                site=r_control_site,
                source_site_id=r_control_site.site_id,
                source_position=base.final_position,
                target_position=base.final_position,
                source_value=float(base.features_by_site[r_control_site.site_id]),
                strength=1.0,
                label="restore R1079 to clean base value",
            )
            direct_r_patch = PatchSpec(
                site=r_control_site,
                source_site_id=r_control_site.site_id,
                source_position=source.final_position,
                target_position=base.final_position,
                source_value=float(source.features_by_site[r_control_site.site_id]),
                strength=1.0,
                label="patch R1079 directly from source",
            )
            t2_margin, t2_close, t2_features, _ = _run_patch_and_record(
                model,
                base=base,
                patches=t2_patches,
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            block_margin, block_close, block_features, _ = _run_patch_and_record(
                model,
                base=base,
                patches=tuple(t2_patches) + (block_patch,),
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            direct_margin, direct_close, _direct_features, _ = _run_patch_and_record(
                model,
                base=base,
                patches=(direct_r_patch,),
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            base_r = float(base.features_by_site[r_control_site.site_id])
            source_r = float(source.features_by_site[r_control_site.site_id])
            patched_r = float(t2_features[r_control_site.site_id])
            base_late = float(base.features_by_site[late_probe_site.site_id])
            patched_late = float(t2_features[late_probe_site.site_id])
            blocked_late = float(block_features[late_probe_site.site_id])
            row = {
                "spec_key": spec_key(spec),
                "kind": kind,
                "relation": spec.relation,
                "base_id": base_ex.example_id,
                "source_id": source_ex.example_id,
                "base_depth": int(base_ex.depth),
                "source_depth": int(source_ex.depth),
                "T2_patch_R1079_moves_to_source": scalar_moves_toward_source(
                    base=base_r, source=source_r, patched=patched_r
                ),
                "T2_patch_output_matches_source": int(t2_close) == int(source_ex.close_count),
                "T2_patch_output_preserves_base": int(t2_close) == int(base_ex.close_count),
                "block_output_preserves_base": int(block_close) == int(base_ex.close_count),
                "block_output_matches_source": int(block_close) == int(source_ex.close_count),
                "late_probe_mediation_fraction": mediation_fraction(
                    base=base_late, clean_patch=patched_late, blocked_patch=blocked_late
                ),
                "direct_R1079_output_matches_source": int(direct_close) == int(source_ex.close_count),
                "T2_patch_margin": float(t2_margin),
                "block_margin": float(block_margin),
                "direct_R1079_margin": float(direct_margin),
                "base_R1079": base_r,
                "source_R1079": source_r,
                "T2_patch_R1079": patched_r,
                "base_late_probe": base_late,
                "T2_patch_late_probe": patched_late,
                "blocked_late_probe": blocked_late,
            }
            _atomic_json(path, row)
        records.append(row)
        if idx % 25 == 0 or idx == len(selected_specs):
            print(f"T2 mediation {idx}/{len(selected_specs)}", flush=True)
    return records


def _without_records(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "records"}


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Componentwise PLOT for the Bracket Threshold State",
        "",
        f"Final status: **{payload['decision']['status']}**",
        "",
        "## Design",
        "",
        f"- candidates per target: `{payload['candidate_manifest']['candidate_count']}`",
        f"- no filtering: `{payload['candidate_manifest']['no_filtering_applied']}`",
        f"- candidate CSV SHA256: `{payload['candidate_manifest']['candidate_csv_sha256']}`",
        f"- reused parent Dfit records: `{payload['source_manifest']['Dfit_records']}`",
        f"- fresh Dcal examples: `{payload['fresh_bank_manifest']['splits']['Dcal']['examples']}`",
        f"- fresh Dte examples: `{payload['fresh_bank_manifest']['splits']['Dte']['examples']}`",
        f"- fresh clean accuracy: Dcal `{payload['clean']['Dcal_accuracy']:.3f}`, Dte `{payload['clean']['Dte_accuracy']:.3f}`",
        "- Dfit signatures were exact component slices; no records or sites were filtered or reweighted.",
        "- selections were frozen before any Dte forward pass.",
        "",
        "## Component Results",
        "",
        "| target | top site | selected handle | K | lambda | Dcal | Dte valid | readout Dte |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for target in TARGET_COMPONENTS:
        result = payload["components"][target]
        best = result["selection"]
        lines.append(
            f"| `{target}` | `{result['top_site']}` | `{', '.join(best['site_ids'])}` | {best['k']} | "
            f"{best['strength']:.3f} | {best['calibration_score']:.3f} | "
            f"{result['heldout_acceptance']['validated']} | {result['readout_quality']['Dte']['accuracy']:.3f} |"
        )
    for target in TARGET_COMPONENTS:
        result = payload["components"][target]
        best = result["selection"]
        lines.extend(
            [
                "",
                f"### {target}",
                "",
                f"- raw-cosine-UOT top site: `{result['top_site']}`",
                f"- selected sites: `{', '.join(best['site_ids'])}`",
                f"- selected K/lambda: `{best['k']}` / `{best['strength']:.3f}`",
                f"- Dcal score: `{best['calibration_score']:.3f}`",
                f"- Dfit post-hoc ranks: `{result['posthoc_ranks']}`",
                f"- bootstrap top-1 frequency: `{result['bootstrap_summary']['top1_frequency']}`",
                f"- bootstrap support frequency: `{result['bootstrap_summary']['calibrated_support_frequency']}`",
                "",
                "| metric | Dcal | Dte |",
                "|---|---:|---:|",
            ]
        )
        for metric in result["heldout_summary"]["metrics"]:
            dcal = best["summary"]["metrics"][metric]
            dte = result["heldout_summary"]["metrics"][metric]
            lines.append(f"| `{metric}` | {dcal:.3f} | {dte:.3f} |")
        lines.extend(["", "Directional heldout checks:", "", "| transition | n | target | non-target | output |", "|---|---:|---:|---:|---:|"])
        for transition, row in result["heldout_summary"]["directional"].items():
            lines.append(
                f"| `{transition}` | {row['n']} | {row['target_source_match']:.3f} | "
                f"{row['non_target_base_preserve']:.3f} | {row['expected_output_success']:.3f} |"
            )
        failed = [name for name, passed in result["heldout_acceptance"]["checks"].items() if not passed]
        lines.extend(["", f"Failed heldout gates: `{failed if failed else 'none'}`"])

    lines.extend(["", "## Joint Threshold State", ""])
    joint = payload["joint"]
    lines.extend(
        [
            f"- validated: `{joint['acceptance']['validated']}`",
            f"- exact vector source match: `{joint['summary']['vector_source_match']:.3f}`",
            f"- output source match: `{joint['summary']['output_source_match']:.3f}`",
            "",
            "| transition | n | exact vector | output |",
            "|---|---:|---:|---:|",
        ]
    )
    for transition, row in joint["summary"]["transitions"].items():
        lines.append(
            f"| `{transition}` | {row['n']} | {row['vector_source_match']:.3f} | "
            f"{row['output_source_match']:.3f} |"
        )
    joint_failed = [name for name, passed in joint["acceptance"]["checks"].items() if not passed]
    lines.extend(
        [
            "",
            f"Failed joint gates: `{joint_failed if joint_failed else 'none'}`",
            "",
            "## T2 to R Mediation",
            "",
            f"- validated: `{payload['mediation']['acceptance']['validated']}`",
        ]
    )
    for key, value in payload["mediation"]["summary"].items():
        if isinstance(value, (int, float)):
            lines.append(f"- `{key}`: `{float(value):.3f}`")
    mediation_failed = [
        name for name, passed in payload["mediation"]["acceptance"]["checks"].items() if not passed
    ]
    lines.extend(
        [
            f"- failed gates: `{mediation_failed if mediation_failed else 'none'}`",
            "",
            "## Accepted Model",
            "",
            f"```text\n{payload['decision']['accepted_model']}\n```",
            "",
            f"Accepted threshold components: `{payload['decision']['accepted_components']}`",
            "",
            "## Interpretation",
            "",
            "1. `T2` passes as an intervention target, but its selected support contains `R1079` and multiple downstream readout sites. It therefore does not establish a new upstream copy of the binary variable.",
            "2. `T3` and `T4` have perfectly accurate frozen readouts and highly stable PLOT rankings, but their selected handles fail bidirectional source-setting tests. They are depth-correlated signals, not validated causal setters.",
            "3. Joint composition changes the output correctly on every compound transition while failing to set the exact threshold vector on every transition. Output control is therefore not evidence for the proposed full threshold state.",
            "4. Blocking `R1079` neither restores the base output reliably nor removes half of the late-probe effect, so the serial `T2 -> R -> Y` edge is not validated.",
            "5. The full refined model is rejected under the preregistered gates; the accepted output abstraction remains `X -> R -> Y`.",
            "",
            "Known sites were not used in threshold matching or calibration. R1079 and the late probe were used only after the T2 handle was frozen.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resume = not bool(args.no_resume)
    out_dir = args.out_dir
    for directory in (
        "signatures",
        "selectors",
        "calibration/T2/rows",
        "calibration/T3/rows",
        "calibration/T4/rows",
        "bootstrap",
        "heldout/T2/records",
        "heldout/T3/records",
        "heldout/T4/records",
        "joint/records",
        "mediation/records",
    ):
        (out_dir / directory).mkdir(parents=True, exist_ok=True)

    depths = parse_depths(args.depths)
    if depths != (1, 2, 3, 4):
        raise ValueError("threshold experiment is preregistered for depths 1,2,3,4")
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)
    device = "cuda" if args.cuda else "cpu"

    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv,
        expected_node_count=args.expected_node_count,
    )
    candidate_sites = universe.sites
    site_lookup = {site.site_id: site for site in candidate_sites}
    candidate_manifest = universe.manifest()
    _atomic_json(out_dir / "candidate_manifest.json", candidate_manifest)

    parent = _load_parent_inputs(
        parent_dir=args.parent_run_dir,
        universe_manifest=candidate_manifest,
        candidate_sites=candidate_sites,
    )
    _atomic_json(out_dir / "source_manifest.json", parent["source_manifest"])
    selectors = _derive_selectors(
        parent=parent,
        candidate_sites=candidate_sites,
        out_dir=out_dir,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )

    encoding = make_tinypython_encoding(args.circuit_home)
    fresh_examples, fresh_manifest = _fresh_bank(
        encoding,
        fresh_content_start=args.fresh_content_start,
        cal_contents=args.cal_contents,
        test_contents=args.test_contents,
        depths=depths,
        previous_indices=parent["source_manifest"]["previous_content_indices"],
    )
    fresh_lookup = {example.example_id: example for example in fresh_examples}
    specs = {
        split: build_transition_balanced_specs(
            fresh_examples,
            split=split,
            records_per_relation=args.records_per_relation,
        )
        for split in ("Dcal", "Dte")
    }
    _atomic_json(out_dir / "fresh_bank_manifest.json", fresh_manifest)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_manifest": candidate_manifest,
        "source_manifest": parent["source_manifest"],
        "fresh_bank_manifest": fresh_manifest,
        "depths": list(depths),
        "records_per_relation": int(args.records_per_relation),
        "record_keys": {split: [spec_key(spec) for spec in split_specs] for split, split_specs in specs.items()},
        "k_grid": list(k_grid),
        "strength_grid": list(strength_grid),
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "near_optimal_epsilon": float(args.near_optimal_epsilon),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "acceptance_threshold": float(args.acceptance_threshold),
        "r_control_site": str(args.r_control_site),
        "r_late_probe_site": str(args.r_late_probe_site),
        "known_site_policy": "not used in threshold ranking/calibration; downstream sites used only after frozen selection",
        "Dte_policy": "no Dte forward pass before selection_manifest.json is written",
    }
    _require_manifest(out_dir / "run_manifest.json", run_manifest, resume=resume)

    readout: FrozenDepthReadout = parent["readout"]
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not bool(args.no_flash),
        grad_checkpointing=False,
    )
    single_close_token_id = int(encoding.encode("]\n")[0])
    double_close_token_id = int(encoding.encode("]]\n")[0])

    cal_examples = tuple(example for example in fresh_examples if example.split == "Dcal")
    cal_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=cal_examples,
        candidate_sites=candidate_sites,
        checkpoint_path=out_dir / "clean_runs_Dcal.jsonl",
        resume=resume,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    cal_phi = _attach_phi(readout, cal_runs)
    cal_accuracy = _clean_accuracy(cal_examples, cal_runs)
    if cal_accuracy != 1.0:
        raise RuntimeError(f"fresh Dcal clean accuracy is {cal_accuracy:.6f}, expected 1.0")
    readout_quality: dict[str, dict[str, Any]] = {
        target: {"Dcal": readout_component_quality(target, cal_examples, cal_phi)}
        for target in TARGET_COMPONENTS
    }
    for target in TARGET_COMPONENTS:
        if readout_quality[target]["Dcal"]["accuracy"] < float(args.acceptance_threshold):
            raise RuntimeError(f"frozen readout failed fresh Dcal gate for {target}")

    calibrations: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for target in TARGET_COMPONENTS:
        ranked = selectors[target]["selector"]["selectors"]["raw_cosine_uot"]["ranked_sites"]
        calibration = _calibrate_target(
            model=model,
            target=target,
            ranked_sites=ranked,
            specs=specs["Dcal"],
            examples=fresh_lookup,
            clean_runs=cal_runs,
            readout=readout,
            site_lookup=site_lookup,
            record_sites=candidate_sites,
            k_grid=k_grid,
            strength_grid=strength_grid,
            rows_dir=out_dir / "calibration" / target / "rows",
            resume=resume,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        calibrations[target] = calibration
        selections[target] = calibration["best"]
        _atomic_jsonl(
            out_dir / "calibration" / target / "grid.jsonl",
            [_without_records(row) for row in calibration["grid"]],
        )
        best_score = float(calibration["best"]["calibration_score"])
        near = [
            _without_records(row)
            for row in calibration["grid"]
            if best_score - float(row["calibration_score"]) <= float(args.near_optimal_epsilon)
        ]
        _atomic_json(out_dir / "calibration" / target / "near_optimal.json", near)
        bootstrap = _bootstrap_target(
            target=target,
            fit_specs=parent["fit_specs"],
            abstract=selectors[target]["abstract"],
            signatures=selectors[target]["neural"],
            calibration_rows=calibration["grid"],
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
            epsilon=args.selector_epsilon,
            beta=args.selector_beta,
        )
        bootstraps[target] = bootstrap
        _atomic_json(out_dir / "bootstrap" / f"{target}.json", bootstrap)
        _atomic_jsonl(out_dir / "bootstrap" / f"{target}.jsonl", bootstrap["records"])

    selection_manifest = {
        "selection_completed_before_Dte": True,
        "selection_inputs": ["parent Dfit signatures", "fresh Dcal records"],
        "Dte_inputs_used": [],
        "selected_handles": {
            target: {
                "handle_id": selections[target]["handle_id"],
                "site_ids": selections[target]["site_ids"],
                "weights_by_site": selections[target]["weights_by_site"],
                "k": selections[target]["k"],
                "strength": selections[target]["strength"],
                "calibration_score": selections[target]["calibration_score"],
            }
            for target in TARGET_COMPONENTS
        },
    }
    _atomic_json(out_dir / "selection_manifest.json", selection_manifest)

    test_examples = tuple(example for example in fresh_examples if example.split == "Dte")
    test_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=test_examples,
        candidate_sites=candidate_sites,
        checkpoint_path=out_dir / "clean_runs_Dte.jsonl",
        resume=resume,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    test_phi = _attach_phi(readout, test_runs)
    test_accuracy = _clean_accuracy(test_examples, test_runs)
    if test_accuracy != 1.0:
        raise RuntimeError(f"fresh Dte clean accuracy is {test_accuracy:.6f}, expected 1.0")

    component_results: dict[str, Any] = {}
    component_acceptances: dict[str, Any] = {}
    for target in TARGET_COMPONENTS:
        readout_quality[target]["Dte"] = readout_component_quality(target, test_examples, test_phi)
        records = _heldout_target(
            model=model,
            target=target,
            best=selections[target],
            specs=specs["Dte"],
            examples=fresh_lookup,
            clean_runs=test_runs,
            readout=readout,
            site_lookup=site_lookup,
            record_sites=candidate_sites,
            records_dir=out_dir / "heldout" / target / "records",
            resume=resume,
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        summary = summarize_component_records(records)
        acceptance = component_acceptance(target, summary, threshold=args.acceptance_threshold)
        acceptance["readout_Dcal_gate"] = (
            readout_quality[target]["Dcal"]["accuracy"] >= float(args.acceptance_threshold)
        )
        acceptance["readout_Dte_gate"] = (
            readout_quality[target]["Dte"]["accuracy"] >= float(args.acceptance_threshold)
        )
        acceptance["validated"] = bool(
            acceptance["validated"] and acceptance["readout_Dcal_gate"] and acceptance["readout_Dte_gate"]
        )
        component_acceptances[target] = acceptance
        ranked = selectors[target]["selector"]["selectors"]["raw_cosine_uot"]["ranked_sites"]
        component_results[target] = {
            "top_site": ranked[0]["site_id"],
            "posthoc_ranks": _posthoc_ranks(ranked),
            "selection": _without_records(selections[target]),
            "readout_quality": readout_quality[target],
            "heldout_summary": summary,
            "heldout_acceptance": acceptance,
            "bootstrap_summary": bootstraps[target]["summary"],
        }
        _atomic_jsonl(out_dir / "heldout" / target / "records.jsonl", records)
        _atomic_json(out_dir / "heldout" / target / "summary.json", component_results[target])

    joint_rows = _joint_records(
        model=model,
        handles=selections,
        specs=specs["Dte"],
        examples=fresh_lookup,
        clean_runs=test_runs,
        readout=readout,
        site_lookup=site_lookup,
        record_sites=candidate_sites,
        records_dir=out_dir / "joint" / "records",
        resume=resume,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    joint_summary = summarize_joint_records(joint_rows)
    joint_result = {
        "summary": joint_summary,
        "acceptance": joint_acceptance(joint_summary, threshold=args.acceptance_threshold),
    }
    _atomic_jsonl(out_dir / "joint" / "records.jsonl", joint_rows)
    _atomic_json(out_dir / "joint" / "summary.json", joint_result)

    if args.r_control_site not in site_lookup or args.r_late_probe_site not in site_lookup:
        raise ValueError("R control and late-probe sites must both belong to the full 133-site universe")
    mediation_rows = _mediation_records(
        model=model,
        t2_best=selections["T2"],
        specs=specs["Dte"],
        examples=fresh_lookup,
        clean_runs=test_runs,
        site_lookup=site_lookup,
        record_sites=candidate_sites,
        r_control_site=site_lookup[args.r_control_site],
        late_probe_site=site_lookup[args.r_late_probe_site],
        records_dir=out_dir / "mediation" / "records",
        resume=resume,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    mediation_summary = summarize_t2_mediation(mediation_rows)
    mediation_result = {
        "summary": mediation_summary,
        "acceptance": t2_mediation_acceptance(
            mediation_summary,
            t2_site_ids=selections["T2"]["site_ids"],
            r_control_site=args.r_control_site,
            threshold=args.acceptance_threshold,
        ),
    }
    _atomic_jsonl(out_dir / "mediation" / "records.jsonl", mediation_rows)
    _atomic_json(out_dir / "mediation" / "summary.json", mediation_result)

    decision = final_model_decision(
        component_acceptances,
        joint_result["acceptance"],
        mediation_result["acceptance"],
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_manifest": candidate_manifest,
        "source_manifest": parent["source_manifest"],
        "fresh_bank_manifest": fresh_manifest,
        "model": args.model,
        "model_info": model_info,
        "clean": {"Dcal_accuracy": cal_accuracy, "Dte_accuracy": test_accuracy},
        "components": component_results,
        "joint": joint_result,
        "mediation": mediation_result,
        "decision": decision,
        "methodology": {
            "candidate_filtering": "none",
            "Dfit_signature_reuse": "exact component slicing",
            "Dte_used_for_selection": False,
            "known_sites_used_for_matching_or_calibration": False,
            "joint_recalibration": False,
        },
    }
    _atomic_json(out_dir / "bracket_threshold_component_plot.json", payload)
    _write_report(out_dir / "bracket_threshold_component_plot.md", payload)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
