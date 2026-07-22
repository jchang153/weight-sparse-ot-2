from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_d_large_bank_frozen_readout import (
    POSTHOC_SITE_IDS,
    SCHEMA_VERSION,
    abstract_depth_signature,
    build_transition_balanced_specs,
    calibration_score,
    canonical_sha256,
    depth_acceptance,
    depth_from_phi,
    neural_depth_signature_component,
    ordered_transition_counts,
    posthoc_ranks,
    readout_quality_payload,
    relation_counts,
    resample_indices_within_relation,
    slice_signature,
    spec_key,
    summarize_validation_records,
    support_overlap,
)
from .bracket_d_rich_signatures import (
    FrozenDepthReadout,
    depth_target_matrix,
    fit_ridge_readout,
    load_full_localized_candidate_universe,
    selector_payload_from_signatures,
    weights_from_ranked,
)
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .bracket_progressive_model_discovery import bank_manifest, build_discovery_bank
from .run_bracket_d_rich_signature_experiments import (
    CleanRun,
    _bracket_margin,
    _feature_dict,
    _feature_vector,
    _hook_regex,
    _run_weighted_patch,
    _sign_from_margin,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_CANDIDATE_CSV = Path(
    "eval/openai_sparse_plot/bracket_counting_inventory_csp_yolo2_prune_v4/string_closing_circuit_nodes.csv"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_d_large_bank_frozen_readout_20260710")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run large-bank full-133 frozen-readout PLOT for bracket depth D.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--candidate-node-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--expected-node-count", type=int, default=133)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--contents", type=int, default=96)
    parser.add_argument("--fit-contents", type=int, default=48)
    parser.add_argument("--cal-contents", type=int, default=24)
    parser.add_argument("--test-contents", type=int, default=24)
    parser.add_argument("--records-per-relation", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--readout-valid-threshold", type=float, default=0.90)
    parser.add_argument("--k-grid", default="1,2,3,5,8")
    parser.add_argument("--strength-grid", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--near-optimal-epsilon", type=float, default=0.02)
    parser.add_argument("--bootstrap-reps", type=int, default=50)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _parse_int_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("empty K grid")
    return values


def _parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("empty strength grid")
    return values


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temp.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _require_manifest(path: Path, payload: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists() and resume:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"checkpoint metadata mismatch: {path}")
    else:
        _atomic_json(path, payload)


def _clean_record(run: CleanRun) -> dict[str, Any]:
    return {
        "example_id": run.example_id,
        "final_position": int(run.final_position),
        "margin": float(run.margin),
        "predicted_close_count": int(run.predicted_close_count),
        "correct": bool(run.correct),
        "feature_vector": [float(value) for value in run.feature_vector],
    }


def _clean_run_from_record(
    record: Mapping[str, Any],
    *,
    example: MultiDepthBracketExample,
    candidate_sites: Sequence[ChannelSite],
    device: str,
) -> CleanRun:
    vector = tuple(float(value) for value in record["feature_vector"])
    features = {site.site_id: vector[idx] for idx, site in enumerate(candidate_sites)}
    token_ids = torch.tensor(example.token_ids, dtype=torch.long, device=device).unsqueeze(0)
    return CleanRun(
        example_id=example.example_id,
        token_ids=token_ids,
        final_position=int(record["final_position"]),
        margin=float(record["margin"]),
        predicted_close_count=int(record["predicted_close_count"]),
        correct=bool(record["correct"]),
        features_by_site=features,
        feature_vector=vector,
        depth_phi=(),
    )


def _collect_or_resume_clean_runs(
    *,
    model: Any,
    examples: Sequence[MultiDepthBracketExample],
    candidate_sites: Sequence[ChannelSite],
    checkpoint_path: Path,
    resume: bool,
    device: str,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, CleanRun]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    lookup = {example.example_id: example for example in examples}
    records = {str(row["example_id"]): row for row in (_load_jsonl(checkpoint_path) if resume else [])}
    runs = {
        example_id: _clean_run_from_record(
            record,
            example=lookup[example_id],
            candidate_sites=candidate_sites,
            device=device,
        )
        for example_id, record in records.items()
        if example_id in lookup
    }
    if runs:
        print(f"resuming clean runs: loaded {len(runs)}/{len(examples)}", flush=True)
    regex = _hook_regex(candidate_sites)
    for idx, example in enumerate(examples, start=1):
        if example.example_id in runs:
            continue
        token_ids = torch.tensor(example.token_ids, dtype=torch.long, device=device).unsqueeze(0)
        final_position = int(token_ids.shape[1] - 1)
        with torch.no_grad():
            with hook_recorder(regex=regex) as context:
                logits, _, _ = model(token_ids)
        cache = {key: value.detach().cpu() for key, value in context.items()}
        features = _feature_dict(cache, candidate_sites, final_position)
        vector = _feature_vector(features, candidate_sites)
        margin = _bracket_margin(
            logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        predicted = _sign_from_margin(margin)
        run = CleanRun(
            example_id=example.example_id,
            token_ids=token_ids,
            final_position=final_position,
            margin=margin,
            predicted_close_count=predicted,
            correct=predicted == example.close_count,
            features_by_site=features,
            feature_vector=vector,
            depth_phi=(),
        )
        runs[example.example_id] = run
        _append_jsonl(checkpoint_path, _clean_record(run))
        if idx % 32 == 0 or idx == len(examples):
            print(f"clean runs {idx}/{len(examples)}", flush=True)
    expected = {example.example_id for example in examples}
    if set(runs) != expected:
        raise ValueError(f"incomplete clean runs: missing={sorted(expected - set(runs))[:5]}")
    return runs


def _fit_or_load_readout(
    *,
    path: Path,
    examples: Sequence[MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    candidate_sites: Sequence[ChannelSite],
    alpha: float,
    max_depth: int,
    metadata_hash: str,
    resume: bool,
) -> FrozenDepthReadout:
    if path.exists() and resume:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["metadata_hash"] != metadata_hash:
            raise ValueError(f"readout metadata mismatch: {path}")
        return FrozenDepthReadout(
            weights=torch.tensor(payload["readout"]["weights"], dtype=torch.float32),
            alpha=float(payload["readout"]["alpha"]),
            feature_site_ids=tuple(payload["readout"]["feature_site_ids"]),
        )
    fit_examples = [example for example in examples if example.split == "Dfit"]
    x = torch.tensor([clean_runs[example.example_id].feature_vector for example in fit_examples], dtype=torch.float32)
    y = depth_target_matrix(fit_examples, max_depth=max_depth)
    readout = fit_ridge_readout(x, y, alpha=float(alpha), feature_site_ids=[site.site_id for site in candidate_sites])
    _atomic_json(path, {"metadata_hash": metadata_hash, "fit_split": "Dfit", "readout": readout.to_json()})
    return readout


def _vector_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _validation_record(
    *,
    model: Any,
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
    base_example = examples[spec.base_id]
    source_example = examples[spec.source_id]
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
    patched_phi = tuple(readout.predict(vector))
    base_distance = _vector_distance(base.depth_phi, source.depth_phi)
    patched_distance = _vector_distance(patched_phi, source.depth_phi)
    effect_fraction = (
        _vector_distance(patched_phi, base.depth_phi) / base_distance if base_distance > 1e-6 else float("nan")
    )
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
        "patched_close_count": int(close_count),
        "patched_margin": float(margin),
        "base_depth_phi": list(base.depth_phi),
        "source_depth_phi": list(source.depth_phi),
        "patched_depth_phi": list(patched_phi),
        "predicted_depth": depth_from_phi(patched_phi),
        "depth_matches_source": depth_from_phi(patched_phi) == int(source_example.depth),
        "depth_matches_base": depth_from_phi(patched_phi) == int(base_example.depth),
        "depth_moves_toward_source": patched_distance + 1e-6 < base_distance,
        "depth_effect_fraction": float(effect_fraction),
        "output_matches_source": int(close_count) == int(source_example.close_count),
        "output_preserves_base": int(close_count) == int(base_example.close_count),
    }


def _build_or_resume_signatures(
    *,
    model: Any,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    candidate_sites: Sequence[ChannelSite],
    checkpoint_path: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, tuple[float, ...]]:
    loaded = _load_jsonl(checkpoint_path) if resume else []
    signatures = {str(row["site_id"]): tuple(float(value) for value in row["signature"]) for row in loaded}
    if signatures:
        print(f"resuming signatures: loaded {len(signatures)}/{len(candidate_sites)}", flush=True)
    for idx, site in enumerate(candidate_sites, start=1):
        if site.site_id in signatures:
            continue
        values: list[float] = []
        for spec in specs:
            base = clean_runs[spec.base_id]
            source = clean_runs[spec.source_id]
            _margin, _close_count, _features, vector = _run_weighted_patch(
                model,
                base=base,
                source=source,
                patch_sites=(site,),
                weights_by_site={site.site_id: 1.0},
                strength=1.0,
                record_sites=candidate_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            patched_phi = readout.predict(vector)
            values.extend(neural_depth_signature_component(base.depth_phi, patched_phi))
        signature = tuple(values)
        signatures[site.site_id] = signature
        _append_jsonl(checkpoint_path, {"site_id": site.site_id, "signature": list(signature)})
        print(f"signature {idx}/{len(candidate_sites)} {site.site_id}", flush=True)
    expected = {site.site_id for site in candidate_sites}
    if set(signatures) != expected:
        raise ValueError(f"incomplete signatures: missing={sorted(expected - set(signatures))[:5]}")
    return signatures


def _calibrate(
    *,
    model: Any,
    ranked_sites: Sequence[Mapping[str, Any]],
    site_lookup: Mapping[str, ChannelSite],
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    readout: FrozenDepthReadout,
    record_sites: Sequence[ChannelSite],
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    rows_dir: Path,
    resume: bool,
    single_close_token_id: int,
    double_close_token_id: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for k in k_grid:
        weights = weights_from_ranked(ranked_sites, k=int(k))
        patch_sites = tuple(site_lookup[site_id] for site_id in weights)
        for strength in strength_grid:
            handle_id = f"C_D_frozen_readout_top{k}_lambda{float(strength):g}"
            path = rows_dir / f"{handle_id}.json"
            if path.exists() and resume:
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                records = [
                    _validation_record(
                        model=model,
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
                    for spec in specs
                ]
                summary = summarize_validation_records(records)
                row = {
                    "handle_id": handle_id,
                    "k": int(k),
                    "strength": float(strength),
                    "site_ids": list(weights),
                    "weights_by_site": weights,
                    "summary": summary,
                    "calibration_score": calibration_score(summary),
                    "support_overlap": support_overlap(list(weights)),
                    "records": records,
                }
                _atomic_json(path, row)
            rows.append(row)
        print(f"calibrated K={k}", flush=True)
    best = sorted(
        rows,
        key=lambda row: (
            -float(row["calibration_score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
        ),
    )[0]
    return {"grid": rows, "best": best}


def _bootstrap(
    *,
    specs_fit: Sequence[MultiDepthResamplingSpec],
    abstract: Sequence[float],
    signatures: Mapping[str, Sequence[float]],
    calibration_rows: Sequence[Mapping[str, Any]],
    reps: int,
    seed: int,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    records = []
    for rep in range(int(reps)):
        rng = random.Random(int(seed) + rep)
        fit_indices = resample_indices_within_relation(specs_fit, rng=rng)
        fit_abstract = slice_signature(abstract, fit_indices)
        fit_neural = {site_id: slice_signature(signature, fit_indices) for site_id, signature in signatures.items()}
        selector = selector_payload_from_signatures(
            variant="C_D_frozen_readout",
            abstract=fit_abstract,
            neural_by_site=fit_neural,
            epsilon=float(epsilon),
            beta=float(beta),
        )
        ranked = selector["selectors"]["raw_cosine_uot"]["ranked_sites"]

        relation_indices: dict[str, list[int]] = defaultdict(list)
        exemplar_records = calibration_rows[0]["records"]
        for idx, record in enumerate(exemplar_records):
            relation_indices[str(record["relation"])].append(idx)
        sampled_cal_indices: list[int] = []
        for relation in sorted(relation_indices):
            indices = relation_indices[relation]
            sampled_cal_indices.extend(rng.choice(indices) for _ in indices)
        rescored = []
        for row in calibration_rows:
            sampled_records = [row["records"][idx] for idx in sampled_cal_indices]
            score = calibration_score(summarize_validation_records(sampled_records))
            rescored.append({**row, "bootstrap_score": score})
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
    top1 = Counter(record["top1"] for record in records)
    calibrated = Counter(site_id for record in records for site_id in record["calibrated_site_ids"])
    return {
        "records": records,
        "summary": {
            "reps": int(reps),
            "top1_frequency": dict(top1.most_common()),
            "calibrated_support_frequency": dict(calibrated.most_common()),
            "posthoc_top1_frequency": {site_id: int(top1.get(site_id, 0)) for site_id in POSTHOC_SITE_IDS},
            "posthoc_calibrated_frequency": {site_id: int(calibrated.get(site_id, 0)) for site_id in POSTHOC_SITE_IDS},
        },
    }


def _heldout_records(
    *,
    model: Any,
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
    patch_sites = tuple(site_lookup[site_id] for site_id in best["site_ids"])
    records = []
    for idx, spec in enumerate(specs, start=1):
        digest = canonical_sha256({"handle": best["handle_id"], "spec": spec_key(spec)})[:20]
        path = records_dir / f"{idx:04d}_{digest}.json"
        if path.exists() and resume:
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = _validation_record(
                model=model,
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
            _atomic_json(path, record)
        records.append(record)
        if idx % 50 == 0 or idx == len(specs):
            print(f"heldout records {idx}/{len(specs)}", flush=True)
    return records


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Large-Bank Frozen-Readout PLOT for Bracket Depth",
        "",
        f"Final status: **{payload['final_status']}**",
        "",
        "## Design",
        "",
        f"- candidate count: `{payload['candidate_manifest']['candidate_count']}`",
        f"- no filtering: `{payload['candidate_manifest']['no_filtering_applied']}`",
        f"- total examples: `{payload['bank_manifest']['total_examples']}`",
        f"- records per relation: `{payload['records_per_relation']}`",
        f"- clean accuracy: `{payload['clean']['accuracy']:.3f}`",
        "",
        "## Frozen Depth Readout",
        "",
    ]
    for split, quality in payload["readout_quality"].items():
        lines.append(
            f"- {split}: exact depth `{quality['exact_depth_accuracy']:.3f}`, "
            f"threshold macro `{quality['threshold_macro_accuracy']:.3f}`, "
            f"norm-D MAE `{quality['norm_depth_mae']:.3f}`"
        )
    if payload.get("selection"):
        selection = payload["selection"]
        heldout = payload["heldout"]
        lines.extend(
            [
                "",
                "## Selected Handle",
                "",
                f"- sites: `{', '.join(selection['site_ids'])}`",
                f"- K: `{selection['k']}`",
                f"- lambda: `{selection['strength']:.3f}`",
                f"- Dcal score: `{selection['calibration_score']:.3f}`",
                f"- Dte validated: `{heldout['acceptance']['D_validated']}`",
                f"- post-hoc ranks: `{payload['posthoc_ranks']}`",
                "",
                "## Heldout Metrics",
                "",
                "| metric | value |",
                "|---|---:|",
            ]
        )
        for key, value in sorted(heldout["summary"]["metrics"].items()):
            lines.append(f"| `{key}` | {float(value):.3f} |")
        lines.extend(["", "## Ordered Transition Checks", ""])
        for relation, transitions in heldout["summary"]["ordered_transitions"].items():
            lines.append(f"### `{relation}`")
            lines.extend(["", "| transition | n | depth match | output success |", "|---|---:|---:|---:|"])
            for transition, row in transitions.items():
                lines.append(
                    f"| `{transition}` | {row['n']} | {row['depth_source_match']:.3f} | "
                    f"{row['expected_output_success']:.3f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Known sites were used only for post-hoc recovery reporting and never for ranking or calibration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    resume = not bool(args.no_resume)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("signatures", "calibration/rows", "heldout/records", "bootstrap"):
        (out_dir / directory).mkdir(parents=True, exist_ok=True)

    depths = parse_depths(args.depths)
    k_grid = _parse_int_grid(args.k_grid)
    strength_grid = _parse_float_grid(args.strength_grid)
    device = "cuda" if args.cuda else "cpu"

    universe = load_full_localized_candidate_universe(
        args.candidate_node_csv,
        expected_node_count=args.expected_node_count,
    )
    candidate_sites = universe.sites
    site_lookup = {site.site_id: site for site in candidate_sites}
    _atomic_json(out_dir / "candidate_manifest.json", universe.manifest())

    encoding = make_tinypython_encoding(args.circuit_home)
    examples = build_discovery_bank(
        encoding,
        contents=args.contents,
        fit_contents=args.fit_contents,
        cal_contents=args.cal_contents,
        test_contents=args.test_contents,
        depths=depths,
    )
    examples_by_id = {example.example_id: example for example in examples}
    bank = bank_manifest(examples)
    _atomic_json(out_dir / "bank_manifest.json", bank)
    specs = {
        split: build_transition_balanced_specs(
            examples,
            split=split,
            records_per_relation=args.records_per_relation,
        )
        for split in ("Dfit", "Dcal", "Dte")
    }

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "candidate_manifest": universe.manifest(),
        "bank_manifest": bank,
        "depths": list(depths),
        "records_per_relation": int(args.records_per_relation),
        "record_keys": {split: [spec_key(spec) for spec in rows] for split, rows in specs.items()},
        "ridge_alpha": float(args.ridge_alpha),
        "readout_valid_threshold": float(args.readout_valid_threshold),
        "k_grid": list(k_grid),
        "strength_grid": list(strength_grid),
        "selector_epsilon": float(args.selector_epsilon),
        "selector_beta": float(args.selector_beta),
        "near_optimal_epsilon": float(args.near_optimal_epsilon),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_seed": int(args.bootstrap_seed),
        "known_sites_policy": "posthoc_only",
    }
    _require_manifest(out_dir / "run_manifest.json", run_manifest, resume=resume)
    metadata_hash = canonical_sha256(run_manifest)

    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not bool(args.no_flash),
        grad_checkpointing=False,
    )
    single_close_token_id = int(encoding.encode("]\n")[0])
    double_close_token_id = int(encoding.encode("]]\n")[0])

    clean_runs = _collect_or_resume_clean_runs(
        model=model,
        examples=examples,
        candidate_sites=candidate_sites,
        checkpoint_path=out_dir / "clean_runs.jsonl",
        resume=resume,
        device=device,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    readout = _fit_or_load_readout(
        path=out_dir / "frozen_depth_readout.json",
        examples=examples,
        clean_runs=clean_runs,
        candidate_sites=candidate_sites,
        alpha=args.ridge_alpha,
        max_depth=max(depths),
        metadata_hash=metadata_hash,
        resume=resume,
    )
    for run in clean_runs.values():
        run.depth_phi = tuple(readout.predict(run.feature_vector))

    features_by_example = {example_id: run.feature_vector for example_id, run in clean_runs.items()}
    quality = {
        split: readout_quality_payload(
            readout,
            examples,
            features_by_example,
            split=split,
            max_depth=max(depths),
        )
        for split in ("Dfit", "Dcal")
    }
    clean_accuracy = sum(1.0 for run in clean_runs.values() if run.correct) / len(clean_runs)
    dcal_eligible = quality["Dcal"]["exact_depth_accuracy"] >= float(args.readout_valid_threshold)

    base_payload: dict[str, Any] = {
        "model": args.model,
        "model_info": model_info,
        "candidate_manifest": universe.manifest(),
        "bank_manifest": bank,
        "records_per_relation": int(args.records_per_relation),
        "relation_counts": {split: relation_counts(rows) for split, rows in specs.items()},
        "ordered_transition_counts": {
            split: ordered_transition_counts(rows, examples_by_id) for split, rows in specs.items()
        },
        "clean": {"n": len(clean_runs), "accuracy": clean_accuracy},
        "readout_quality": quality,
        "readout_Dcal_eligible": dcal_eligible,
    }

    if not dcal_eligible:
        quality["Dte"] = readout_quality_payload(
            readout,
            examples,
            features_by_example,
            split="Dte",
            max_depth=max(depths),
        )
        payload = {
            **base_payload,
            "readout_quality": quality,
            "final_status": "experiment invalid because frozen D readout failed",
            "selection": None,
            "heldout": None,
            "posthoc_ranks": {},
            "interpretation": (
                "The frozen depth readout failed the predeclared Dcal exact-depth gate. "
                "No PLOT ranking or calibration was interpreted."
            ),
        }
        _atomic_json(out_dir / "readout_quality.json", quality)
        _atomic_json(out_dir / "bracket_d_large_bank_frozen_readout.json", payload)
        _write_report(out_dir / "bracket_d_large_bank_frozen_readout.md", payload)
        print(json.dumps({"final_status": payload["final_status"]}, indent=2), flush=True)
        return

    abstract = abstract_depth_signature(specs["Dfit"], examples_by_id, max_depth=max(depths))
    signatures = _build_or_resume_signatures(
        model=model,
        specs=specs["Dfit"],
        examples=examples_by_id,
        clean_runs=clean_runs,
        readout=readout,
        candidate_sites=candidate_sites,
        checkpoint_path=out_dir / "signatures" / "C_D_frozen_readout.jsonl",
        resume=resume,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    selector = selector_payload_from_signatures(
        variant="C_D_frozen_readout",
        abstract=abstract,
        neural_by_site=signatures,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    _atomic_json(out_dir / "selector.json", selector)
    ranked = selector["selectors"]["raw_cosine_uot"]["ranked_sites"]

    calibration = _calibrate(
        model=model,
        ranked_sites=ranked,
        site_lookup=site_lookup,
        specs=specs["Dcal"],
        examples=examples_by_id,
        clean_runs=clean_runs,
        readout=readout,
        record_sites=candidate_sites,
        k_grid=k_grid,
        strength_grid=strength_grid,
        rows_dir=out_dir / "calibration" / "rows",
        resume=resume,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    grid_without_records = [{key: value for key, value in row.items() if key != "records"} for row in calibration["grid"]]
    _atomic_json(out_dir / "calibration" / "grid.json", grid_without_records)
    _atomic_jsonl(out_dir / "calibration" / "grid.jsonl", grid_without_records)
    best = calibration["best"]
    best_score = float(best["calibration_score"])
    near = [
        {key: value for key, value in row.items() if key != "records"}
        for row in calibration["grid"]
        if best_score - float(row["calibration_score"]) <= float(args.near_optimal_epsilon)
    ]

    bootstrap = _bootstrap(
        specs_fit=specs["Dfit"],
        abstract=abstract,
        signatures=signatures,
        calibration_rows=calibration["grid"],
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
        epsilon=args.selector_epsilon,
        beta=args.selector_beta,
    )
    _atomic_json(out_dir / "bootstrap" / "ranking_and_calibration.json", bootstrap)
    _atomic_jsonl(out_dir / "bootstrap" / "ranking_and_calibration.jsonl", bootstrap["records"])

    heldout_records = _heldout_records(
        model=model,
        best=best,
        specs=specs["Dte"],
        examples=examples_by_id,
        clean_runs=clean_runs,
        readout=readout,
        site_lookup=site_lookup,
        record_sites=candidate_sites,
        records_dir=out_dir / "heldout" / "records" / str(best["handle_id"]),
        resume=resume,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    heldout_summary = summarize_validation_records(heldout_records)
    heldout_acceptance = depth_acceptance(heldout_summary, threshold=args.readout_valid_threshold)
    quality["Dte"] = readout_quality_payload(
        readout,
        examples,
        features_by_example,
        split="Dte",
        max_depth=max(depths),
    )
    dte_readout_valid = quality["Dte"]["exact_depth_accuracy"] >= float(args.readout_valid_threshold)
    if not dte_readout_valid:
        final_status = "experiment invalid because frozen D readout failed"
        interpretation = (
            "The Dcal readout gate passed, but the frozen readout failed exact depth accuracy on Dte. "
            "The selected neural handle cannot be interpreted as a validated D abstraction."
        )
    elif heldout_acceptance["D_validated"]:
        final_status = "D validated"
        interpretation = (
            "The full-133 PLOT search found a compact handle that transfers exact depth on every required relation "
            "while preserving or changing the output exactly as the D -> R -> Y model predicts."
        )
    else:
        final_status = "D not validated"
        interpretation = (
            "The frozen readout is valid, but the strict Dcal-selected handle fails one or more heldout exact-depth "
            "or ordered-transition gates. This supports a depth-sensitive pathway, not a validated full D variable."
        )

    heldout = {
        "summary": heldout_summary,
        "acceptance": heldout_acceptance,
        "records": heldout_records,
    }
    _atomic_json(out_dir / "heldout" / "selected_handle.json", heldout)
    _atomic_jsonl(out_dir / "heldout" / "selected_handle.jsonl", heldout_records)
    _atomic_json(out_dir / "readout_quality.json", quality)
    selection = {key: value for key, value in best.items() if key != "records"}
    payload = {
        **base_payload,
        "readout_quality": quality,
        "final_status": final_status,
        "selection": selection,
        "near_optimal_Dcal_support": near,
        "heldout": heldout,
        "posthoc_ranks": posthoc_ranks(ranked),
        "bootstrap": bootstrap,
        "interpretation": interpretation,
        "known_sites_policy": "posthoc_only",
    }
    _atomic_json(out_dir / "depth_transition_table.json", heldout_summary["ordered_transitions"])
    _atomic_json(out_dir / "bracket_d_large_bank_frozen_readout.json", payload)
    _write_report(out_dir / "bracket_d_large_bank_frozen_readout.md", payload)
    print(
        json.dumps(
            {
                "final_status": final_status,
                "selected_sites": selection["site_ids"],
                "Dcal_score": selection["calibration_score"],
                "Dte_validated": heldout_acceptance["D_validated"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
