from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch

from .bracket_d_rich_signatures import FrozenDepthReadout, depth_targets_for_example
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, relation_for_pair


EXPERIMENT_RELATIONS: tuple[str, ...] = (
    "same_D",
    "different_D_same_R",
    "different_D_different_R",
    "same_surface_different_active_context",
    "wrong_numeric_content",
    "wrong_tail_length",
)

POSTHOC_SITE_IDS: tuple[str, ...] = (
    "2.attn.resid_delta:1249",
    "3.attn.act_in:1249",
    "4.attn.act_in:1249",
    "4.attn.resid_delta:1079",
    "4.attn.q:1292",
    "7.mlp.act_in:1079",
    "7.mlp.post_act:4133",
    "7.mlp.resid_delta:2041",
)

TARGET_NAMES: tuple[str, ...] = ("norm_D", "D_ge_2", "D_ge_3", "D_ge_4")
SCHEMA_VERSION = 1


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def spec_key(spec: MultiDepthResamplingSpec) -> str:
    return f"{spec.relation}|{spec.base_id}|{spec.source_id}|{spec.wrong_variable or ''}"


def _transition_group(
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
) -> tuple[Any, ...]:
    base = examples[spec.base_id]
    source = examples[spec.source_id]
    if base.depth != source.depth:
        return (int(base.depth), int(source.depth))
    return (int(base.depth),)


def _round_robin_groups(
    groups: Mapping[tuple[Any, ...], Sequence[MultiDepthResamplingSpec]],
    limit: int,
) -> list[MultiDepthResamplingSpec]:
    queues = {key: list(rows) for key, rows in sorted(groups.items())}
    selected: list[MultiDepthResamplingSpec] = []
    while len(selected) < int(limit) and any(queues.values()):
        for key in sorted(queues):
            if queues[key] and len(selected) < int(limit):
                selected.append(queues[key].pop(0))
    return selected


def build_transition_balanced_specs(
    examples: Sequence[MultiDepthBracketExample],
    *,
    split: str,
    records_per_relation: int,
    relations: Sequence[str] = EXPERIMENT_RELATIONS,
) -> tuple[MultiDepthResamplingSpec, ...]:
    split_examples = sorted((ex for ex in examples if ex.split == split), key=lambda ex: ex.example_id)
    lookup = {ex.example_id: ex for ex in split_examples}
    selected: list[MultiDepthResamplingSpec] = []
    for relation in relations:
        groups: dict[tuple[Any, ...], list[MultiDepthResamplingSpec]] = defaultdict(list)
        for base in split_examples:
            for source in split_examples:
                spec = relation_for_pair(base, source, relation)
                if spec is not None:
                    groups[_transition_group(spec, lookup)].append(spec)
        rows = _round_robin_groups(groups, int(records_per_relation))
        if len(rows) != int(records_per_relation):
            raise ValueError(f"relation {relation} produced {len(rows)} records, expected {records_per_relation}")
        selected.extend(rows)
    return tuple(selected)


def relation_counts(specs: Iterable[MultiDepthResamplingSpec]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec.relation] += 1
    return dict(sorted(counts.items()))


def ordered_transition_counts(
    specs: Iterable[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for spec in specs:
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        transition = f"{base.depth}->{source.depth}"
        counts[spec.relation][transition] += 1
    return {relation: dict(sorted(rows.items())) for relation, rows in sorted(counts.items())}


def abstract_depth_signature(
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    *,
    max_depth: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for spec in specs:
        base = depth_targets_for_example(examples[spec.base_id], max_depth=max_depth)
        source = depth_targets_for_example(examples[spec.source_id], max_depth=max_depth)
        values.extend(float(source_value - base_value) for source_value, base_value in zip(source, base))
    return tuple(values)


def neural_depth_signature_component(
    base_phi: Sequence[float],
    patched_phi: Sequence[float],
) -> tuple[float, ...]:
    return tuple(float(patched - base) for patched, base in zip(patched_phi, base_phi))


def threshold_bits(phi: Sequence[float]) -> tuple[int, int, int]:
    return tuple(1 if float(value) >= 0.5 else 0 for value in phi[1:4])


def depth_from_phi(phi: Sequence[float]) -> int | None:
    bits = threshold_bits(phi)
    mapping = {
        (0, 0, 0): 1,
        (1, 0, 0): 2,
        (1, 1, 0): 3,
        (1, 1, 1): 4,
    }
    return mapping.get(bits)


def readout_quality_payload(
    readout: FrozenDepthReadout,
    examples: Sequence[MultiDepthBracketExample],
    features_by_example: Mapping[str, Sequence[float]],
    *,
    split: str,
    max_depth: int,
) -> dict[str, Any]:
    rows = [example for example in examples if example.split == split]
    predictions = [readout.predict(features_by_example[example.example_id]) for example in rows]
    exact = [depth_from_phi(phi) == int(example.depth) for phi, example in zip(predictions, rows)]
    threshold_accuracies: dict[str, float] = {}
    for idx, name in enumerate(("D_ge_2", "D_ge_3", "D_ge_4"), start=1):
        correct = []
        for phi, example in zip(predictions, rows):
            target = 1 if int(example.depth) >= idx + 1 else 0
            correct.append((1 if float(phi[idx]) >= 0.5 else 0) == target)
        threshold_accuracies[name] = mean(correct)
    norm_targets = [depth_targets_for_example(example, max_depth=max_depth)[0] for example in rows]
    mae = mean(abs(float(phi[0]) - float(target)) for phi, target in zip(predictions, norm_targets))
    return {
        "split": split,
        "n": len(rows),
        "exact_depth_accuracy": mean(exact),
        "threshold_macro_accuracy": mean(threshold_accuracies.values()),
        "threshold_accuracies": threshold_accuracies,
        "norm_depth_mae": mae,
    }


def mean(values: Iterable[Any]) -> float:
    rows = [float(value) for value in values]
    return float(sum(rows) / len(rows)) if rows else float("nan")


def safe_mean(values: Iterable[Any]) -> float:
    rows = [float(value) for value in values]
    rows = [value for value in rows if not math.isnan(value)]
    return float(sum(rows) / len(rows)) if rows else float("nan")


def summarize_validation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_relation[str(record["relation"])].append(record)

    def rows(relation: str) -> list[Mapping[str, Any]]:
        return by_relation.get(relation, [])

    same_surface = rows("same_surface_different_active_context")
    same_surface_same_r = [row for row in same_surface if row["base_close_count"] == row["source_close_count"]]
    same_surface_different_r = [row for row in same_surface if row["base_close_count"] != row["source_close_count"]]

    metrics = {
        "different_D_same_R_depth_source_match": mean(row["depth_matches_source"] for row in rows("different_D_same_R")),
        "different_D_same_R_output_preserve": mean(row["output_preserves_base"] for row in rows("different_D_same_R")),
        "different_D_different_R_depth_source_match": mean(row["depth_matches_source"] for row in rows("different_D_different_R")),
        "different_D_different_R_output_source_match": mean(row["output_matches_source"] for row in rows("different_D_different_R")),
        "same_D_depth_base_preserve": mean(row["depth_matches_base"] for row in rows("same_D")),
        "same_D_output_preserve": mean(row["output_preserves_base"] for row in rows("same_D")),
        "wrong_numeric_depth_base_preserve": mean(row["depth_matches_base"] for row in rows("wrong_numeric_content")),
        "wrong_numeric_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_numeric_content")),
        "wrong_tail_depth_base_preserve": mean(row["depth_matches_base"] for row in rows("wrong_tail_length")),
        "wrong_tail_output_preserve": mean(row["output_preserves_base"] for row in rows("wrong_tail_length")),
        "same_surface_same_R_depth_source_match": mean(row["depth_matches_source"] for row in same_surface_same_r),
        "same_surface_same_R_output_preserve": mean(row["output_preserves_base"] for row in same_surface_same_r),
        "same_surface_different_R_depth_source_match": mean(row["depth_matches_source"] for row in same_surface_different_r),
        "same_surface_different_R_output_source_match": mean(row["output_matches_source"] for row in same_surface_different_r),
    }

    transitions: dict[str, dict[str, Any]] = {}
    for relation in ("different_D_same_R", "different_D_different_R"):
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows(relation):
            grouped[f"{row['base_depth']}->{row['source_depth']}"] .append(row)
        transitions[relation] = {}
        for transition, transition_rows in sorted(grouped.items()):
            output_key = "output_preserves_base" if relation == "different_D_same_R" else "output_matches_source"
            transitions[relation][transition] = {
                "n": len(transition_rows),
                "depth_source_match": mean(row["depth_matches_source"] for row in transition_rows),
                "expected_output_success": mean(row[output_key] for row in transition_rows),
            }

    return {
        "records": len(records),
        "relation_counts": {relation: len(relation_rows) for relation, relation_rows in sorted(by_relation.items())},
        "metrics": metrics,
        "ordered_transitions": transitions,
        "diagnostics": {
            "different_depth_moves_toward_source": mean(
                row["depth_moves_toward_source"]
                for relation in ("different_D_same_R", "different_D_different_R")
                for row in rows(relation)
            ),
            "different_depth_effect_fraction": safe_mean(
                row["depth_effect_fraction"]
                for relation in ("different_D_same_R", "different_D_different_R")
                for row in rows(relation)
            ),
        },
    }


def calibration_score(summary: Mapping[str, Any]) -> float:
    metrics = summary["metrics"]
    return mean(float(metrics[key]) for key in sorted(metrics))


def depth_acceptance(
    summary: Mapping[str, Any],
    *,
    threshold: float = 0.90,
) -> dict[str, Any]:
    metrics = summary["metrics"]
    aggregate_checks = {key: float(value) >= float(threshold) for key, value in metrics.items()}
    transition_checks: dict[str, bool] = {}
    for relation, transitions in summary["ordered_transitions"].items():
        for transition, row in transitions.items():
            transition_checks[f"{relation}:{transition}:depth_source_match"] = (
                float(row["depth_source_match"]) >= float(threshold)
            )
            transition_checks[f"{relation}:{transition}:expected_output_success"] = (
                float(row["expected_output_success"]) >= float(threshold)
            )
    checks = {**aggregate_checks, **transition_checks}
    return {"D_validated": bool(checks) and all(checks.values()), "checks": checks}


def resample_indices_within_relation(
    specs: Sequence[MultiDepthResamplingSpec],
    *,
    rng: random.Random,
) -> list[int]:
    by_relation: dict[str, list[int]] = defaultdict(list)
    for idx, spec in enumerate(specs):
        by_relation[spec.relation].append(idx)
    selected: list[int] = []
    for relation in sorted(by_relation):
        indices = by_relation[relation]
        selected.extend(rng.choice(indices) for _ in indices)
    return selected


def slice_signature(signature: Sequence[float], record_indices: Sequence[int], *, width: int = 4) -> tuple[float, ...]:
    values: list[float] = []
    for record_idx in record_indices:
        start = int(record_idx) * int(width)
        values.extend(float(value) for value in signature[start : start + int(width)])
    return tuple(values)


def posthoc_ranks(ranked_sites: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {site_id: None for site_id in POSTHOC_SITE_IDS}
    for rank, row in enumerate(ranked_sites, start=1):
        site_id = str(row["site_id"])
        if site_id in ranks:
            ranks[site_id] = rank
    return ranks


def support_overlap(site_ids: Sequence[str]) -> dict[str, Any]:
    selected = set(str(site_id) for site_id in site_ids)
    late = {"7.mlp.post_act:4133", "7.mlp.resid_delta:2041"}
    return {
        "posthoc_overlap": sorted(selected & set(POSTHOC_SITE_IDS)),
        "late_readout_overlap": sorted(selected & late),
        "final_resid_overlap": sorted(site_id for site_id in selected if site_id.startswith("final_resid:")),
    }

