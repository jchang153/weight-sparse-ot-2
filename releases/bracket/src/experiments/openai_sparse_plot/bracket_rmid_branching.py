from __future__ import annotations

import math
import random
import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .bracket_joint_rmid_rlate import topk_handle
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec, relation_for_pair


DIRECTIONS: tuple[str, ...] = ("one_to_two", "two_to_one")
DEFAULT_RELATIONS: tuple[str, ...] = (
    "same_D",
    "different_D_same_R",
    "different_D_different_R",
    "same_surface_different_active_context",
    "wrong_numeric_content",
    "wrong_tail_length",
)


def _transition_key(
    relation: str,
    base: MultiDepthBracketExample,
    source: MultiDepthBracketExample,
) -> tuple[int, ...]:
    if relation in {"different_D_same_R", "different_D_different_R"}:
        return int(base.depth), int(source.depth)
    return (int(base.depth),)


def _stable_spec_hash(spec: MultiDepthResamplingSpec) -> str:
    text = f"{spec.relation}|{spec.base_id}|{spec.source_id}|{spec.wrong_variable or ''}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_content_transition_balanced_specs(
    examples: Sequence[MultiDepthBracketExample],
    *,
    split: str,
    records_per_relation: int,
    relations: Sequence[str] = DEFAULT_RELATIONS,
) -> tuple[MultiDepthResamplingSpec, ...]:
    """Balance each relation over ordered depth transitions and base contents."""

    split_examples = sorted((row for row in examples if row.split == split), key=lambda row: row.example_id)
    selected: list[MultiDepthResamplingSpec] = []
    for relation in relations:
        queues: dict[tuple[tuple[int, ...], str], list[MultiDepthResamplingSpec]] = defaultdict(list)
        for base in split_examples:
            for source in split_examples:
                spec = relation_for_pair(base, source, relation)
                if spec is not None:
                    queues[(_transition_key(relation, base, source), str(base.numeric_content))].append(spec)
        for rows in queues.values():
            rows.sort(key=_stable_spec_hash)
        transition_counts: dict[tuple[int, ...], int] = defaultdict(int)
        transition_content_counts: dict[tuple[tuple[int, ...], str], int] = defaultdict(int)
        content_counts: dict[str, int] = defaultdict(int)
        relation_rows: list[MultiDepthResamplingSpec] = []
        while len(relation_rows) < int(records_per_relation):
            available = [key for key, rows in queues.items() if rows]
            if not available:
                break
            key = min(
                available,
                key=lambda value: (
                    transition_counts[value[0]],
                    transition_content_counts[value],
                    content_counts[value[1]],
                    hashlib.sha256(repr(value).encode("utf-8")).hexdigest(),
                ),
            )
            relation_rows.append(queues[key].pop(0))
            transition_counts[key[0]] += 1
            transition_content_counts[key] += 1
            content_counts[key[1]] += 1
        if len(relation_rows) != int(records_per_relation):
            raise ValueError(
                f"relation {relation} produced {len(relation_rows)} records, expected {records_per_relation}"
            )
        selected.extend(relation_rows)
    return tuple(selected)


def build_rlate_calibration_candidates(
    coupling_payload: Mapping[str, Any],
    *,
    k_grid: Sequence[int],
    strength_grid: Sequence[float],
    rmid_site_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Create top-K handles from singleton rankings without pair matching."""

    rmid_support = set(str(site_id) for site_id in rmid_site_ids)
    candidates: list[dict[str, Any]] = []
    for sweep in coupling_payload["sweep"]:
        ranking = sweep["rankings"]["R_late"]
        for k in k_grid:
            site_ids, weights = topk_handle(ranking, k=int(k))
            for strength in strength_grid:
                coefficients = tuple(float(strength) * float(weight) for weight in weights)
                overlap = sorted(rmid_support.intersection(site_ids))
                candidates.append(
                    {
                        "config_id": (
                            f"eps{float(sweep['epsilon']):g}:beta{float(sweep['beta']):g}:"
                            f"K{int(k)}:lambda{float(strength):g}"
                        ),
                        "epsilon": float(sweep["epsilon"]),
                        "beta": float(sweep["beta"]),
                        "k": int(k),
                        "strength": float(strength),
                        "site_ids": list(site_ids),
                        "weights": list(weights),
                        "coefficients": list(coefficients),
                        "rmid_overlap": overlap,
                        "chain_eligible": not overlap,
                    }
                )
    return candidates


def direction(base_close: int, source_close: int) -> str:
    if int(base_close) == int(source_close):
        return "same_R"
    return "one_to_two" if int(base_close) == 1 else "two_to_one"


def signed_controlled_direct_fraction(*, base: float, mid: float, blocked: float) -> float:
    denominator = float(mid) - float(base)
    if abs(denominator) <= 1e-8:
        return float("nan")
    return (float(blocked) - float(base)) / denominator


def _mean_boolean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(bool(row[key])) for row in rows) / len(rows)


def _mean_finite(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else float("nan")


def summarize_blocking_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    by_direction = {
        value: [row for row in records if row["direction"] == value]
        for value in DIRECTIONS
    }
    metrics = {
        "mid_output_source": _mean_boolean(records, "mid_output_source"),
        "mid_T2_source": _mean_boolean(records, "mid_T2_source"),
        "blocked_output_base": _mean_boolean(records, "blocked_output_base"),
        "blocked_T2_source": _mean_boolean(records, "blocked_T2_source"),
        "late_output_source": _mean_boolean(records, "late_output_source"),
        "late_T2_base": _mean_boolean(records, "late_T2_base"),
    }
    for value, rows in by_direction.items():
        metrics[f"blocked_output_base_{value}"] = _mean_boolean(rows, "blocked_output_base")
        metrics[f"late_output_source_{value}"] = _mean_boolean(rows, "late_output_source")
        metrics[f"mean_CDE_fraction_{value}"] = _mean_finite(rows, "CDE_fraction")
    required = (
        "mid_output_source",
        "mid_T2_source",
        "blocked_output_base",
        "blocked_T2_source",
        "late_output_source",
        "late_T2_base",
        "blocked_output_base_one_to_two",
        "blocked_output_base_two_to_one",
        "late_output_source_one_to_two",
        "late_output_source_two_to_one",
    )
    checks = {
        key: math.isfinite(float(metrics[key])) and float(metrics[key]) >= float(threshold)
        for key in required
    }
    finite = [float(metrics[key]) for key in required if math.isfinite(float(metrics[key]))]
    return {
        "record_count": len(records),
        "direction_counts": {key: len(value) for key, value in by_direction.items()},
        "metrics": metrics,
        "checks": checks,
        "score": sum(finite) / len(finite),
        "worst_gate": min(finite),
        "validated": all(checks.values()),
    }


def select_mediation_aware_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_epsilon: float = 0.08,
    reference_beta: float = 0.08,
) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty calibration grid")
    return sorted(
        rows,
        key=lambda row: (
            -int(bool(row["chain_eligible"] and row["direct_validated"] and row["blocking_validated"])),
            -float(row["worst_gate"]),
            -float(row["score"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
            abs(math.log(float(row["epsilon"]) / float(reference_epsilon))),
            abs(math.log(float(row["beta"]) / float(reference_beta))),
            str(row["config_id"]),
        ),
    )[0]


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    generator = random.Random(int(seed))
    means = []
    for _ in range(int(repetitions)):
        sample = [finite[generator.randrange(len(finite))] for _ in finite]
        means.append(sum(sample) / len(sample))
    means.sort()
    low_index = max(0, int(0.025 * len(means)))
    high_index = min(len(means) - 1, int(0.975 * len(means)))
    return {
        "mean": sum(finite) / len(finite),
        "low": means[low_index],
        "high": means[high_index],
    }


def bootstrap_cluster_mean_interval(
    records: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    cluster_key: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | str]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = float(row[value_key])
        if math.isfinite(value):
            clusters[str(row[cluster_key])].append(value)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {
            "mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "clusters": 0,
            "cluster_key": cluster_key,
        }
    generator = random.Random(int(seed))
    means = []
    for _ in range(int(repetitions)):
        sampled_ids = [cluster_ids[generator.randrange(len(cluster_ids))] for _ in cluster_ids]
        sample = [value for cluster_id in sampled_ids for value in clusters[cluster_id]]
        means.append(sum(sample) / len(sample))
    means.sort()
    all_values = [value for cluster_id in cluster_ids for value in clusters[cluster_id]]
    low_index = max(0, int(0.025 * len(means)))
    high_index = min(len(means) - 1, int(0.975 * len(means)))
    return {
        "mean": sum(all_values) / len(all_values),
        "low": means[low_index],
        "high": means[high_index],
        "clusters": len(cluster_ids),
        "cluster_key": cluster_key,
    }


def summarize_factorial_records(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    direct_fraction_tolerance: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_direction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_direction[str(row["direction"])].append(row)
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(DIRECTIONS):
        rows = by_direction[value]
        metrics[f"mid_output_source_{value}"] = _mean_boolean(rows, "mid_output_source")
        metrics[f"mid_T2_source_{value}"] = _mean_boolean(rows, "mid_T2_source")
        metrics[f"late_output_source_{value}"] = _mean_boolean(rows, "late_output_source")
        metrics[f"late_T2_base_{value}"] = _mean_boolean(rows, "late_T2_base")
        metrics[f"both_source_output_source_{value}"] = _mean_boolean(rows, "both_source_output_source")
        metrics[f"both_source_T2_source_{value}"] = _mean_boolean(rows, "both_source_T2_source")
        metrics[f"blocked_output_base_{value}"] = _mean_boolean(rows, "blocked_output_base")
        metrics[f"blocked_output_source_{value}"] = _mean_boolean(rows, "blocked_output_source")
        metrics[f"blocked_T2_source_{value}"] = _mean_boolean(rows, "blocked_T2_source")
        intervals[value] = bootstrap_cluster_mean_interval(
            rows,
            value_key="CDE_fraction",
            cluster_key="base_content",
            repetitions=int(bootstrap_repetitions),
            seed=int(bootstrap_seed) + index,
        )
        metrics[f"mean_CDE_fraction_{value}"] = float(intervals[value]["mean"])
    sufficiency_checks = {
        f"late_output_source_{value}": metrics[f"late_output_source_{value}"] >= float(threshold)
        for value in DIRECTIONS
    }
    sufficiency_checks.update(
        {
            f"late_T2_base_{value}": metrics[f"late_T2_base_{value}"] >= float(threshold)
            for value in DIRECTIONS
        }
    )
    pure_chain_checks = {
        f"blocked_output_base_{value}": metrics[f"blocked_output_base_{value}"] >= float(threshold)
        for value in DIRECTIONS
    }
    pure_chain_checks.update(
        {
            f"CDE_near_zero_{value}": abs(metrics[f"mean_CDE_fraction_{value}"])
            <= float(direct_fraction_tolerance)
            for value in DIRECTIONS
        }
    )
    branching_checks = {
        f"positive_CDE_{value}": (
            intervals[value]["mean"] > float(direct_fraction_tolerance)
            and intervals[value]["low"] > 0.0
        )
        for value in DIRECTIONS
    }
    sufficiency_valid = all(sufficiency_checks.values())
    pure_chain_valid = sufficiency_valid and all(pure_chain_checks.values())
    symmetric_branch_valid = sufficiency_valid and all(branching_checks.values())
    if not sufficiency_valid:
        conclusion = "residual_effect_but_Rlate_invariance_failed"
    elif pure_chain_valid:
        conclusion = "pure_chain_supported"
    elif symmetric_branch_valid:
        conclusion = "branching_model_supported"
    elif any(branching_checks.values()):
        conclusion = "state_dependent_bypass_evidence"
    else:
        conclusion = "inconclusive_incomplete_mediation"
    return {
        "record_count": len(records),
        "direction_counts": {key: len(value) for key, value in sorted(by_direction.items())},
        "metrics": metrics,
        "CDE_bootstrap_95": intervals,
        "bootstrap_unit": "base numeric content",
        "sufficiency_checks": sufficiency_checks,
        "pure_chain_checks": pure_chain_checks,
        "branching_checks": branching_checks,
        "pure_chain_validated": pure_chain_valid,
        "symmetric_branch_validated": symmetric_branch_valid,
        "conclusion": conclusion,
    }
