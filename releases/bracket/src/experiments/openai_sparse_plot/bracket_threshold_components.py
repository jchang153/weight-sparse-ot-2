from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .bracket_d_large_bank_frozen_readout import mean
from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec


SCHEMA_VERSION = 1
TARGET_COMPONENTS: dict[str, int] = {"T2": 1, "T3": 2, "T4": 3}
TARGET_ADJACENT_DEPTHS: dict[str, tuple[int, int]] = {
    "T2": (1, 2),
    "T3": (2, 3),
    "T4": (3, 4),
}
COMPOUND_DEPTH_PAIRS: frozenset[frozenset[int]] = frozenset(
    {frozenset((1, 3)), frozenset((1, 4)), frozenset((2, 4))}
)
POSTHOC_SITE_IDS: tuple[str, ...] = (
    "2.attn.resid_delta:1249",
    "3.attn.v:1365",
    "4.attn.resid_delta:1079",
    "7.mlp.post_act:4133",
    "7.mlp.resid_delta:2041",
)
CORE_METRIC_NAMES: tuple[str, ...] = (
    "isolated_target_source_match",
    "isolated_non_target_base_preserve",
    "isolated_expected_output_success",
    "other_depth_vector_base_preserve",
    "other_depth_output_base_preserve",
    "same_D_control_success",
    "same_surface_control_success",
    "wrong_numeric_control_success",
    "wrong_tail_control_success",
)


def target_index(target: str) -> int:
    try:
        return TARGET_COMPONENTS[str(target)] - 1
    except KeyError as exc:
        raise ValueError(f"unknown threshold target: {target}") from exc


def threshold_vector(depth: int) -> tuple[int, int, int]:
    value = int(depth)
    return (int(value >= 2), int(value >= 3), int(value >= 4))


def predicted_threshold_vector(phi: Sequence[float]) -> tuple[int, int, int]:
    if len(phi) < 4:
        raise ValueError("frozen depth readout must return [norm_D, T2, T3, T4]")
    return tuple(int(float(value) >= 0.5) for value in phi[1:4])


def changed_threshold_targets(base_depth: int, source_depth: int) -> tuple[str, ...]:
    base = threshold_vector(base_depth)
    source = threshold_vector(source_depth)
    return tuple(target for idx, target in enumerate(TARGET_COMPONENTS) if base[idx] != source[idx])


def slice_component_signature(
    signature: Sequence[float],
    *,
    target: str,
    record_count: int,
    width: int = 4,
) -> tuple[float, ...]:
    if len(signature) != int(record_count) * int(width):
        raise ValueError(
            f"signature length {len(signature)} does not equal record_count*width "
            f"({record_count}*{width})"
        )
    component = TARGET_COMPONENTS[str(target)]
    return tuple(float(signature[idx * int(width) + component]) for idx in range(int(record_count)))


def abstract_component_signature(
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    *,
    target: str,
) -> tuple[float, ...]:
    idx = target_index(target)
    return tuple(
        float(threshold_vector(examples[spec.source_id].depth)[idx] - threshold_vector(examples[spec.base_id].depth)[idx])
        for spec in specs
    )


def pair_category(
    target: str,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
) -> str:
    base = examples[spec.base_id]
    source = examples[spec.source_id]
    changed = changed_threshold_targets(base.depth, source.depth)
    if target in changed:
        return "isolated_target_change" if len(changed) == 1 else "compound_target_change"
    if int(base.depth) != int(source.depth):
        return "other_depth_invariance"
    if spec.relation == "same_D":
        return "same_D"
    if spec.relation == "wrong_numeric_content":
        return "wrong_numeric_content"
    if spec.relation == "wrong_tail_length":
        return "wrong_tail_length"
    return "same_value_control"


def individual_spec_eligible(
    target: str,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
) -> bool:
    return pair_category(target, spec, examples) != "compound_target_change"


def is_joint_compound_spec(
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
) -> bool:
    base = examples[spec.base_id]
    source = examples[spec.source_id]
    return (
        spec.relation in {"different_D_same_R", "different_D_different_R"}
        and frozenset((int(base.depth), int(source.depth))) in COMPOUND_DEPTH_PAIRS
    )


def expected_component_record(
    *,
    target: str,
    spec: MultiDepthResamplingSpec,
    examples: Mapping[str, MultiDepthBracketExample],
    patched_phi: Sequence[float],
    patched_close_count: int,
) -> dict[str, Any]:
    base = examples[spec.base_id]
    source = examples[spec.source_id]
    idx = target_index(target)
    base_bits = threshold_vector(base.depth)
    source_bits = threshold_vector(source.depth)
    patched_bits = predicted_threshold_vector(patched_phi)
    category = pair_category(target, spec, examples)
    if category == "compound_target_change":
        raise ValueError("compound pairs are not individual-component records")

    isolated = category == "isolated_target_change"
    target_matches_source = patched_bits[idx] == source_bits[idx]
    target_preserves_base = patched_bits[idx] == base_bits[idx]
    non_target_preserves = all(patched_bits[j] == base_bits[j] for j in range(3) if j != idx)
    vector_preserves_base = patched_bits == base_bits
    expected_output = int(source.close_count) if isolated and target == "T2" else int(base.close_count)
    output_success = int(patched_close_count) == expected_output
    row_success = (
        target_matches_source and non_target_preserves and output_success
        if isolated
        else vector_preserves_base and output_success
    )
    return {
        "target": target,
        "relation": spec.relation,
        "category": category,
        "base_id": base.example_id,
        "source_id": source.example_id,
        "base_depth": int(base.depth),
        "source_depth": int(source.depth),
        "transition": f"{base.depth}->{source.depth}",
        "base_bits": list(base_bits),
        "source_bits": list(source_bits),
        "patched_bits": list(patched_bits),
        "patched_phi": [float(value) for value in patched_phi],
        "patched_close_count": int(patched_close_count),
        "expected_close_count": expected_output,
        "target_matches_source": bool(target_matches_source),
        "target_preserves_base": bool(target_preserves_base),
        "non_target_bits_preserve_base": bool(non_target_preserves),
        "vector_preserves_base": bool(vector_preserves_base),
        "output_success": bool(output_success),
        "row_success": bool(row_success),
    }


def _rows(
    records: Sequence[Mapping[str, Any]],
    *,
    category: str | None = None,
    relation: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in records
        if (category is None or row["category"] == category)
        and (relation is None or row["relation"] == relation)
    ]


def _required_mean(values: Iterable[Any], *, name: str) -> float:
    rows = list(values)
    if not rows:
        raise ValueError(f"no records for required metric {name}")
    return mean(rows)


def summarize_component_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    isolated = _rows(records, category="isolated_target_change")
    other_depth = [
        row
        for row in _rows(records, category="other_depth_invariance")
        if row["relation"] != "same_surface_different_active_context"
    ]
    same_d = _rows(records, relation="same_D")
    same_surface = _rows(records, relation="same_surface_different_active_context")
    wrong_numeric = _rows(records, relation="wrong_numeric_content")
    wrong_tail = _rows(records, relation="wrong_tail_length")

    metrics = {
        "isolated_target_source_match": _required_mean(
            (row["target_matches_source"] for row in isolated), name="isolated_target_source_match"
        ),
        "isolated_non_target_base_preserve": _required_mean(
            (row["non_target_bits_preserve_base"] for row in isolated),
            name="isolated_non_target_base_preserve",
        ),
        "isolated_expected_output_success": _required_mean(
            (row["output_success"] for row in isolated), name="isolated_expected_output_success"
        ),
        "other_depth_vector_base_preserve": _required_mean(
            (row["vector_preserves_base"] for row in other_depth),
            name="other_depth_vector_base_preserve",
        ),
        "other_depth_output_base_preserve": _required_mean(
            (row["output_success"] for row in other_depth), name="other_depth_output_base_preserve"
        ),
        "same_D_control_success": _required_mean(
            (row["row_success"] for row in same_d), name="same_D_control_success"
        ),
        "same_surface_control_success": _required_mean(
            (row["row_success"] for row in same_surface), name="same_surface_control_success"
        ),
        "wrong_numeric_control_success": _required_mean(
            (row["row_success"] for row in wrong_numeric), name="wrong_numeric_control_success"
        ),
        "wrong_tail_control_success": _required_mean(
            (row["row_success"] for row in wrong_tail), name="wrong_tail_control_success"
        ),
    }

    directional: dict[str, dict[str, Any]] = {}
    by_transition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in isolated:
        by_transition[str(row["transition"])].append(row)
    for transition, rows in sorted(by_transition.items()):
        directional[transition] = {
            "n": len(rows),
            "target_source_match": mean(row["target_matches_source"] for row in rows),
            "non_target_base_preserve": mean(row["non_target_bits_preserve_base"] for row in rows),
            "expected_output_success": mean(row["output_success"] for row in rows),
        }

    return {
        "records": len(records),
        "relation_counts": dict(sorted(_count_by(records, "relation").items())),
        "category_counts": dict(sorted(_count_by(records, "category").items())),
        "metrics": metrics,
        "directional": directional,
    }


def _count_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[str(row[key])] += 1
    return dict(counts)


def component_calibration_score(summary: Mapping[str, Any]) -> float:
    metrics = summary["metrics"]
    values = [float(metrics[name]) for name in CORE_METRIC_NAMES]
    if any(math.isnan(value) for value in values):
        raise ValueError("component calibration score contains NaN")
    return float(sum(values) / len(values))


def component_acceptance(
    target: str,
    summary: Mapping[str, Any],
    *,
    threshold: float = 0.90,
) -> dict[str, Any]:
    checks = {
        name: float(summary["metrics"][name]) >= float(threshold)
        for name in CORE_METRIC_NAMES
    }
    expected_depths = TARGET_ADJACENT_DEPTHS[target]
    expected_transitions = {
        f"{expected_depths[0]}->{expected_depths[1]}",
        f"{expected_depths[1]}->{expected_depths[0]}",
    }
    actual_transitions = set(summary["directional"])
    checks["both_adjacent_directions_present"] = actual_transitions == expected_transitions
    for transition in sorted(expected_transitions):
        row = summary["directional"].get(transition, {})
        for metric in ("target_source_match", "non_target_base_preserve", "expected_output_success"):
            checks[f"{transition}:{metric}"] = float(row.get(metric, float("nan"))) >= float(threshold)
    return {"validated": bool(checks) and all(checks.values()), "checks": checks}


def readout_component_quality(
    target: str,
    examples: Sequence[MultiDepthBracketExample],
    phi_by_example: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    idx = target_index(target)
    rows = []
    for example in examples:
        predicted = predicted_threshold_vector(phi_by_example[example.example_id])[idx]
        expected = threshold_vector(example.depth)[idx]
        rows.append(predicted == expected)
    return {"target": target, "n": len(rows), "accuracy": mean(rows)}


def resample_record_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random,
) -> list[int]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        strata[(str(row["relation"]), str(row["category"]))].append(idx)
    sampled: list[int] = []
    for stratum in sorted(strata):
        indices = strata[stratum]
        sampled.extend(rng.choice(indices) for _ in indices)
    return sampled


def combined_joint_coefficients(
    handles: Mapping[str, Mapping[str, Any]],
    changed_targets: Sequence[str],
) -> dict[str, float]:
    coefficients: dict[str, float] = defaultdict(float)
    for target in changed_targets:
        handle = handles[str(target)]
        strength = float(handle["strength"])
        for site_id, weight in handle["weights_by_site"].items():
            coefficients[str(site_id)] += strength * float(weight)
    return dict(sorted(coefficients.items()))


def summarize_joint_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    transitions: dict[str, dict[str, Any]] = {}
    by_transition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_transition[str(row["transition"])].append(row)
    for transition, rows in sorted(by_transition.items()):
        transitions[transition] = {
            "n": len(rows),
            "vector_source_match": mean(row["vector_matches_source"] for row in rows),
            "output_source_match": mean(row["output_matches_source"] for row in rows),
        }
    return {
        "records": len(records),
        "vector_source_match": mean(row["vector_matches_source"] for row in records),
        "output_source_match": mean(row["output_matches_source"] for row in records),
        "transitions": transitions,
    }


def joint_acceptance(summary: Mapping[str, Any], *, threshold: float = 0.90) -> dict[str, Any]:
    expected = {"1->3", "3->1", "1->4", "4->1", "2->4", "4->2"}
    checks = {
        "all_compound_directions_present": set(summary["transitions"]) == expected,
        "aggregate_vector_source_match": float(summary["vector_source_match"]) >= float(threshold),
        "aggregate_output_source_match": float(summary["output_source_match"]) >= float(threshold),
    }
    for transition in sorted(expected):
        row = summary["transitions"].get(transition, {})
        checks[f"{transition}:vector_source_match"] = float(row.get("vector_source_match", float("nan"))) >= float(threshold)
        checks[f"{transition}:output_source_match"] = float(row.get("output_source_match", float("nan"))) >= float(threshold)
    return {"validated": bool(checks) and all(checks.values()), "checks": checks}


def mediation_fraction(*, base: float, clean_patch: float, blocked_patch: float) -> float:
    clean_effect = abs(float(clean_patch) - float(base))
    if clean_effect <= 1e-6:
        return float("nan")
    return (clean_effect - abs(float(blocked_patch) - float(base))) / clean_effect


def summarize_t2_mediation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    isolated = [row for row in records if row["kind"] == "isolated_T2"]
    wrong = [row for row in records if row["kind"] == "wrong_control"]
    if not isolated or not wrong:
        raise ValueError("mediation records require isolated T2 and wrong-control rows")
    return {
        "records": len(records),
        "isolated_records": len(isolated),
        "wrong_control_records": len(wrong),
        "T2_patch_R1079_move": mean(row["T2_patch_R1079_moves_to_source"] for row in isolated),
        "T2_patch_output_source_match": mean(row["T2_patch_output_matches_source"] for row in isolated),
        "R1079_block_output_base_preserve": mean(row["block_output_preserves_base"] for row in isolated),
        "R1079_block_late_probe_mediation_fraction": _safe_mean(
            row["late_probe_mediation_fraction"] for row in isolated
        ),
        "direct_R1079_output_source_match": mean(row["direct_R1079_output_matches_source"] for row in isolated),
        "wrong_control_output_preserve": mean(row["T2_patch_output_preserves_base"] for row in wrong),
    }


def _safe_mean(values: Iterable[Any]) -> float:
    rows = [float(value) for value in values]
    rows = [value for value in rows if not math.isnan(value)]
    return float(sum(rows) / len(rows)) if rows else float("nan")


def t2_mediation_acceptance(
    summary: Mapping[str, Any],
    *,
    t2_site_ids: Sequence[str],
    r_control_site: str,
    threshold: float = 0.90,
) -> dict[str, Any]:
    checks = {
        "T2_support_excludes_R1079": str(r_control_site) not in set(t2_site_ids),
        "T2_patch_R1079_move": float(summary["T2_patch_R1079_move"]) >= float(threshold),
        "T2_patch_output_source_match": float(summary["T2_patch_output_source_match"]) >= float(threshold),
        "R1079_block_output_base_preserve": float(summary["R1079_block_output_base_preserve"]) >= float(threshold),
        "R1079_block_late_probe_mediation_fraction": float(
            summary["R1079_block_late_probe_mediation_fraction"]
        ) >= 0.50,
        "direct_R1079_output_source_match": float(summary["direct_R1079_output_source_match"]) >= float(threshold),
        "wrong_control_output_preserve": float(summary["wrong_control_output_preserve"]) >= float(threshold),
    }
    return {"validated": bool(checks) and all(checks.values()), "checks": checks}


def final_model_decision(
    component_acceptances: Mapping[str, Mapping[str, Any]],
    joint: Mapping[str, Any],
    mediation: Mapping[str, Any],
) -> dict[str, Any]:
    accepted_components = sorted(
        target for target, payload in component_acceptances.items() if bool(payload.get("validated"))
    )
    all_components = accepted_components == ["T2", "T3", "T4"]
    full = all_components and bool(joint.get("validated")) and bool(mediation.get("validated"))
    if full:
        status = "full threshold model accepted"
        model = "X -> (T2,T3,T4); T2 -> R -> Y"
    elif accepted_components:
        status = "partial threshold state; coarse output model retained"
        model = "X -> R -> Y"
    else:
        status = "no threshold refinement; coarse model retained"
        model = "X -> R -> Y"
    return {
        "full_model_accepted": full,
        "accepted_components": accepted_components,
        "joint_validated": bool(joint.get("validated")),
        "T2_to_R_validated": bool(mediation.get("validated")),
        "status": status,
        "accepted_model": model,
    }
