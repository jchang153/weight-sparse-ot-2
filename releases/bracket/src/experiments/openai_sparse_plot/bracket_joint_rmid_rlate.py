from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from .bracket_multidepth import MultiDepthBracketExample, MultiDepthResamplingSpec
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot


TARGETS: tuple[str, ...] = ("R_mid", "R_late")
SIGNATURE_COMPONENTS: tuple[str, ...] = ("T2", "P_one_close", "P_two_close")


def restricted_output_probabilities(margin: float) -> tuple[float, float]:
    """Return probabilities restricted to the one-close/two-close logits."""

    value = float(margin)
    if value >= 0.0:
        exp_neg = math.exp(-value)
        p_two = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(value)
        p_two = exp_pos / (1.0 + exp_pos)
    return 1.0 - p_two, p_two


def _abstract_state(example: MultiDepthBracketExample) -> tuple[float, float, float]:
    p_one = 1.0 if int(example.close_count) == 1 else 0.0
    p_two = 1.0 - p_one
    return float(int(example.depth) >= 2), p_one, p_two


def abstract_joint_signatures(
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
) -> torch.Tensor:
    """Build the two abstract rows in one shared three-component space."""

    rows: dict[str, list[float]] = {target: [] for target in TARGETS}
    for spec in specs:
        base_t2, base_one, base_two = _abstract_state(examples[spec.base_id])
        source_t2, source_one, source_two = _abstract_state(examples[spec.source_id])
        output_delta = (source_one - base_one, source_two - base_two)
        rows["R_mid"].extend((source_t2 - base_t2, *output_delta))
        rows["R_late"].extend((0.0, *output_delta))
    return torch.tensor([rows[target] for target in TARGETS], dtype=torch.float32)


def neural_joint_signatures(
    *,
    delta_t2: torch.Tensor,
    patched_margins: torch.Tensor,
    base_margins: Sequence[float],
) -> torch.Tensor:
    """Build one shared neural signature per candidate singleton site."""

    if delta_t2.ndim != 2 or patched_margins.ndim != 2:
        raise ValueError("delta_t2 and patched_margins must have shape [sites, records]")
    if delta_t2.shape != patched_margins.shape:
        raise ValueError("delta_t2 and patched_margins must have identical shapes")
    if int(delta_t2.shape[1]) != len(base_margins):
        raise ValueError("base margin count differs from record count")
    base_probs = torch.tensor(
        [restricted_output_probabilities(value) for value in base_margins],
        dtype=torch.float32,
    )
    patched_probs = torch.tensor(
        [
            [restricted_output_probabilities(float(value)) for value in row]
            for row in patched_margins.tolist()
        ],
        dtype=torch.float32,
    )
    delta_probs = patched_probs - base_probs.unsqueeze(0)
    stacked = torch.cat((delta_t2.unsqueeze(-1).to(torch.float32), delta_probs), dim=-1)
    return stacked.reshape(int(delta_t2.shape[0]), -1)


def fit_joint_coupling(
    abstract: torch.Tensor,
    neural: torch.Tensor,
    *,
    epsilon: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abstract.ndim != 2 or int(abstract.shape[0]) != len(TARGETS):
        raise ValueError("abstract signatures must have shape [2, signature_dim]")
    if neural.ndim != 2 or int(neural.shape[1]) != int(abstract.shape[1]):
        raise ValueError("neural signatures must share the abstract signature dimension")
    cost = cost_matrix(abstract, neural, mode="cosine")
    coupling = sinkhorn_one_sided_uot(
        cost,
        epsilon=float(epsilon),
        beta_neural=float(beta),
        n_iter=300,
    )
    return cost, coupling


def ranked_row(
    *,
    target: str,
    site_ids: Sequence[str],
    cost: torch.Tensor,
    coupling: torch.Tensor,
) -> list[dict[str, Any]]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    row_index = TARGETS.index(target)
    rows = [
        {
            "site_id": str(site_id),
            "weight": float(coupling[row_index, index]),
            "cost": float(cost[row_index, index]),
        }
        for index, site_id in enumerate(site_ids)
    ]
    ranked = sorted(rows, key=lambda row: (-float(row["weight"]), float(row["cost"]), str(row["site_id"])))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def topk_handle(ranked: Sequence[Mapping[str, Any]], *, k: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    chosen = list(ranked[: max(1, int(k))])
    if not chosen:
        raise ValueError("cannot extract a handle from an empty ranking")
    total = sum(float(row["weight"]) for row in chosen)
    if total <= 0.0:
        weights = tuple(1.0 / len(chosen) for _ in chosen)
    else:
        weights = tuple(float(row["weight"]) / total for row in chosen)
    return tuple(str(row["site_id"]) for row in chosen), weights


def validation_records(
    *,
    target: str,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    base_phi: Sequence[Sequence[float]],
    patched_delta_phi: Sequence[Sequence[float]],
    patched_margins: Sequence[float],
) -> list[dict[str, Any]]:
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target}")
    if not (len(specs) == len(base_phi) == len(patched_delta_phi) == len(patched_margins)):
        raise ValueError("validation inputs have inconsistent record counts")
    records = []
    for index, spec in enumerate(specs):
        base = examples[spec.base_id]
        source = examples[spec.source_id]
        base_t2 = int(base.depth >= 2)
        source_t2 = int(source.depth >= 2)
        patched_t2_value = float(base_phi[index][1]) + float(patched_delta_phi[index][1])
        patched_t2 = int(patched_t2_value >= 0.5)
        patched_close = 2 if float(patched_margins[index]) > 0.0 else 1
        expected_t2 = source_t2 if target == "R_mid" else base_t2
        expected_close = int(source.close_count)
        direction = "same_R"
        if int(base.close_count) != int(source.close_count):
            direction = "one_to_two" if int(base.close_count) == 1 else "two_to_one"
        records.append(
            {
                "relation": str(spec.relation),
                "base_id": str(spec.base_id),
                "source_id": str(spec.source_id),
                "direction": direction,
                "base_depth": int(base.depth),
                "source_depth": int(source.depth),
                "base_T2": base_t2,
                "source_T2": source_t2,
                "patched_T2": patched_t2,
                "patched_T2_value": patched_t2_value,
                "base_close": int(base.close_count),
                "source_close": int(source.close_count),
                "patched_close": patched_close,
                "expected_T2": expected_t2,
                "expected_close": expected_close,
                "T2_expected_success": patched_t2 == expected_t2,
                "output_expected_success": patched_close == expected_close,
            }
        )
    return records


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(bool(row[key])) for row in rows) / len(rows)


def summarize_validation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    different = [row for row in records if row["direction"] != "same_R"]
    same = [row for row in records if row["direction"] == "same_R"]
    one_to_two = [row for row in records if row["direction"] == "one_to_two"]
    two_to_one = [row for row in records if row["direction"] == "two_to_one"]
    wrong_numeric = [row for row in records if row["relation"] == "wrong_numeric_content"]
    wrong_tail = [row for row in records if row["relation"] == "wrong_tail_length"]
    same_surface = [row for row in records if row["relation"] == "same_surface_different_active_context"]
    gates = {
        "different_R_T2_expected": _mean(different, "T2_expected_success"),
        "different_R_output_expected": _mean(different, "output_expected_success"),
        "one_to_two_T2_expected": _mean(one_to_two, "T2_expected_success"),
        "one_to_two_output_expected": _mean(one_to_two, "output_expected_success"),
        "two_to_one_T2_expected": _mean(two_to_one, "T2_expected_success"),
        "two_to_one_output_expected": _mean(two_to_one, "output_expected_success"),
        "same_R_T2_expected": _mean(same, "T2_expected_success"),
        "same_R_output_expected": _mean(same, "output_expected_success"),
        "wrong_numeric_T2_expected": _mean(wrong_numeric, "T2_expected_success"),
        "wrong_numeric_output_expected": _mean(wrong_numeric, "output_expected_success"),
        "wrong_tail_T2_expected": _mean(wrong_tail, "T2_expected_success"),
        "wrong_tail_output_expected": _mean(wrong_tail, "output_expected_success"),
        "same_surface_T2_expected": _mean(same_surface, "T2_expected_success"),
        "same_surface_output_expected": _mean(same_surface, "output_expected_success"),
    }
    finite = [float(value) for value in gates.values() if math.isfinite(float(value))]
    return {
        "records": len(records),
        "gates": gates,
        "score": sum(finite) / len(finite),
        "worst_gate": min(finite),
        "relation_counts": dict(sorted((key, len(rows)) for key, rows in _group_by(records, "relation").items())),
    }


def _group_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        out[str(row[key])].append(row)
    return dict(out)


def validation_acceptance(summary: Mapping[str, Any], *, threshold: float = 0.90) -> dict[str, Any]:
    checks = {
        key: math.isfinite(float(value)) and float(value) >= float(threshold)
        for key, value in summary["gates"].items()
    }
    return {"threshold": float(threshold), "checks": checks, "validated": all(checks.values())}


def select_target_calibration(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("cannot select from an empty target calibration grid")
    return sorted(
        rows,
        key=lambda row: (
            -int(bool(row["validated"])),
            -float(row["score"]),
            -float(row["worst_gate"]),
            int(row["k"]),
            abs(float(row["strength"]) - 1.0),
            str(row["handle_id"]),
        ),
    )[0]


def select_global_coupling(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_epsilon: float = 0.08,
    reference_beta: float = 0.08,
) -> dict[str, Any]:
    """Select one coupling on Dcal; K and strength remain row-specific."""

    grouped: dict[tuple[float, float], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(float(row["epsilon"]), float(row["beta"]))][str(row["target"])].append(row)
    candidates = []
    for (epsilon, beta), target_rows in sorted(grouped.items()):
        if set(target_rows) != set(TARGETS):
            raise ValueError("every coupling must have calibration rows for both targets")
        selected = {target: select_target_calibration(target_rows[target]) for target in TARGETS}
        candidates.append(
            {
                "epsilon": epsilon,
                "beta": beta,
                "selected": selected,
                "both_validated": all(bool(selected[target]["validated"]) for target in TARGETS),
                "joint_worst_gate": min(float(selected[target]["worst_gate"]) for target in TARGETS),
                "joint_mean_score": sum(float(selected[target]["score"]) for target in TARGETS) / len(TARGETS),
                "total_k": sum(int(selected[target]["k"]) for target in TARGETS),
            }
        )
    best = sorted(
        candidates,
        key=lambda row: (
            -int(bool(row["both_validated"])),
            -float(row["joint_worst_gate"]),
            -float(row["joint_mean_score"]),
            int(row["total_k"]),
            abs(math.log(float(row["epsilon"]) / float(reference_epsilon))),
            abs(math.log(float(row["beta"]) / float(reference_beta))),
            float(row["epsilon"]),
            float(row["beta"]),
        ),
    )[0]
    return {"best": best, "candidates": candidates}
