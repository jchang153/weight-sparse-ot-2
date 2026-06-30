from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CandidateModelSpec:
    model_id: str
    label: str
    variable_count: int
    neural_site_count: int
    required_min_metrics: tuple[str, ...] = ()
    required_max_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateModelScore:
    model_id: str
    label: str
    variable_count: int
    neural_site_count: int
    metrics: dict[str, float]
    passed: bool
    score: float
    failed_metrics: tuple[str, ...]


def _metric(metrics: Mapping[str, float], key: str) -> float:
    if key not in metrics:
        return float("nan")
    return float(metrics[key])


def score_candidate_model(
    spec: CandidateModelSpec,
    metrics: Mapping[str, float],
    *,
    min_threshold: float = 0.90,
    max_threshold: float = 0.10,
) -> CandidateModelScore:
    """Evaluate one candidate abstraction with pass/fail before ranking.

    Metrics listed in ``required_min_metrics`` must be at least
    ``min_threshold``. Metrics listed in ``required_max_metrics`` must be at
    most ``max_threshold``. The scalar score is only a tie-break aid; selection
    should still prefer the simplest passing model.
    """

    failed: list[str] = []
    metric_copy = {str(k): float(v) for k, v in metrics.items()}
    for key in spec.required_min_metrics:
        val = _metric(metric_copy, key)
        if not (val >= float(min_threshold)):
            failed.append(key)
    for key in spec.required_max_metrics:
        val = _metric(metric_copy, key)
        if not (val <= float(max_threshold)):
            failed.append(key)

    scored_values: list[float] = []
    for key in spec.required_min_metrics:
        val = _metric(metric_copy, key)
        if val == val:
            scored_values.append(max(0.0, min(1.0, val)))
    for key in spec.required_max_metrics:
        val = _metric(metric_copy, key)
        if val == val:
            scored_values.append(max(0.0, min(1.0, 1.0 - val)))
    score = sum(scored_values) / len(scored_values) if scored_values else 0.0

    return CandidateModelScore(
        model_id=spec.model_id,
        label=spec.label,
        variable_count=int(spec.variable_count),
        neural_site_count=int(spec.neural_site_count),
        metrics=metric_copy,
        passed=not failed,
        score=float(score),
        failed_metrics=tuple(failed),
    )


def select_simplest_passing(scores: Sequence[CandidateModelScore]) -> CandidateModelScore | None:
    """Return the simplest passing candidate, then highest score."""

    passing = [score for score in scores if score.passed]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda row: (
            int(row.variable_count),
            int(row.neural_site_count),
            -float(row.score),
            row.model_id,
        ),
    )[0]


def scores_to_jsonable(scores: Sequence[CandidateModelScore]) -> list[dict[str, object]]:
    return [
        {
            "model_id": row.model_id,
            "label": row.label,
            "variable_count": row.variable_count,
            "neural_site_count": row.neural_site_count,
            "metrics": row.metrics,
            "passed": row.passed,
            "score": row.score,
            "failed_metrics": list(row.failed_metrics),
        }
        for row in scores
    ]
