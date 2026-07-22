from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .run_bracket_counting_abstraction import _handle_signature as bracket_signature
from .run_bracket_counting_abstraction import _metric as bracket_metric
from .run_unmatched_quote_abstraction import _handle_signature as quote_signature
from .run_unmatched_quote_abstraction import _metric as quote_metric


SignatureFn = Callable[[Mapping[str, Any]], tuple[float, ...]]
MetricFn = Callable[[Mapping[str, Any], str], float]


TASKS: dict[str, dict[str, Any]] = {
    "quote": {
        "path": Path(
            "results/quote/"
            "unmatched_quote_abstraction_csp_yolo1_template/"
            "unmatched_quote_abstraction.json"
        ),
        "signature_fn": quote_signature,
        "metric_fn": quote_metric,
        "result_name": "unmatched quote type",
        "same_key": "same_u_preserve_rate",
        "flip_key": "opposite_u_flip_rate",
        "wrong_keys": (
            "wrong_position_preserve_rate",
            "wrong_content_preserve_rate",
            "wrong_length_preserve_rate",
        ),
        "shift_key": "opposite_u_mean_source_signed_shift",
    },
    "bracket": {
        "path": Path(
            "results/bracket/"
            "bracket_counting_abstraction_csp_yolo2_depth_vs_controls_flash_balanced_r8/"
            "bracket_counting_abstraction.json"
        ),
        "signature_fn": bracket_signature,
        "metric_fn": bracket_metric,
        "result_name": "active bracket depth",
        "same_key": "same_depth_preserve_rate",
        "flip_key": "different_depth_flip_rate",
        "wrong_keys": (
            "wrong_length_preserve_rate",
            "wrong_content_preserve_rate",
        ),
        "shift_key": "different_depth_mean_source_signed_shift",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay hard-handle selector variants from saved results.")
    parser.add_argument("--quote-json", type=Path, default=TASKS["quote"]["path"])
    parser.add_argument("--bracket-json", type=Path, default=TASKS["bracket"]["path"])
    parser.add_argument("--out-dir", type=Path, default=Path("results/combined/hard_handle_selector_replay"))
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--beta", type=float, default=0.08)
    return parser.parse_args()


def _safe_metric(metric_fn: MetricFn, row: Mapping[str, Any], key: str) -> float:
    return float(metric_fn(row, key))  # type: ignore[arg-type]


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _ranked_rows(
    *,
    handle_ids: tuple[str, ...],
    signatures: tuple[tuple[float, ...], ...],
    weights: torch.Tensor,
    cost: torch.Tensor,
    similarity: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for idx, handle_id in enumerate(handle_ids):
        rows.append(
            {
                "handle_id": handle_id,
                "weight": float(weights[idx]),
                "cost": float(cost[0, idx]),
                "similarity": None if similarity is None else float(similarity[idx]),
                "signature": signatures[idx],
            }
        )
    return sorted(rows, key=lambda row: (-float(row["weight"]), float(row["cost"])))


def _selector_variants(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    signature_fn: SignatureFn,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    handle_ids = tuple(summary)
    signatures = tuple(signature_fn(summary[handle_id]) for handle_id in handle_ids)
    desired = torch.ones((1, len(signatures[0])), dtype=torch.float32)
    neural = torch.tensor(signatures, dtype=torch.float32)

    squared_cost = cost_matrix(desired, neural, mode="squared")
    squared_uot = sinkhorn_one_sided_uot(squared_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]

    cosine_cost = cost_matrix(desired, neural, mode="cosine")
    cosine_uot = sinkhorn_one_sided_uot(cosine_cost, epsilon=epsilon, beta_neural=beta, n_iter=300)[0]
    cosine_similarity = 1.0 - cosine_cost[0]
    direct = cosine_similarity.clamp_min(0.0)
    if float(direct.sum()) <= 0.0:
        direct = torch.softmax(cosine_similarity, dim=0)
    else:
        direct = direct / direct.sum().clamp_min(1e-12)

    return {
        "handle_ids": handle_ids,
        "desired_signature": tuple(float(x) for x in desired[0].tolist()),
        "signatures": signatures,
        "selectors": {
            "squared_uot_existing": {
                "cost_mode": "squared",
                "coupling_rule": "one_sided_uot",
                "ranked_handles": _ranked_rows(
                    handle_ids=handle_ids,
                    signatures=signatures,
                    weights=squared_uot,
                    cost=squared_cost,
                ),
            },
            "cosine_uot": {
                "cost_mode": "cosine",
                "coupling_rule": "one_sided_uot",
                "ranked_handles": _ranked_rows(
                    handle_ids=handle_ids,
                    signatures=signatures,
                    weights=cosine_uot,
                    cost=cosine_cost,
                    similarity=cosine_similarity,
                ),
            },
            "cosine_similarity_normalized": {
                "cost_mode": "cosine",
                "coupling_rule": "row_normalized_positive_cosine_similarity",
                "ranked_handles": _ranked_rows(
                    handle_ids=handle_ids,
                    signatures=signatures,
                    weights=direct,
                    cost=cosine_cost,
                    similarity=cosine_similarity,
                ),
            },
        },
    }


def _task_payload(
    task_name: str,
    path: Path,
    *,
    epsilon: float,
    beta: float,
) -> dict[str, Any]:
    spec = TASKS[task_name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibration = payload["splits"]["calibration"]["summary"]
    heldout = payload["splits"]["heldout"]["summary"]
    selectors = _selector_variants(
        calibration,
        signature_fn=spec["signature_fn"],
        epsilon=epsilon,
        beta=beta,
    )
    metric_fn = spec["metric_fn"]
    for selector in selectors["selectors"].values():
        for row in selector["ranked_handles"]:
            heldout_row = heldout[row["handle_id"]]
            row["heldout"] = {
                "same_preserve": _safe_metric(metric_fn, heldout_row, spec["same_key"]),
                "flip": _safe_metric(metric_fn, heldout_row, spec["flip_key"]),
                "wrong_preserve_mean": _mean(
                    [_safe_metric(metric_fn, heldout_row, key) for key in spec["wrong_keys"]]
                ),
                "source_signed_shift": _safe_metric(metric_fn, heldout_row, spec["shift_key"]),
            }
    return {
        "task": task_name,
        "source_json": str(path),
        "result_name": spec["result_name"],
        "selectors": selectors,
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "# Hard-Handle Selector Replay",
        "",
        "This reuses the saved hard-handle patch results. It changes only how the one-row coupling `pi` is built from calibration signatures.",
        "",
    ]
    for task_name, task in payload["tasks"].items():
        lines.extend(
            [
                f"## {task_name.title()}",
                "",
                f"- source JSON: `{task['source_json']}`",
                f"- abstract variable: `{task['result_name']}`",
                "",
            ]
        )
        for selector_name, selector in task["selectors"]["selectors"].items():
            lines.extend(
                [
                    f"### {selector_name}",
                    "",
                    "| rank | handle | weight | cost | cosine sim | heldout same | heldout flip | heldout wrong-preserve | signed shift |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for rank, row in enumerate(selector["ranked_handles"][:8], start=1):
                heldout = row["heldout"]
                sim = row["similarity"]
                sim_text = "n/a" if sim is None else f"{sim:.3f}"
                lines.append(
                    f"| {rank} | `{row['handle_id']}` | {row['weight']:.3f} | {row['cost']:.3f} | "
                    f"{sim_text} | {heldout['same_preserve']:.3f} | {heldout['flip']:.3f} | "
                    f"{heldout['wrong_preserve_mean']:.3f} | {heldout['source_signed_shift']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        "quote": _task_payload("quote", args.quote_json, epsilon=args.epsilon, beta=args.beta),
        "bracket": _task_payload("bracket", args.bracket_json, epsilon=args.epsilon, beta=args.beta),
    }
    payload = {
        "epsilon": float(args.epsilon),
        "beta": float(args.beta),
        "tasks": tasks,
    }
    (args.out_dir / "hard_handle_selector_replay.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.out_dir / "hard_handle_selector_replay.md", payload)
    compact = {
        task_name: {
            selector_name: rows["ranked_handles"][:3]
            for selector_name, rows in task["selectors"]["selectors"].items()
        }
        for task_name, task in tasks.items()
    }
    print(json.dumps({"out_dir": str(args.out_dir), "top3": compact}, indent=2))


if __name__ == "__main__":
    main()
