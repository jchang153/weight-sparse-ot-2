from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .artifacts import DEFAULT_MODEL, DEFAULT_TASK, candidate_viz_paths, load_viz_data, summarize_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit available faithfulness evidence in OpenAI sparse viz artifacts.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--sweep", default="prune_v2")
    parser.add_argument("--k", default="64")
    parser.add_argument("--viz-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/faithfulness_audit"))
    return parser.parse_args()


def _len_or_none(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


def compare_task_samples(task_samples: Any, task_samples_circuit: Any) -> dict[str, Any]:
    if not (
        isinstance(task_samples, tuple)
        and isinstance(task_samples_circuit, tuple)
        and len(task_samples) == 2
        and len(task_samples_circuit) == 2
    ):
        return {"available": False, "reason": "unexpected task sample structure"}
    tokens, samples = task_samples
    circuit_tokens, circuit_samples = task_samples_circuit
    if not isinstance(samples, dict) or not isinstance(circuit_samples, dict):
        return {"available": False, "reason": "sample payloads are not dictionaries"}

    rows = []
    for hook_key, channels in samples.items():
        if not isinstance(channels, dict):
            continue
        circuit_channels = circuit_samples.get(hook_key, {})
        if not isinstance(circuit_channels, dict):
            continue
        for channel, tensor in channels.items():
            if channel not in circuit_channels:
                continue
            circuit_tensor = circuit_channels[channel]
            if not hasattr(tensor, "detach") or not hasattr(circuit_tensor, "detach"):
                continue
            diff = (circuit_tensor.detach().cpu() - tensor.detach().cpu()).abs()
            rows.append(
                {
                    "hook_key": str(hook_key),
                    "channel": int(channel),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                }
            )
    token_equal = bool(torch.equal(tokens.detach().cpu(), circuit_tokens.detach().cpu())) if hasattr(tokens, "detach") and hasattr(circuit_tokens, "detach") else None
    return {
        "available": True,
        "token_tensors_equal": token_equal,
        "compared_retained_tensors": len(rows),
        "max_abs_diff": max((row["max_abs_diff"] for row in rows), default=0.0),
        "mean_abs_diff_over_tensors": sum((row["mean_abs_diff"] for row in rows), 0.0) / max(1, len(rows)),
        "largest_diffs": sorted(rows, key=lambda row: row["max_abs_diff"], reverse=True)[:10],
    }


def audit_viz_data(viz_data: dict[str, Any], *, viz_path: str) -> dict[str, Any]:
    importances = viz_data.get("importances", {})
    circuit_data = viz_data.get("circuit_data", {})
    samples = viz_data.get("samples", {})
    task_samples = importances.get("task_samples") if isinstance(importances, dict) else None
    task_samples_circuit = importances.get("task_samples_circuit") if isinstance(importances, dict) else None
    task_sample_comparison = compare_task_samples(task_samples, task_samples_circuit)
    retained_counts = {}
    if isinstance(circuit_data, dict):
        for key, value in circuit_data.items():
            if key == "prune_config":
                continue
            retained_counts[key] = _len_or_none(value)
    return {
        "viz_path": viz_path,
        "top_level_keys": sorted(viz_data.keys()),
        "num_total_nodes": viz_data.get("num_total_nodes"),
        "all_loss": viz_data.get("all_loss"),
        "prune_config": viz_data.get("prune_config"),
        "reported_original_or_task_loss": importances.get("loss") if isinstance(importances, dict) else None,
        "reported_pruned_intervention_loss": importances.get("interv_loss") if isinstance(importances, dict) else None,
        "task_samples_count": _len_or_none(task_samples),
        "task_samples_circuit_count": _len_or_none(task_samples_circuit),
        "sample_location_count": len(samples) if isinstance(samples, dict) else None,
        "retained_location_count": len(retained_counts),
        "retained_scalar_count": sum(v for v in retained_counts.values() if isinstance(v, int)),
        "retained_counts": retained_counts,
        "task_samples_summary": summarize_value(task_samples, max_depth=2),
        "task_samples_circuit_summary": summarize_value(task_samples_circuit, max_depth=2),
        "task_sample_comparison": task_sample_comparison,
        "executable_pruned_model_available": False,
        "executable_pruned_model_note": (
            "The local OpenAI artifact exposes retained nodes, edge/importances, task samples, and reported pruned loss. "
            "This script did not find or instantiate a separate executable pruned model; local causal tests are interventions on the original sparse model."
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    viz_path = args.viz_path
    if viz_path is None:
        candidates = candidate_viz_paths(model=args.model, task=args.task, sweeps=(args.sweep,), ks=(args.k,))
        viz_path = candidates[0]
    viz_data = load_viz_data(viz_path)
    audit = audit_viz_data(viz_data, viz_path=viz_path)
    (args.out_dir / "faithfulness_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# OpenAI Sparse Faithfulness Audit", ""]
    lines.append(f"- viz artifact: `{viz_path}`")
    lines.append(f"- total possible scalar nodes: `{audit['num_total_nodes']}`")
    lines.append(f"- retained scalar count: `{audit['retained_scalar_count']}`")
    lines.append(f"- reported original/task loss: `{audit['reported_original_or_task_loss']}`")
    lines.append(f"- reported pruned intervention loss: `{audit['reported_pruned_intervention_loss']}`")
    lines.append(f"- all_loss: `{audit['all_loss']}`")
    lines.append(f"- task samples: `{audit['task_samples_count']}`")
    lines.append(f"- task samples circuit: `{audit['task_samples_circuit_count']}`")
    cmp = audit["task_sample_comparison"]
    if cmp.get("available"):
        lines.append(f"- retained sample tensors compared: `{cmp['compared_retained_tensors']}`")
        lines.append(f"- retained sample max abs diff: `{cmp['max_abs_diff']}`")
    lines.extend(["", "## Executability", ""])
    lines.append(
        "The artifact provides released pruning/fidelity numbers and visualizer circuit data, but this local audit does not instantiate an executable pruned model."
    )
    lines.append(
        "Therefore, current local interventions test candidate sites inside the original sparse model. A strict original-vs-pruned faithfulness comparison remains open until we either reconstruct the mask execution path or obtain an executable pruned-circuit API/artifact."
    )
    lines.extend(["", "## Retained Location Counts", ""])
    for key, count in sorted(audit["retained_counts"].items()):
        if count:
            lines.append(f"- `{key}`: `{count}`")
    (args.out_dir / "faithfulness_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote faithfulness audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
