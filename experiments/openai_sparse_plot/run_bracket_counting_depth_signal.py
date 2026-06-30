from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .artifacts import load_viz_data
from .runtime import make_tinypython_encoding


DEFAULT_VIZ_PATH = (
    "https://openaipublic.blob.core.windows.net/circuit-sparsity/"
    "viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
)


@dataclass(frozen=True)
class SampleInfo:
    idx: int
    token_ids: tuple[int, ...]
    prompt: str
    tail: str
    depth: int
    positions: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find depth-separating nodes in OpenAI bracket-counting viz data.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/bracket_counting_depth_signal"))
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def _truncate_padding(sample: torch.Tensor) -> torch.Tensor:
    nonzero = sample.nonzero()
    if nonzero.numel() == 0:
        return sample
    return sample[: int(nonzero[-1]) + 1]


def _token_spans(enc: Any, token_ids: tuple[int, ...]) -> list[tuple[int, int, str]]:
    spans = []
    cursor = 0
    for tok in token_ids:
        text = enc.decode([int(tok)])
        spans.append((cursor, cursor + len(text), text))
        cursor += len(text)
    return spans


def _token_at_char(spans: list[tuple[int, int, str]], char_idx: int) -> int:
    for idx, (start, end, _text) in enumerate(spans):
        if start <= char_idx < end:
            return idx
    if char_idx == spans[-1][1]:
        return len(spans) - 1
    raise ValueError(f"no token span covers char index {char_idx}")


def _sample_info(enc: Any, idx: int, token_ids: tuple[int, ...]) -> SampleInfo:
    prompt = enc.decode(token_ids)
    tail_start = prompt.rfind("values =")
    if tail_start < 0:
        raise ValueError("sample prompt does not contain `values =`")
    tail = prompt[tail_start:]
    depth = tail.count("[") - tail.count("]")
    bracket_chars = [tail_start + i for i, ch in enumerate(tail) if ch == "["]
    spans = _token_spans(enc, token_ids)
    positions = {
        "final": len(token_ids) - 1,
        "tail_start": _token_at_char(spans, tail_start),
        "first_open": _token_at_char(spans, bracket_chars[0]),
        "last_open": _token_at_char(spans, bracket_chars[-1]),
    }
    return SampleInfo(idx=idx, token_ids=token_ids, prompt=prompt, tail=tail, depth=depth, positions=positions)


def _load_samples(viz_data: Mapping[str, Any], enc: Any) -> tuple[SampleInfo, ...]:
    tokens = viz_data["importances"]["task_samples"][0]
    out = []
    for idx, row in enumerate(tokens):
        trimmed = _truncate_padding(row.detach().cpu())
        token_ids = tuple(int(x) for x in trimmed.tolist())
        out.append(_sample_info(enc, idx, token_ids))
    return tuple(out)


def _numbers_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit() or ch in {",", " "})


def _pair_indices(samples: tuple[SampleInfo, ...]) -> list[tuple[int, int]]:
    by_nums: dict[str, dict[int, int]] = {}
    for sample in samples:
        by_nums.setdefault(_numbers_only(sample.tail), {})[sample.depth] = sample.idx
    pairs = []
    for depths in by_nums.values():
        if 1 in depths and 2 in depths:
            pairs.append((depths[2], depths[1]))
    return sorted(pairs)


def _safe_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu())


def _node_metrics(
    *,
    hook: str,
    channel: int,
    values: torch.Tensor,
    samples: tuple[SampleInfo, ...],
    pairs: list[tuple[int, int]],
    position_role: str,
) -> dict[str, Any]:
    role_values = []
    for sample in samples:
        pos = sample.positions[position_role]
        role_values.append(_safe_float(values[sample.idx, pos]))
    depth1 = [role_values[s.idx] for s in samples if s.depth == 1]
    depth2 = [role_values[s.idx] for s in samples if s.depth == 2]
    deltas = [role_values[d2_idx] - role_values[d1_idx] for d2_idx, d1_idx in pairs]
    if deltas:
        positive_rate = sum(delta > 0 for delta in deltas) / len(deltas)
        negative_rate = sum(delta < 0 for delta in deltas) / len(deltas)
    else:
        positive_rate = float("nan")
        negative_rate = float("nan")
    mean1 = sum(depth1) / len(depth1)
    mean2 = sum(depth2) / len(depth2)
    pair_delta_mean = sum(deltas) / len(deltas) if deltas else float("nan")
    pair_delta_abs_mean = sum(abs(delta) for delta in deltas) / len(deltas) if deltas else float("nan")
    return {
        "node_id": f"{hook}:{channel}",
        "hook": hook,
        "channel": int(channel),
        "position_role": position_role,
        "mean_depth1": mean1,
        "mean_depth2": mean2,
        "mean_depth2_minus_depth1": mean2 - mean1,
        "paired_delta_mean": pair_delta_mean,
        "paired_abs_delta_mean": pair_delta_abs_mean,
        "paired_positive_rate": positive_rate,
        "paired_negative_rate": negative_rate,
        "n_pairs": len(pairs),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.viz_path)
    samples = _load_samples(viz_data, enc)
    pairs = _pair_indices(samples)

    activation_by_hook = viz_data["importances"]["task_samples"][1]
    rows: list[dict[str, Any]] = []
    for hook, channel_map in activation_by_hook.items():
        for raw_channel, values in channel_map.items():
            channel = int(raw_channel)
            for position_role in ("final", "tail_start", "first_open", "last_open"):
                rows.append(
                    _node_metrics(
                        hook=hook,
                        channel=channel,
                        values=values,
                        samples=samples,
                        pairs=pairs,
                        position_role=position_role,
                    )
                )
    rows.sort(key=lambda row: abs(float(row["paired_delta_mean"])), reverse=True)

    summary = {
        "viz_path": args.viz_path,
        "n_samples": len(samples),
        "n_pairs": len(pairs),
        "depth_counts": {
            "1": sum(sample.depth == 1 for sample in samples),
            "2": sum(sample.depth == 2 for sample in samples),
        },
        "pairs": pairs,
        "top_by_abs_paired_delta": rows[: args.top_k],
        "sample_rows": [
            {
                "idx": sample.idx,
                "depth": sample.depth,
                "positions": sample.positions,
                "tail": sample.tail,
            }
            for sample in samples
        ],
    }
    (args.out_dir / "depth_signal_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "depth_signal_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Bracket Counting Depth Signal",
        "",
        f"- released samples: `{len(samples)}`",
        f"- matched depth-2/depth-1 pairs: `{len(pairs)}`",
        f"- depth-1 samples: `{summary['depth_counts']['1']}`",
        f"- depth-2 samples: `{summary['depth_counts']['2']}`",
        "",
        "Top nodes by paired depth-2 minus depth-1 activation difference:",
        "",
        "| rank | node | position | paired delta | abs delta | sign rate |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[: args.top_k], start=1):
        sign_rate = max(float(row["paired_positive_rate"]), float(row["paired_negative_rate"]))
        lines.append(
            f"| {rank} | `{row['node_id']}` | `{row['position_role']}` | "
            f"{float(row['paired_delta_mean']):.3f} | "
            f"{float(row['paired_abs_delta_mean']):.3f} | "
            f"{sign_rate:.3f} |"
        )
    lines.extend(["", "## Sample Position Roles", ""])
    for sample in samples[:8] + samples[16:24]:
        lines.append(
            f"- sample `{sample.idx}` depth `{sample.depth}` positions `{sample.positions}` "
            f"tail `{sample.tail}`"
        )
    (args.out_dir / "depth_signal_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(args.out_dir), "n_pairs": len(pairs), "top": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
