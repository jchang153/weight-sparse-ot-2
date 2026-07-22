from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite, binary_quote_margin, record_activations, run_with_group_patch
from .artifacts import load_viz_data
from .plot_matching import cost_matrix, sinkhorn_one_sided_uot
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_VIZ_PATH = (
    "https://openaipublic.blob.core.windows.net/circuit-sparsity/"
    "viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
)


@dataclass(frozen=True)
class BracketExample:
    example_id: str
    prompt: str
    token_ids: tuple[int, ...]
    tail: str
    depth: int
    close_count: int
    split: str
    pair_id: str

    def sign(self) -> int:
        return 1 if self.close_count == 2 else -1


@dataclass(frozen=True)
class CandidateHandle:
    handle_id: str
    label: str
    node_ids: tuple[str, ...]
    kind: str
    position_role: str

    def sites(self) -> tuple[ChannelSite, ...]:
        return tuple(ChannelSite.from_node_id(node_id, label=self.label) for node_id in self.node_ids)


@dataclass(frozen=True)
class BracketRun:
    example: BracketExample
    token_ids: torch.Tensor
    logits: torch.Tensor
    cache: dict[str, torch.Tensor]
    margin: float
    predicted_close_count: int
    positions: dict[str, int]


@dataclass(frozen=True)
class ResamplingSpec:
    relation: str
    base_id: str
    source_id: str
    wrong_variable: str | None = None


DEFAULT_HANDLES: tuple[CandidateHandle, ...] = (
    CandidateHandle(
        "late_depth_state_7_mlp_input",
        "late layer-7 MLP input depth state found by matched depth diagnostic",
        ("7.mlp.act_in:1079", "7.mlp.act_in:1249"),
        "depth_signal_group",
        "final",
    ),
    CandidateHandle(
        "late_depth_readout_7_mlp_post",
        "late layer-7 MLP post-activation depth readout found by matched depth diagnostic",
        ("7.mlp.post_act:6561", "7.mlp.post_act:2511", "7.mlp.post_act:4133"),
        "depth_signal_group",
        "final",
    ),
    CandidateHandle(
        "depth_path_1249",
        "channel-1249 depth path across layer-4 attention input and layer-7 MLP input",
        ("4.attn.act_in:1249", "7.mlp.act_in:1249"),
        "depth_signal_group",
        "final",
    ),
    CandidateHandle(
        "late_depth_signal_core",
        "combined late depth signal nodes found by matched depth diagnostic",
        (
            "4.attn.act_in:1249",
            "7.mlp.act_in:1079",
            "7.mlp.act_in:1249",
            "7.mlp.post_act:6561",
            "7.mlp.post_act:2511",
            "7.mlp.post_act:4133",
        ),
        "depth_signal_group",
        "final",
    ),
    CandidateHandle(
        "layer2_attention_depth_head",
        "layer-2 high-importance attention head/output depth path",
        ("2.attn.q:2007", "2.attn.k:2007", "2.attn.v:2012", "2.attn.resid_delta:1249"),
        "openai_depth_candidate",
        "final",
    ),
    CandidateHandle(
        "layer2_value_write",
        "layer-2 value channel and residual depth write",
        ("2.attn.v:2012", "2.attn.resid_delta:1249"),
        "openai_depth_candidate",
        "final",
    ),
    CandidateHandle(
        "layer2_mlp_depth",
        "layer-2 MLP depth/count path",
        ("2.mlp.act_in:1249", "2.mlp.post_act:5676", "2.mlp.resid_delta:1850"),
        "openai_depth_candidate",
        "final",
    ),
    CandidateHandle(
        "layer4_attention_readout",
        "layer-4 attention readout over depth channel",
        ("4.attn.act_in:1249", "4.attn.q:1284", "4.attn.q:1292", "4.attn.resid_delta:1079"),
        "openai_readout_candidate",
        "final",
    ),
    CandidateHandle(
        "late_output_431",
        "late output/readout channel 431",
        ("7.mlp.resid_delta:431", "final_resid:431"),
        "openai_output_candidate",
        "final",
    ),
    CandidateHandle(
        "full_visible_core",
        "visible high-importance core from exported graph",
        (
            "2.attn.q:2007",
            "2.attn.k:2007",
            "2.attn.v:2012",
            "2.attn.resid_delta:1249",
            "2.mlp.act_in:1249",
            "2.mlp.post_act:5676",
            "2.mlp.resid_delta:1850",
            "4.attn.act_in:1249",
            "4.attn.resid_delta:1079",
            "7.mlp.resid_delta:431",
            "final_resid:431",
        ),
        "openai_core_group",
        "final",
    ),
    CandidateHandle(
        "layer1_control_1643",
        "layer-1 attention residual control",
        ("1.attn.resid_delta:1643",),
        "control_openai_group",
        "final",
    ),
    CandidateHandle(
        "random_channels_seed0",
        "fixed random-channel control",
        ("2.attn.resid_delta:1", "2.mlp.resid_delta:1", "final_resid:1"),
        "random_control",
        "final",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a parser-level causal model for bracket counting.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--viz-path", default=DEFAULT_VIZ_PATH)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/bracket_counting_abstraction"))
    parser.add_argument("--max-records-per-relation", type=int, default=16)
    parser.add_argument("--selector-epsilon", type=float, default=0.08)
    parser.add_argument("--selector-beta", type=float, default=0.08)
    parser.add_argument("--handle-id", action="append", default=None, help="Restrict patching to selected handle IDs.")
    parser.add_argument("--clean-only", action="store_true", help="Only score the released samples; skip patching.")
    parser.add_argument(
        "--no-flash",
        action="store_true",
        help="Disable PyTorch SDPA. This breaks sink-attention csp_yolo2 models and is for diagnostics only.",
    )
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _truncate_padding(sample: torch.Tensor) -> torch.Tensor:
    nonzero = sample.nonzero()
    if nonzero.numel() == 0:
        return sample
    return sample[: int(nonzero[-1]) + 1]


def _active_tail(prompt: str) -> str:
    idx = prompt.rfind("values =")
    if idx < 0:
        raise ValueError("sample prompt does not contain `values =`")
    return prompt[idx:]


def _depth_from_tail(tail: str) -> int:
    return int(tail.count("[") - tail.count("]"))


def _close_count_from_depth(depth: int) -> int:
    if depth <= 1:
        return 1
    return 2


def _load_released_examples(viz_data: Mapping[str, Any], enc: Any) -> tuple[BracketExample, ...]:
    tokens = viz_data["importances"]["task_samples"][0]
    examples = []
    for idx, sample in enumerate(tokens):
        trimmed = _truncate_padding(sample.detach().cpu())
        token_ids = tuple(int(x) for x in trimmed.tolist())
        prompt = enc.decode(token_ids)
        tail = _active_tail(prompt)
        depth = _depth_from_tail(tail)
        close_count = _close_count_from_depth(depth)
        split = "calibration" if idx % 2 == 0 else "heldout"
        pair_id = f"tail-{tail.strip()}"
        examples.append(
            BracketExample(
                example_id=f"released-{idx:02d}",
                prompt=prompt,
                token_ids=token_ids,
                tail=tail,
                depth=depth,
                close_count=close_count,
                split=split,
                pair_id=pair_id,
            )
        )
    return tuple(examples)


def _token_tensor(example: BracketExample, *, device: str) -> torch.Tensor:
    return torch.tensor(example.token_ids, dtype=torch.long, device=device).unsqueeze(0)


def bracket_margin(logits: torch.Tensor, *, single_close_token_id: int, double_close_token_id: int) -> float:
    last = logits[0, -1]
    return float(last[double_close_token_id] - last[single_close_token_id])


def _record_sites(handles: Sequence[CandidateHandle]) -> tuple[ChannelSite, ...]:
    by_id: dict[str, ChannelSite] = {}
    for handle in handles:
        for site in handle.sites():
            by_id.setdefault(site.site_id, site)
    return tuple(by_id.values())


def _collect_runs(
    model: Any,
    examples: Sequence[BracketExample],
    *,
    sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
    device: str,
) -> dict[str, BracketRun]:
    runs = {}
    for example in examples:
        token_ids = _token_tensor(example, device=device)
        logits, cache = record_activations(model, token_ids, sites)
        margin = bracket_margin(
            logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        predicted = 2 if margin > 0 else 1
        runs[example.example_id] = BracketRun(
            example=example,
            token_ids=token_ids,
            logits=logits.detach().cpu(),
            cache=cache,
            margin=margin,
            predicted_close_count=predicted,
            positions={
                "final": len(example.token_ids) - 1,
            },
        )
    return runs


def _mean(values: Iterable[float | bool]) -> float:
    vals = [float(x) for x in values]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _sign_from_margin(margin: float) -> int:
    return 1 if float(margin) > 0 else -1


def _positions_for_handle(handle: CandidateHandle, positions: Mapping[str, int]) -> dict[str, list[int]]:
    pos = int(positions[handle.position_role])
    return {site.site_id: [pos] for site in handle.sites()}


def _candidate_specs_for_relation(
    relation: str,
    examples: Sequence[BracketExample],
) -> list[ResamplingSpec]:
    specs = []
    for base in examples:
        for source in examples:
            if base.example_id == source.example_id:
                continue
            same_depth = base.close_count == source.close_count
            if relation == "same_depth":
                if same_depth and base.tail != source.tail:
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id))
            elif relation == "different_depth":
                if not same_depth:
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id))
            elif relation == "wrong_same_tail_length":
                if not same_depth and len(base.tail) == len(source.tail):
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id, "tail_length"))
            elif relation == "wrong_same_numeric_content":
                if not same_depth and _numbers_only(base.tail) == _numbers_only(source.tail):
                    specs.append(ResamplingSpec(relation, base.example_id, source.example_id, "numeric_content"))
            else:
                raise ValueError(f"unknown relation: {relation}")
    return specs


def _numbers_only(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit() or ch in {",", " "})


def _build_resampling_specs(
    examples: Sequence[BracketExample],
    *,
    split: str,
    max_records_per_relation: int,
) -> tuple[ResamplingSpec, ...]:
    split_examples = sorted((ex for ex in examples if ex.split == split), key=lambda ex: ex.example_id)
    by_id = {ex.example_id: ex for ex in split_examples}
    out: list[ResamplingSpec] = []
    for relation in ("same_depth", "different_depth", "wrong_same_tail_length", "wrong_same_numeric_content"):
        specs = _candidate_specs_for_relation(relation, split_examples)
        out.extend(_balanced_prefix(specs, by_id, int(max_records_per_relation)))
    return tuple(out)


def _balanced_prefix(
    specs: Sequence[ResamplingSpec],
    examples_by_id: Mapping[str, BracketExample],
    limit: int,
) -> list[ResamplingSpec]:
    if limit <= 0 or len(specs) <= limit:
        return list(specs[:limit])
    by_base_depth: dict[int, list[ResamplingSpec]] = defaultdict(list)
    for spec in specs:
        by_base_depth[examples_by_id[spec.base_id].close_count].append(spec)
    out: list[ResamplingSpec] = []
    depths = sorted(by_base_depth)
    while len(out) < limit and any(by_base_depth.values()):
        for depth in depths:
            rows = by_base_depth[depth]
            if rows and len(out) < limit:
                out.append(rows.pop(0))
    return out


def _run_records_for_handle(
    *,
    model: Any,
    handle: CandidateHandle,
    specs: Sequence[ResamplingSpec],
    examples: Mapping[str, BracketExample],
    runs: Mapping[str, BracketRun],
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    records = []
    sites = handle.sites()
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = runs[spec.base_id]
        source = runs[spec.source_id]
        patched_logits = run_with_group_patch(
            model,
            base.token_ids,
            sites=sites,
            source_cache=source.cache,
            positions_by_site=_positions_for_handle(handle, base.positions),
            source_positions_by_site=_positions_for_handle(handle, source.positions),
        )
        patched_margin = bracket_margin(
            patched_logits.detach().cpu(),
            single_close_token_id=single_close_token_id,
            double_close_token_id=double_close_token_id,
        )
        base_sign = base_ex.sign()
        source_sign = source_ex.sign()
        patched_sign = _sign_from_margin(patched_margin)
        records.append(
            {
                "handle_id": handle.handle_id,
                "handle_label": handle.label,
                "handle_kind": handle.kind,
                "node_ids": handle.node_ids,
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_example_id": base_ex.example_id,
                "source_example_id": source_ex.example_id,
                "base_tail": base_ex.tail,
                "source_tail": source_ex.tail,
                "base_depth": base_ex.depth,
                "source_depth": source_ex.depth,
                "base_close_count": base_ex.close_count,
                "source_close_count": source_ex.close_count,
                "base_margin": base.margin,
                "source_margin": source.margin,
                "patched_margin": patched_margin,
                "patched_sign": patched_sign,
                "patched_preserves_base_sign": patched_sign == base_sign,
                "patched_matches_source_sign": patched_sign == source_sign,
                "moves_toward_source_sign": (patched_margin - base.margin) * source_sign > 0,
                "source_signed_shift": (patched_margin - base.margin) * source_sign,
                "abs_margin_delta": abs(patched_margin - base.margin),
            }
        )
    return records


def _summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_handle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_handle[str(row["handle_id"])].append(row)

    summary: dict[str, Any] = {}
    for handle_id, rows in sorted(by_handle.items()):
        relation_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            relation_rows[str(row["relation"])].append(row)
        same = relation_rows["same_depth"]
        different = relation_rows["different_depth"]
        wrong_len = relation_rows["wrong_same_tail_length"]
        wrong_content = relation_rows["wrong_same_numeric_content"]
        summary[handle_id] = {
            "handle_label": rows[0]["handle_label"],
            "handle_kind": rows[0]["handle_kind"],
            "node_ids": rows[0]["node_ids"],
            "records": len(rows),
            "same_depth_records": len(same),
            "same_depth_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in same),
            "same_depth_mean_abs_margin_delta": _mean(row["abs_margin_delta"] for row in same),
            "different_depth_records": len(different),
            "different_depth_flip_rate": _mean(row["patched_matches_source_sign"] for row in different),
            "different_depth_move_rate": _mean(row["moves_toward_source_sign"] for row in different),
            "different_depth_mean_source_signed_shift": _mean(row["source_signed_shift"] for row in different),
            "wrong_length_records": len(wrong_len),
            "wrong_length_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in wrong_len),
            "wrong_content_records": len(wrong_content),
            "wrong_content_preserve_rate": _mean(row["patched_preserves_base_sign"] for row in wrong_content),
        }
    return summary


def _metric(row: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    value = float(value)
    if math.isnan(value):
        return default
    return value


def _handle_signature(row: Mapping[str, Any]) -> tuple[float, ...]:
    same_delta = _metric(row, "same_depth_mean_abs_margin_delta", default=100.0)
    signed_shift = _metric(row, "different_depth_mean_source_signed_shift", default=0.0)
    return (
        _metric(row, "same_depth_preserve_rate"),
        _metric(row, "different_depth_flip_rate"),
        1.0 - _metric(row, "wrong_length_preserve_rate", default=1.0),
        1.0 - _metric(row, "wrong_content_preserve_rate", default=1.0),
        1.0 / (1.0 + max(0.0, same_delta)),
        (math.tanh(signed_shift / 10.0) + 1.0) / 2.0,
    )


def _causal_score(row: Mapping[str, Any]) -> float:
    sig = _handle_signature(row)
    return float(sum(sig) / len(sig))


def _selector_payload(
    calibration_summary: Mapping[str, Mapping[str, Any]],
    *,
    epsilon: float,
    beta_neural: float,
) -> dict[str, Any]:
    handle_ids = tuple(calibration_summary)
    signatures = tuple(_handle_signature(calibration_summary[handle_id]) for handle_id in handle_ids)
    desired = torch.ones((1, len(signatures[0])), dtype=torch.float32)
    neural = torch.tensor(signatures, dtype=torch.float32)
    cost = cost_matrix(desired, neural, mode="squared")
    coupling = sinkhorn_one_sided_uot(cost, epsilon=float(epsilon), beta_neural=float(beta_neural), n_iter=300)
    weights = coupling[0]
    ranked = sorted(
        (
            {
                "handle_id": handle_id,
                "weight": float(weights[i]),
                "cost": float(cost[0, i]),
                "causal_score": _causal_score(calibration_summary[handle_id]),
                "signature": signatures[i],
            }
            for i, handle_id in enumerate(handle_ids)
        ),
        key=lambda row: (-float(row["weight"]), float(row["cost"])),
    )
    return {
        "feature_names": (
            "same_depth_preserve",
            "different_depth_flip",
            "wrong_length_failure",
            "wrong_content_failure",
            "same_depth_low_delta",
            "different_depth_shift_score",
        ),
        "desired_signature": tuple(float(x) for x in desired[0].tolist()),
        "handle_ids": handle_ids,
        "signatures": signatures,
        "cost": cost.tolist(),
        "coupling": coupling.tolist(),
        "ranked_handles": ranked,
        "note": (
            "This is a PLOT/UOT selector over candidate handles for the depth variable D. "
            "The main evidence is the heldout resampling table."
        ),
    }


def _clean_summary(examples: Sequence[BracketExample], runs: Mapping[str, BracketRun]) -> dict[str, Any]:
    rows = []
    for example in examples:
        run = runs[example.example_id]
        rows.append(
            {
                "example_id": example.example_id,
                "split": example.split,
                "tail": example.tail,
                "depth": example.depth,
                "close_count": example.close_count,
                "margin": run.margin,
                "predicted_close_count": run.predicted_close_count,
                "correct": run.predicted_close_count == example.close_count,
            }
        )
    return {
        "accuracy": _mean(row["correct"] for row in rows),
        "n": len(rows),
        "depth1_accuracy": _mean(row["correct"] for row in rows if row["close_count"] == 1),
        "depth2_accuracy": _mean(row["correct"] for row in rows if row["close_count"] == 2),
        "mean_depth1_margin": _mean(row["margin"] for row in rows if row["close_count"] == 1),
        "mean_depth2_margin": _mean(row["margin"] for row in rows if row["close_count"] == 2),
        "rows": rows,
    }


def _write_markdown(
    path: Path,
    *,
    args: argparse.Namespace,
    clean: Mapping[str, Any],
    calibration_summary: Mapping[str, Mapping[str, Any]],
    heldout_summary: Mapping[str, Mapping[str, Any]],
    selector: Mapping[str, Any],
) -> None:
    lines = [
        "# Bracket Counting Abstraction",
        "",
        "Hypothesized causal model:",
        "",
        "```text",
        "X -> R",
        "X, R -> D",
        "D -> C",
        "C -> Y",
        "```",
        "",
        "`D` is the active square-bracket depth in the current `values =...` expression.",
        "",
        f"- model: `{args.model}`",
        f"- released samples: `{clean['n']}`",
        f"- clean accuracy: `{clean['accuracy']:.3f}`",
        f"- depth-1 accuracy: `{clean['depth1_accuracy']:.3f}`",
        f"- depth-2 accuracy: `{clean['depth2_accuracy']:.3f}`",
        f"- mean depth-1 margin `logit(]]\\n)-logit(]\\n)`: `{clean['mean_depth1_margin']:.3f}`",
        f"- mean depth-2 margin `logit(]]\\n)-logit(]\\n)`: `{clean['mean_depth2_margin']:.3f}`",
        "",
        "## PLOT/UOT Selector On Calibration",
        "",
        "| rank | handle | weight | cost | causal score |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(selector["ranked_handles"], start=1):
        lines.append(
            f"| {rank} | `{row['handle_id']}` | {row['weight']:.3f} | "
            f"{row['cost']:.3f} | {row['causal_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Heldout Validation",
            "",
            "| handle | kind | same preserve | different flip | wrong length preserve | wrong content preserve | same abs delta | different signed shift |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ranked_ids = [str(row["handle_id"]) for row in selector["ranked_handles"]]
    remaining_ids = [handle_id for handle_id in heldout_summary if handle_id not in ranked_ids]
    for handle_id in [*ranked_ids, *remaining_ids]:
        row = heldout_summary[handle_id]
        lines.append(
            f"| `{handle_id}` | `{row['handle_kind']}` | "
            f"{_metric(row, 'same_depth_preserve_rate'):.3f} | "
            f"{_metric(row, 'different_depth_flip_rate'):.3f} | "
            f"{_metric(row, 'wrong_length_preserve_rate'):.3f} | "
            f"{_metric(row, 'wrong_content_preserve_rate'):.3f} | "
            f"{_metric(row, 'same_depth_mean_abs_margin_delta'):.3f} | "
            f"{_metric(row, 'different_depth_mean_source_signed_shift'):.3f} |"
        )
    lines.extend(["", "## Clean Sample Rows", ""])
    for row in clean["rows"]:
        lines.append(
            f"- `{row['example_id']}` split `{row['split']}` depth `{row['depth']}` "
            f"target `{row['close_count']}` pred `{row['predicted_close_count']}` "
            f"margin `{row['margin']:.3f}` tail `{row['tail'][-80:]}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    flash = not bool(args.no_flash)
    print("loading tokenizer/artifact/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    viz_data = load_viz_data(args.viz_path)
    examples = _load_released_examples(viz_data, enc)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    selected_handles = DEFAULT_HANDLES
    if args.handle_id:
        wanted = set(args.handle_id)
        selected_handles = tuple(handle for handle in DEFAULT_HANDLES if handle.handle_id in wanted)
        missing = sorted(wanted - {handle.handle_id for handle in selected_handles})
        if missing:
            raise ValueError(f"unknown handle IDs: {missing}")
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=flash,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    record_sites = _record_sites(selected_handles)
    runs = _collect_runs(
        model,
        examples,
        sites=record_sites,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    clean = _clean_summary(examples, runs)
    lookup = {ex.example_id: ex for ex in examples}

    if args.clean_only:
        payload = {
            "model_info": model_info,
            "viz_path": args.viz_path,
            "single_close_token_id": single_close_token_id,
            "double_close_token_id": double_close_token_id,
            "clean": clean,
            "handles": [handle.__dict__ for handle in selected_handles],
            "note": "Clean-only run: no activation patching or PLOT/UOT selector was performed.",
        }
        (args.out_dir / "bracket_counting_clean.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Bracket Counting Clean Check",
            "",
            f"- model: `{args.model}`",
            f"- released samples: `{clean['n']}`",
            f"- clean accuracy: `{clean['accuracy']:.3f}`",
            f"- depth-1 accuracy: `{clean['depth1_accuracy']:.3f}`",
            f"- depth-2 accuracy: `{clean['depth2_accuracy']:.3f}`",
            f"- mean depth-1 margin `logit(]]\\n)-logit(]\\n)`: `{clean['mean_depth1_margin']:.3f}`",
            f"- mean depth-2 margin `logit(]]\\n)-logit(]\\n)`: `{clean['mean_depth2_margin']:.3f}`",
            "",
            "## Rows",
            "",
        ]
        for row in clean["rows"]:
            lines.append(
                f"- `{row['example_id']}` split `{row['split']}` depth `{row['depth']}` "
                f"target `{row['close_count']}` pred `{row['predicted_close_count']}` "
                f"margin `{row['margin']:.3f}` tail `{row['tail'][-80:]}`"
            )
        (args.out_dir / "bracket_counting_clean.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"out_dir": str(args.out_dir), "clean": clean}, indent=2))
        return

    split_payloads = {}
    for split in ("calibration", "heldout"):
        specs = _build_resampling_specs(
            examples,
            split=split,
            max_records_per_relation=args.max_records_per_relation,
        )
        records: list[dict[str, Any]] = []
        for handle in selected_handles:
            print(f"{split}: patching {handle.handle_id}", flush=True)
            records.extend(
                _run_records_for_handle(
                    model=model,
                    handle=handle,
                    specs=specs,
                    examples=lookup,
                    runs=runs,
                    single_close_token_id=single_close_token_id,
                    double_close_token_id=double_close_token_id,
                )
            )
        split_payloads[split] = {
            "specs": [spec.__dict__ for spec in specs],
            "records": records,
            "summary": _summarize_records(records),
        }
    selector = _selector_payload(
        split_payloads["calibration"]["summary"],
        epsilon=args.selector_epsilon,
        beta_neural=args.selector_beta,
    )
    payload = {
        "model_info": model_info,
        "viz_path": args.viz_path,
        "single_close_token_id": single_close_token_id,
        "double_close_token_id": double_close_token_id,
        "causal_model": {
            "variables": ["X", "R", "D", "C", "Y"],
            "edges": [["X", "R"], ["X", "D"], ["R", "D"], ["D", "C"], ["C", "Y"]],
            "D": "active square-bracket depth in the current values expression",
            "C": "close count class, one or two brackets",
        },
        "handles": [handle.__dict__ for handle in selected_handles],
        "clean": clean,
        "selector": selector,
        "splits": split_payloads,
    }
    (args.out_dir / "bracket_counting_abstraction.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(
        args.out_dir / "bracket_counting_abstraction.md",
        args=args,
        clean=clean,
        calibration_summary=split_payloads["calibration"]["summary"],
        heldout_summary=split_payloads["heldout"]["summary"],
        selector=selector,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "clean": clean, "top_selector": selector["ranked_handles"][:5]}, indent=2))


if __name__ == "__main__":
    main()
