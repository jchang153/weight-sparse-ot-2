from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .activation import ChannelSite
from .bracket_d_rich_signatures import (
    build_relation_specs_for_split,
    generate_content_split_multidepth_examples,
    relation_counts,
    split_summary,
)
from .bracket_multidepth import DEFAULT_NUMERIC_CONTENTS, MultiDepthBracketExample, MultiDepthResamplingSpec, parse_depths
from .run_bracket_d_rich_signature_experiments import (
    CleanRun,
    _bracket_margin,
    _clean_summary,
    _collect_clean_runs,
    _feature_dict,
    _feature_vector,
    _hook_regex,
    _sign_from_margin,
)
from .runtime import load_sparse_gpt_model, make_tinypython_encoding


DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/bracket_d1249_r1079_mediation_20260702")
PAPER_D_SITE = ChannelSite.from_node_id("2.attn.resid_delta:1249", label="paper nesting-depth D_1249")
PAPER_R1079_SITE = ChannelSite.from_node_id("4.attn.resid_delta:1079", label="paper threshold/readout R_1079")
R_LATE_SITES: tuple[ChannelSite, ...] = (
    ChannelSite.from_node_id("7.mlp.post_act:4133", label="validated R_late post-act"),
    ChannelSite.from_node_id("7.mlp.resid_delta:2041", label="validated R_late residual write"),
)
R_LATE_PROBE_SITE = ChannelSite.from_node_id("7.mlp.resid_delta:2041", label="R_late scalar probe")
DEFAULT_RELATIONS: tuple[str, ...] = (
    "same_D",
    "different_D_same_R",
    "different_D_different_R",
    "wrong_numeric_content",
    "wrong_tail_length",
)


@dataclass(frozen=True)
class PatchSpec:
    site: ChannelSite
    source_site_id: str
    source_position: int
    target_position: int
    source_value: float
    strength: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test paper D_1249 -> R_1079 -> R_late/Y mediation in the bracket circuit.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo2")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--depths", default="1,2,3,4")
    parser.add_argument("--max-records-per-relation", type=int, default=12)
    parser.add_argument("--d-strength", type=float, default=1.0)
    parser.add_argument("--r1079-strength", type=float, default=1.0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--no-flash", action="store_true")
    return parser.parse_args()


def _mean(values: Iterable[Any]) -> float:
    vals = [float(x) for x in values]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if not math.isnan(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _record_sites() -> tuple[ChannelSite, ...]:
    by_id = {
        PAPER_D_SITE.site_id: PAPER_D_SITE,
        PAPER_R1079_SITE.site_id: PAPER_R1079_SITE,
        R_LATE_PROBE_SITE.site_id: R_LATE_PROBE_SITE,
    }
    for site in R_LATE_SITES:
        by_id.setdefault(site.site_id, site)
    return tuple(by_id.values())


def _make_patch_interventions(patches: Sequence[PatchSpec]) -> dict[str, Any]:
    by_hook: dict[str, list[PatchSpec]] = defaultdict(list)
    for patch in patches:
        by_hook[patch.site.hook_key].append(patch)
    interventions: dict[str, Any] = {}
    for hook_key, hook_patches in by_hook.items():
        patch_tuple = tuple(hook_patches)

        def _patch(tensor: torch.Tensor, *, patch_tuple: tuple[PatchSpec, ...] = patch_tuple) -> torch.Tensor:
            patched = tensor.clone()
            for spec in patch_tuple:
                current = patched[0, int(spec.target_position), int(spec.site.channel)]
                source = torch.tensor(float(spec.source_value), device=patched.device, dtype=patched.dtype)
                patched[0, int(spec.target_position), int(spec.site.channel)] = current + float(spec.strength) * (source - current)
            return patched

        interventions[hook_key] = _patch
    return interventions


def _run_patch_and_record(
    model: Any,
    *,
    base: CleanRun,
    patches: Sequence[PatchSpec],
    record_sites: Sequence[ChannelSite],
    single_close_token_id: int,
    double_close_token_id: int,
) -> tuple[float, int, dict[str, float], tuple[float, ...]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = _make_patch_interventions(patches)
    with torch.no_grad():
        with hook_recorder(regex=_hook_regex(record_sites), interventions=interventions) as ctx:
            logits, _, _ = model(base.token_ids)
    cache = {k: v.detach().cpu() for k, v in ctx.items()}
    features = _feature_dict(cache, record_sites, base.final_position)
    vector = _feature_vector(features, record_sites)
    margin = _bracket_margin(
        logits.detach().cpu(),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    return margin, _sign_from_margin(margin), features, vector


def scalar_moves_toward_source(*, base: float, source: float, patched: float) -> bool:
    if abs(float(source) - float(base)) <= 1e-6:
        return False
    return abs(float(source) - float(patched)) + 1e-6 < abs(float(source) - float(base))


def scalar_preserves_base(*, base: float, source: float, patched: float) -> bool:
    if abs(float(source) - float(base)) <= 1e-6:
        return abs(float(patched) - float(base)) <= 1e-6
    return abs(float(patched) - float(base)) <= 0.25 * abs(float(source) - float(base))


def signed_effect_fraction(*, base: float, source: float, patched: float) -> float:
    denom = float(source) - float(base)
    if abs(denom) <= 1e-6:
        return float("nan")
    return (float(patched) - float(base)) / denom


def mediation_fraction(*, base: float, clean_patch: float, blocked_patch: float) -> float:
    clean_effect = abs(float(clean_patch) - float(base))
    if clean_effect <= 1e-6:
        return float("nan")
    blocked_effect = abs(float(blocked_patch) - float(base))
    return (clean_effect - blocked_effect) / clean_effect


def _patch_specs_for_record(
    *,
    base: CleanRun,
    source: CleanRun,
    d_strength: float,
    r1079_strength: float,
    kind: str,
) -> tuple[PatchSpec, ...]:
    patches: list[PatchSpec] = []
    if kind in {"d_patch", "d_patch_r1079_block"}:
        patches.append(
            PatchSpec(
                site=PAPER_D_SITE,
                source_site_id=PAPER_D_SITE.site_id,
                source_position=source.final_position,
                target_position=base.final_position,
                source_value=float(source.features_by_site[PAPER_D_SITE.site_id]),
                strength=float(d_strength),
                label="patch D_1249 from source",
            )
        )
    if kind == "d_patch_r1079_block":
        patches.append(
            PatchSpec(
                site=PAPER_R1079_SITE,
                source_site_id=PAPER_R1079_SITE.site_id,
                source_position=base.final_position,
                target_position=base.final_position,
                source_value=float(base.features_by_site[PAPER_R1079_SITE.site_id]),
                strength=1.0,
                label="restore R_1079 to clean base value",
            )
        )
    if kind == "r1079_patch":
        patches.append(
            PatchSpec(
                site=PAPER_R1079_SITE,
                source_site_id=PAPER_R1079_SITE.site_id,
                source_position=source.final_position,
                target_position=base.final_position,
                source_value=float(source.features_by_site[PAPER_R1079_SITE.site_id]),
                strength=float(r1079_strength),
                label="patch R_1079 from source",
            )
        )
    if not patches:
        raise ValueError(f"unknown patch kind: {kind}")
    return tuple(patches)


def run_mediation_records(
    *,
    model: Any,
    specs: Sequence[MultiDepthResamplingSpec],
    examples: Mapping[str, MultiDepthBracketExample],
    clean_runs: Mapping[str, CleanRun],
    record_sites: Sequence[ChannelSite],
    d_strength: float,
    r1079_strength: float,
    single_close_token_id: int,
    double_close_token_id: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in specs:
        base_ex = examples[spec.base_id]
        source_ex = examples[spec.source_id]
        base = clean_runs[base_ex.example_id]
        source = clean_runs[source_ex.example_id]

        outcomes: dict[str, dict[str, Any]] = {}
        for kind in ("d_patch", "d_patch_r1079_block", "r1079_patch"):
            margin, close_count, features, _vector = _run_patch_and_record(
                model,
                base=base,
                patches=_patch_specs_for_record(
                    base=base,
                    source=source,
                    d_strength=d_strength,
                    r1079_strength=r1079_strength,
                    kind=kind,
                ),
                record_sites=record_sites,
                single_close_token_id=single_close_token_id,
                double_close_token_id=double_close_token_id,
            )
            outcomes[kind] = {
                "margin": margin,
                "close_count": close_count,
                "features": features,
            }

        base_d = float(base.features_by_site[PAPER_D_SITE.site_id])
        source_d = float(source.features_by_site[PAPER_D_SITE.site_id])
        base_m = float(base.features_by_site[PAPER_R1079_SITE.site_id])
        source_m = float(source.features_by_site[PAPER_R1079_SITE.site_id])
        base_r = float(base.features_by_site[R_LATE_PROBE_SITE.site_id])
        source_r = float(source.features_by_site[R_LATE_PROBE_SITE.site_id])
        d_patch_m = float(outcomes["d_patch"]["features"][PAPER_R1079_SITE.site_id])
        d_patch_r = float(outcomes["d_patch"]["features"][R_LATE_PROBE_SITE.site_id])
        block_m = float(outcomes["d_patch_r1079_block"]["features"][PAPER_R1079_SITE.site_id])
        block_r = float(outcomes["d_patch_r1079_block"]["features"][R_LATE_PROBE_SITE.site_id])
        r_patch_m = float(outcomes["r1079_patch"]["features"][PAPER_R1079_SITE.site_id])
        r_patch_r = float(outcomes["r1079_patch"]["features"][R_LATE_PROBE_SITE.site_id])

        records.append(
            {
                "relation": spec.relation,
                "wrong_variable": spec.wrong_variable,
                "base_id": base_ex.example_id,
                "source_id": source_ex.example_id,
                "base_depth": base_ex.depth,
                "source_depth": source_ex.depth,
                "base_close_count": base_ex.close_count,
                "source_close_count": source_ex.close_count,
                "base_output_close_count": base.predicted_close_count,
                "source_output_close_count": source.predicted_close_count,
                "d_strength": float(d_strength),
                "r1079_strength": float(r1079_strength),
                "base_D1249": base_d,
                "source_D1249": source_d,
                "base_R1079": base_m,
                "source_R1079": source_m,
                "base_R_late": base_r,
                "source_R_late": source_r,
                "D1249_clean_source_fraction": signed_effect_fraction(base=base_d, source=source_d, patched=source_d),
                "d_patch_R1079": d_patch_m,
                "d_patch_R_late": d_patch_r,
                "d_patch_margin": outcomes["d_patch"]["margin"],
                "d_patch_close_count": outcomes["d_patch"]["close_count"],
                "d_patch_R1079_moves_to_source": scalar_moves_toward_source(base=base_m, source=source_m, patched=d_patch_m),
                "d_patch_R_late_moves_to_source": scalar_moves_toward_source(base=base_r, source=source_r, patched=d_patch_r),
                "d_patch_R1079_fraction": signed_effect_fraction(base=base_m, source=source_m, patched=d_patch_m),
                "d_patch_R_late_fraction": signed_effect_fraction(base=base_r, source=source_r, patched=d_patch_r),
                "d_patch_output_preserves_base": outcomes["d_patch"]["close_count"] == base_ex.close_count,
                "d_patch_output_matches_source": outcomes["d_patch"]["close_count"] == source_ex.close_count,
                "block_R1079": block_m,
                "block_R_late": block_r,
                "block_margin": outcomes["d_patch_r1079_block"]["margin"],
                "block_close_count": outcomes["d_patch_r1079_block"]["close_count"],
                "block_R1079_preserves_base": scalar_preserves_base(base=base_m, source=source_m, patched=block_m),
                "block_R_late_moves_to_source": scalar_moves_toward_source(base=base_r, source=source_r, patched=block_r),
                "block_R_late_fraction": signed_effect_fraction(base=base_r, source=source_r, patched=block_r),
                "block_R_late_mediation_fraction": mediation_fraction(base=base_r, clean_patch=d_patch_r, blocked_patch=block_r),
                "block_output_preserves_base": outcomes["d_patch_r1079_block"]["close_count"] == base_ex.close_count,
                "block_output_matches_source": outcomes["d_patch_r1079_block"]["close_count"] == source_ex.close_count,
                "r1079_patch_R1079": r_patch_m,
                "r1079_patch_R_late": r_patch_r,
                "r1079_patch_margin": outcomes["r1079_patch"]["margin"],
                "r1079_patch_close_count": outcomes["r1079_patch"]["close_count"],
                "r1079_patch_R1079_moves_to_source": scalar_moves_toward_source(base=base_m, source=source_m, patched=r_patch_m),
                "r1079_patch_R_late_moves_to_source": scalar_moves_toward_source(base=base_r, source=source_r, patched=r_patch_r),
                "r1079_patch_R_late_fraction": signed_effect_fraction(base=base_r, source=source_r, patched=r_patch_r),
                "r1079_patch_output_preserves_base": outcomes["r1079_patch"]["close_count"] == base_ex.close_count,
                "r1079_patch_output_matches_source": outcomes["r1079_patch"]["close_count"] == source_ex.close_count,
            }
        )
    return records


def summarize_mediation_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)

    def rows(name: str) -> list[Mapping[str, Any]]:
        return by_relation.get(name, [])

    crossing = rows("different_D_different_R")
    same_r = rows("different_D_same_R")
    same_d = rows("same_D")
    wrong = rows("wrong_numeric_content") + rows("wrong_tail_length")
    summary = {
        "records": len(records),
        "relation_counts": {key: len(val) for key, val in sorted(by_relation.items())},
        "threshold_crossing_D_patch_R1079_move": _mean(row["d_patch_R1079_moves_to_source"] for row in crossing),
        "threshold_crossing_D_patch_R_late_move": _mean(row["d_patch_R_late_moves_to_source"] for row in crossing),
        "threshold_crossing_D_patch_output_flip": _mean(row["d_patch_output_matches_source"] for row in crossing),
        "threshold_crossing_D_patch_R1079_fraction": _safe_mean(float(row.get("d_patch_R1079_fraction", float("nan"))) for row in crossing),
        "threshold_crossing_D_patch_R_late_fraction": _safe_mean(float(row.get("d_patch_R_late_fraction", float("nan"))) for row in crossing),
        "threshold_crossing_block_R1079_preserve": _mean(row["block_R1079_preserves_base"] for row in crossing),
        "threshold_crossing_block_R_late_still_moves": _mean(row["block_R_late_moves_to_source"] for row in crossing),
        "threshold_crossing_block_R_late_mediation_fraction": _safe_mean(row["block_R_late_mediation_fraction"] for row in crossing),
        "threshold_crossing_block_output_flip": _mean(row["block_output_matches_source"] for row in crossing),
        "threshold_crossing_block_output_preserve": _mean(row["block_output_preserves_base"] for row in crossing),
        "threshold_crossing_R1079_patch_R_late_move": _mean(row["r1079_patch_R_late_moves_to_source"] for row in crossing),
        "threshold_crossing_R1079_patch_output_flip": _mean(row["r1079_patch_output_matches_source"] for row in crossing),
        "threshold_crossing_R1079_patch_R_late_fraction": _safe_mean(float(row.get("r1079_patch_R_late_fraction", float("nan"))) for row in crossing),
        "same_R_D_patch_R1079_move": _mean(row["d_patch_R1079_moves_to_source"] for row in same_r),
        "same_R_D_patch_R1079_fraction": _safe_mean(float(row.get("d_patch_R1079_fraction", float("nan"))) for row in same_r),
        "same_R_D_patch_R_late_fraction": _safe_mean(float(row.get("d_patch_R_late_fraction", float("nan"))) for row in same_r),
        "same_R_D_patch_output_preserve": _mean(row["d_patch_output_preserves_base"] for row in same_r),
        "same_D_D_patch_output_preserve": _mean(row["d_patch_output_preserves_base"] for row in same_d),
        "wrong_controls_D_patch_output_preserve": _mean(row["d_patch_output_preserves_base"] for row in wrong),
        "wrong_controls_R1079_patch_output_preserve": _mean(row["r1079_patch_output_preserves_base"] for row in wrong),
    }
    summary["mediation_pattern_support"] = (
        summary["threshold_crossing_D_patch_R1079_move"] >= 0.90
        and summary["threshold_crossing_D_patch_R_late_move"] >= 0.90
        and summary["threshold_crossing_block_R_late_mediation_fraction"] >= 0.50
        and summary["threshold_crossing_R1079_patch_R_late_move"] >= 0.90
        and summary["same_R_D_patch_output_preserve"] >= 0.90
        and summary["wrong_controls_D_patch_output_preserve"] >= 0.90
    )
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "# Bracket D1249 -> R1079 Mediation Test",
        "",
        "Fixed hypothesis sites:",
        f"- paper D site: `{payload['sites']['D1249']}`",
        f"- paper mediator/readout site: `{payload['sites']['R1079']}`",
        f"- validated late R sites: `{', '.join(payload['sites']['R_late'])}`",
        "",
        "No neural-site search is performed in this run.",
        "",
        "Hypothesis under test:",
        "",
        "```text",
        "X -> D_1249 -> R_1079 -> R_late -> Y",
        "```",
        "",
        "## Clean Behavior",
        "",
        f"- examples: `{payload['clean']['n']}`",
        f"- clean accuracy: `{payload['clean']['accuracy']:.3f}`",
        "",
        "## Heldout Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| D patch moves R1079 on threshold crossing | {s['threshold_crossing_D_patch_R1079_move']:.3f} |",
        f"| D patch moves R_late on threshold crossing | {s['threshold_crossing_D_patch_R_late_move']:.3f} |",
        f"| D patch flips output on threshold crossing | {s['threshold_crossing_D_patch_output_flip']:.3f} |",
        f"| D patch R1079 source-fraction on threshold crossing | {s['threshold_crossing_D_patch_R1079_fraction']:.3f} |",
        f"| D patch R_late source-fraction on threshold crossing | {s['threshold_crossing_D_patch_R_late_fraction']:.3f} |",
        f"| R1079 block restores R1079 to base | {s['threshold_crossing_block_R1079_preserve']:.3f} |",
        f"| R1079 block R_late mediation fraction | {s['threshold_crossing_block_R_late_mediation_fraction']:.3f} |",
        f"| R1079 block still flips output | {s['threshold_crossing_block_output_flip']:.3f} |",
        f"| direct R1079 patch moves R_late | {s['threshold_crossing_R1079_patch_R_late_move']:.3f} |",
        f"| direct R1079 patch flips output | {s['threshold_crossing_R1079_patch_output_flip']:.3f} |",
        f"| direct R1079 patch R_late source-fraction | {s['threshold_crossing_R1079_patch_R_late_fraction']:.3f} |",
        f"| same-R D patch moves R1079 | {s['same_R_D_patch_R1079_move']:.3f} |",
        f"| same-R D patch R1079 source-fraction | {s['same_R_D_patch_R1079_fraction']:.3f} |",
        f"| same-R D patch R_late source-fraction | {s['same_R_D_patch_R_late_fraction']:.3f} |",
        f"| same-R D patch preserves output | {s['same_R_D_patch_output_preserve']:.3f} |",
        f"| wrong controls D patch preserves output | {s['wrong_controls_D_patch_output_preserve']:.3f} |",
        "",
        f"Pattern support: `{s['mediation_pattern_support']}`",
        "",
        "## Interpretation Rule",
        "",
        "A positive mediation pattern requires D_1249 patching to move R1079/R_late, R1079 blocking to remove at least half of the R_late effect, direct R1079 patching to move R_late, and same-R/wrong-variable controls to preserve output.",
        "If output follows R_late even when R1079 is blocked, then R1079 is not the sole mediator between D_1249 and the late readout.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "heldout").mkdir(exist_ok=True)
    depths = parse_depths(args.depths)
    device = "cuda" if args.cuda else "cpu"

    enc = make_tinypython_encoding(args.circuit_home)
    single_close_token_id = int(enc.encode("]\n")[0])
    double_close_token_id = int(enc.encode("]]\n")[0])
    examples = generate_content_split_multidepth_examples(enc, depths=depths, numeric_contents=DEFAULT_NUMERIC_CONTENTS)
    examples_by_id = {ex.example_id: ex for ex in examples}
    specs = build_relation_specs_for_split(
        examples,
        split="Dte",
        max_records_per_relation=int(args.max_records_per_relation),
        relations=DEFAULT_RELATIONS,
    )
    record_sites = _record_sites()

    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=not args.no_flash,
        grad_checkpointing=False,
    )
    clean_runs = _collect_clean_runs(
        model,
        examples,
        sites=record_sites,
        readout=None,
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
        device=device,
    )
    records = run_mediation_records(
        model=model,
        specs=specs,
        examples=examples_by_id,
        clean_runs=clean_runs,
        record_sites=record_sites,
        d_strength=float(args.d_strength),
        r1079_strength=float(args.r1079_strength),
        single_close_token_id=single_close_token_id,
        double_close_token_id=double_close_token_id,
    )
    payload = {
        "model_info": model_info,
        "hypothesis": "X -> D_1249 -> R_1079 -> R_late -> Y",
        "site_policy": "Fixed paper sites plus previously validated R_late handle; no neural-site search.",
        "sites": {
            "D1249": PAPER_D_SITE.site_id,
            "R1079": PAPER_R1079_SITE.site_id,
            "R_late": [site.site_id for site in R_LATE_SITES],
            "record_sites": [site.site_id for site in record_sites],
        },
        "strengths": {
            "D1249": float(args.d_strength),
            "R1079": float(args.r1079_strength),
        },
        "split_summary": split_summary(examples),
        "relation_counts": relation_counts(specs),
        "clean": _clean_summary(examples, clean_runs),
        "records": records,
        "summary": summarize_mediation_records(records),
    }
    _write_json(args.out_dir / "bracket_d1249_r1079_mediation_test.json", payload)
    _write_jsonl(args.out_dir / "heldout" / "records.jsonl", records)
    _write_report(args.out_dir / "bracket_d1249_r1079_mediation_test.md", payload)
    print(json.dumps({"out_dir": str(args.out_dir), "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

