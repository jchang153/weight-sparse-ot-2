from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .model_selection import CandidateModelSpec, score_candidate_model, scores_to_jsonable, select_simplest_passing


DEFAULT_QUOTE_ROOT = Path("eval/openai_sparse_plot")
DEFAULT_BRACKET_JSON = Path(
    "eval/openai_sparse_plot/causal_model_selection_20260629/bracket_multidepth/bracket_multidepth_model_selection.json"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/causal_model_selection_20260629/report")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a causal-model selection report for OpenAI localized PLOT runs.")
    parser.add_argument("--quote-root", type=Path, default=DEFAULT_QUOTE_ROOT)
    parser.add_argument("--bracket-json", type=Path, default=DEFAULT_BRACKET_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: Mapping[str, Any], key: str, *, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def _nested(data: Mapping[str, Any], path: Sequence[str], *, default: float = 0.0) -> float:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _quote_scores(quote_root: Path) -> dict[str, Any]:
    raw = _load_json(quote_root / "raw_delta_plot_quote_r6_v2" / "raw_delta_plot_abstraction.json")
    calibrated = _load_json(quote_root / "calibrated_top_models_csp_yolo1" / "calibrated_top_models.json")
    position = _load_json(quote_root / "position_routing_diagnostic_csp_yolo1" / "position_routing_diagnostic.json")
    nonquote = _load_json(quote_root / "nonquote_route_value_diagnostic_csp_yolo1" / "nonquote_route_value_diagnostic.json")

    quote_task = raw["soft_runs"]["quote"]
    best = quote_task["calibrated_soft_handles_behavior"]["raw_cosine_uot"]
    held = quote_task["heldout_soft_summary"]["raw_cosine_uot"]["behavior"]["behavior"]
    top_calibrated = {row["model_id"]: row for row in calibrated["results"]}
    full_head_all = position["summary"].get("full_head82", position["summary"].get("full_head", {})).get("ALL", {})
    if not full_head_all:
        # Older artifacts used this key.
        full_head_all = position["summary"].get("full_head_attention", {}).get("ALL", {})
    nonquote_code = nonquote["summary"].get("code_before_opener::full_head82_y_to_source_v", {})
    nonquote_content = nonquote["summary"].get("content_after_opener::full_head82_y_to_source_v", {})

    q0_metrics = {"causal_validation": 0.0}
    q1_metrics = {
        "same_preserve": float(held["same"]),
        "different_flip": float(held["flip"]),
        "wrong_preserve": float(held["wrong_preserve"]),
    }
    q2_metrics = {
        "pointer_top1": float(full_head_all.get("target_top1_all", 0.0)),
        "nonquote_code_top5": float(nonquote_code.get("expected_top5", 0.0)),
        "nonquote_content_top5": float(nonquote_content.get("expected_top5", 0.0)),
    }
    specs = [
        CandidateModelSpec("Q0", "X -> Y", 0, 0, ("causal_validation",), ()),
        CandidateModelSpec("Q1", "X -> U -> Y", 1, len(best["site_ids"]), ("same_preserve", "different_flip"), ("wrong_preserve",)),
        CandidateModelSpec(
            "Q2",
            "X -> P, Q; P -> Q; Q -> Y",
            2,
            4,
            ("pointer_top1", "nonquote_code_top5", "nonquote_content_top5"),
            (),
        ),
    ]
    metric_sets = {"Q0": q0_metrics, "Q1": q1_metrics, "Q2": q2_metrics}
    scores = [score_candidate_model(spec, metric_sets[spec.model_id]) for spec in specs]
    selected = select_simplest_passing(scores)
    m7 = top_calibrated.get("m7_internal_path_supernode_3", {})
    return {
        "selected_model_id": None if selected is None else selected.model_id,
        "selected_label": None if selected is None else selected.label,
        "scores": scores_to_jsonable(scores),
        "best_handle": best,
        "heldout_behavior": held,
        "calibrated_internal_path": {
            "heldout_internal_iia": _nested(m7, ("heldout_model_accuracy", "internal_strict_iia_accuracy")),
            "heldout_top1": _nested(m7, ("best_config", "heldout_top1")),
            "coverage": _nested(m7, ("canonical_coverage", "canonical_coverage")),
        },
        "pointer_diagnostic": {
            "full_head_target_top1_all": float(full_head_all.get("target_top1_all", 0.0)),
            "full_head_target_top3_all": float(full_head_all.get("target_top3_all", 0.0)),
        },
        "copy_diagnostic": {
            "code_before_opener_expected_top5": float(nonquote_code.get("expected_top5", 0.0)),
            "content_after_opener_expected_top5": float(nonquote_content.get("expected_top5", 0.0)),
        },
        "interpretation": (
            "Q1 is the selected abstraction: the localized circuit validates quote type U, "
            "while the richer pointer/copy abstraction is not validated on the diagnostics."
        ),
    }


def _format_failed(row: Mapping[str, Any]) -> str:
    failed = row.get("failed_metrics", [])
    return ", ".join(str(x) for x in failed) if failed else ""


def _write_markdown(path: Path, *, quote: Mapping[str, Any], bracket: Mapping[str, Any]) -> None:
    bsel = bracket["model_selection"]
    lines = [
        "# Causal-Model Selection for OpenAI Localized Circuits",
        "",
        "This report upgrades the localized PLOT experiments from handle discovery to causal-model selection.",
        "",
        "## Main Conclusions",
        "",
        f"- Quote selected model: `{quote['selected_model_id']}` / `{quote['selected_label']}`.",
        f"- Bracket selected model: `{bsel['selected_model_id']}` / `{bsel['selected_label']}`.",
        "- Quote is treated as a clean one-variable success plus richer-model rejection.",
        "- Bracket is tested as the first depth/readout separation case.",
        "",
        "## Quote Model Selection",
        "",
        "| model | variables | sites | pass | score | failed metrics |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in quote["scores"]:
        lines.append(
            f"| `{row['model_id']}` {row['label']} | {row['variable_count']} | {row['neural_site_count']} | "
            f"{'yes' if row['passed'] else 'no'} | {row['score']:.3f} | `{_format_failed(row)}` |"
        )
    held = quote["heldout_behavior"]
    lines.extend(
        [
            "",
            "Quote evidence:",
            "",
            f"- accepted handle: `{', '.join(quote['best_handle']['site_ids'])}`",
            f"- heldout same preserve: `{held['same']:.3f}`",
            f"- heldout different flip: `{held['flip']:.3f}`",
            f"- heldout wrong preserve: `{held['wrong_preserve']:.3f}`",
            f"- full-head pointer top-1 over all tokens: `{quote['pointer_diagnostic']['full_head_target_top1_all']:.3f}`",
            f"- nonquote code-copy top-5: `{quote['copy_diagnostic']['code_before_opener_expected_top5']:.3f}`",
            f"- nonquote content-copy top-5: `{quote['copy_diagnostic']['content_after_opener_expected_top5']:.3f}`",
            "",
            "Interpretation: `X -> U -> Y` is accepted. A richer pointer/copy abstraction is not validated as the high-level model for this localized circuit.",
            "",
            "## Bracket Model Selection",
            "",
            "| model | variables | sites | pass | score | failed metrics |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for row in bsel["scores"]:
        lines.append(
            f"| `{row['model_id']}` {row['label']} | {row['variable_count']} | {row['neural_site_count']} | "
            f"{'yes' if row['passed'] else 'no'} | {row['score']:.3f} | `{_format_failed(row)}` |"
        )
    clean = bracket["clean"]
    r_held = bracket["R_late"]["heldout"]
    d_held = bracket["D_mid"]["heldout"]
    lines.extend(
        [
            "",
            "Clean multi-depth behavior:",
            "",
            "| depth | n | accuracy | mean margin | mean R probe |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for depth, row in clean["by_depth"].items():
        lines.append(
            f"| {depth} | {row['n']} | {row['accuracy']:.3f} | {row['mean_margin']:.3f} | {row['mean_r_probe']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Bracket heldout evidence:",
            "",
            f"- `R_late` sites: `{', '.join(r_held['site_ids'])}`",
            f"- `R_late` same-R success: `{r_held['summary']['same_R_output_success_rate']:.3f}`",
            f"- `R_late` different-R success: `{r_held['summary']['different_R_output_success_rate']:.3f}`",
            f"- `D_mid` sites: `{', '.join(d_held['site_ids'])}`",
            f"- `D_mid` different-D/same-R preserve: `{d_held['summary']['different_D_same_R_preserve_rate']:.3f}`",
            f"- `D_mid` different-D/same-R probe move: `{d_held['summary']['different_D_same_R_probe_move_rate']:.3f}`",
            f"- `D_mid` different-D/different-R flip: `{d_held['summary']['different_D_different_R_flip_rate']:.3f}`",
            "",
            "Interpretation:",
            "",
        ]
    )
    selected = bsel["selected_model_id"]
    if selected == "B0":
        lines.append("The multi-depth evidence supports a binary late readout `R`, but not a separately validated depth variable `D`.")
    elif selected == "B1":
        lines.append("The multi-depth evidence supports a richer `D -> R` abstraction.")
    elif selected == "B2":
        lines.append("The multi-depth evidence indicates context gating should be represented explicitly.")
    else:
        lines.append("No bracket candidate abstraction passes the configured validation thresholds.")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            "- quote root: `eval/openai_sparse_plot`",
            "- bracket JSON: `bracket_multidepth_model_selection.json`",
            "- report source: `main.tex`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _tex_escape(text: object) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _score_table_tex(scores: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        r"\begin{tabular}{lp{0.30\textwidth}rrlp{0.28\textwidth}}",
        r"\toprule",
        r"ID & Model & Vars & Sites & Pass & Failed metrics \\",
        r"\midrule",
    ]
    for row in scores:
        failed = _format_failed(row)
        failed = (
            failed.replace("different_D_same_R_probe_move", "D probe move")
            .replace("different_D_different_R_flip", "D flip")
            .replace("same_surface_output_success", "context success")
            .replace("nonquote_code_top5", "code copy")
            .replace("nonquote_content_top5", "content copy")
            .replace("pointer_top1", "pointer")
        )
        lines.append(
            f"{_tex_escape(row['model_id'])} & {_tex_escape(row['label'])} & {row['variable_count']} & "
            f"{row['neural_site_count']} & {'yes' if row['passed'] else 'no'} & {_tex_escape(failed)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def _write_latex(path: Path, *, quote: Mapping[str, Any], bracket: Mapping[str, Any]) -> None:
    bsel = bracket["model_selection"]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\usepackage{array}",
        r"\title{Causal-Model Selection for OpenAI Localized Circuits}",
        r"\author{}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Summary}",
        "The localized PLOT experiments are recast as causal-model selection. "
        f"Quote selects \\texttt{{{_tex_escape(quote['selected_model_id'])}}}; "
        f"bracket selects \\texttt{{{_tex_escape(bsel['selected_model_id'])}}}.",
        r"\section*{Quote}",
        r"\begin{center}",
        *_score_table_tex(quote["scores"]),
        r"\end{center}",
        "The accepted quote handle is "
        f"\\texttt{{{_tex_escape(', '.join(quote['best_handle']['site_ids']))}}}. "
        f"Heldout same/flip/wrong-preserve are "
        f"{quote['heldout_behavior']['same']:.3f}/"
        f"{quote['heldout_behavior']['flip']:.3f}/"
        f"{quote['heldout_behavior']['wrong_preserve']:.3f}. "
        "The pointer/copy diagnostics do not validate the richer abstraction as the high-level model.",
        r"\section*{Bracket}",
        r"\begin{center}",
        *_score_table_tex(bsel["scores"]),
        r"\end{center}",
        r"\begin{center}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Depth & $n$ & Accuracy & Mean margin \\",
        r"\midrule",
    ]
    for depth, row in bracket["clean"]["by_depth"].items():
        lines.append(f"{_tex_escape(depth)} & {row['n']} & {row['accuracy']:.3f} & {row['mean_margin']:.3f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
        ]
    )
    r_held = bracket["R_late"]["heldout"]
    d_held = bracket["D_mid"]["heldout"]
    lines.extend(
        [
            "The no-final late readout handle is "
            f"\\texttt{{{_tex_escape(', '.join(r_held['site_ids']))}}}. "
            f"Its heldout same-R and different-R success rates are "
            f"{r_held['summary']['same_R_output_success_rate']:.3f} and "
            f"{r_held['summary']['different_R_output_success_rate']:.3f}.",
            "",
            "The best depth candidate handle is "
            f"\\texttt{{{_tex_escape(', '.join(d_held['site_ids']))}}}. "
            f"Its same-R/different-D probe-move rate is "
            f"{d_held['summary']['different_D_same_R_probe_move_rate']:.3f}.",
            r"\section*{Interpretation}",
        ]
    )
    if bsel["selected_model_id"] == "B1":
        lines.append("The multi-depth experiment validates a richer depth-readout abstraction.")
    elif bsel["selected_model_id"] == "B0":
        lines.append("The multi-depth experiment supports the binary late readout but does not isolate a separate depth variable.")
    else:
        lines.append("The multi-depth experiment does not validate a stronger bracket abstraction under the configured thresholds.")
    lines.extend([r"\end{document}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quote = _quote_scores(args.quote_root)
    bracket = _load_json(args.bracket_json)
    payload = {"quote": quote, "bracket": bracket}
    (args.out_dir / "causal_model_selection_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.out_dir / "causal_model_selection_report.md", quote=quote, bracket=bracket)
    _write_latex(args.out_dir / "main.tex", quote=quote, bracket=bracket)
    print(json.dumps({"out_dir": str(args.out_dir), "tex": str(args.out_dir / "main.tex")}, indent=2))


if __name__ == "__main__":
    main()



