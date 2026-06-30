from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_QUOTE_JSON = Path("eval/openai_sparse_plot/raw_delta_plot_quote_r6_v2/raw_delta_plot_abstraction.json")
DEFAULT_BRACKET_JSON = Path("eval/openai_sparse_plot/fresh_raw_delta_bracket_fullscan_top12_r6/raw_delta_plot_abstraction.json")
DEFAULT_BRACKET_FULL_SCAN_JSON = Path("eval/openai_sparse_plot/fresh_raw_delta_bracket_full_scan/bracket_raw_delta_scan.json")
DEFAULT_BRACKET_ROBUST_JSON = Path(
    "eval/openai_sparse_plot/bracket_refined_causal_diagnostic_r12_lambda4_records/bracket_refined_causal_diagnostic.json"
)
DEFAULT_OUT_DIR = Path("eval/openai_sparse_plot/localized_plot_report")


PLOT_PAPER_URL = "https://arxiv.org/abs/2605.06979"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated report for OpenAI localized-circuit PLOT runs.")
    parser.add_argument("--quote-json", type=Path, default=DEFAULT_QUOTE_JSON)
    parser.add_argument("--bracket-json", type=Path, default=DEFAULT_BRACKET_JSON)
    parser.add_argument("--bracket-full-scan-json", type=Path, default=DEFAULT_BRACKET_FULL_SCAN_JSON)
    parser.add_argument("--bracket-robust-json", type=Path, default=DEFAULT_BRACKET_ROBUST_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_label(site: str) -> str:
    return site.replace("resid_delta", "resid").replace("post_act", "post")


def _behavior_score(behavior: Mapping[str, float]) -> float:
    return float((behavior["same"] + behavior["flip"] + (1.0 - behavior["wrong_preserve"])) / 3.0)


def _selector_rows(task_payload: Mapping[str, Any], selector_name: str) -> list[dict[str, Any]]:
    rows = task_payload["selector"]["selectors"][selector_name]["ranked_sites"]
    return [dict(row) for row in rows]


def _metric_from_records(records: Sequence[Mapping[str, Any]], metric: str) -> float:
    if metric == "same":
        vals = [float(row["patched_preserves_base_sign"]) for row in records if row["relation"] == "same_depth"]
    elif metric == "flip":
        vals = [float(row["patched_matches_source_sign"]) for row in records if row["relation"] == "different_depth"]
    elif metric == "wrong_preserve":
        vals = [
            float(row["patched_preserves_base_sign"])
            for row in records
            if str(row["relation"]).startswith("wrong_")
        ]
    elif metric == "wrong_reject":
        return 1.0 - _metric_from_records(records, "wrong_preserve")
    elif metric == "shift":
        vals = [float(row["source_signed_shift"]) for row in records if row["relation"] == "different_depth"]
    else:
        raise ValueError(metric)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _bootstrap_metric(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_samples: int,
    rng: random.Random,
) -> dict[str, float]:
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)
    estimates = []
    relation_keys = sorted(by_relation)
    for _ in range(n_samples):
        sample: list[Mapping[str, Any]] = []
        for key in relation_keys:
            rows = by_relation[key]
            sample.extend(rng.choice(rows) for _ in range(len(rows)))
        estimates.append(_metric_from_records(sample, metric))
    lo, mid, hi = np.percentile(np.asarray(estimates, dtype=float), [2.5, 50.0, 97.5])
    return {"lo": float(lo), "mid": float(mid), "hi": float(hi)}


def _bootstrap_handle_records(
    handle_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, (handle_id, records) in enumerate(handle_records.items()):
        rng = random.Random(seed + 1009 * idx)
        out[handle_id] = {
            metric: _bootstrap_metric(records, metric, n_samples=n_samples, rng=rng)
            for metric in ("same", "flip", "wrong_reject", "shift")
        }
    return out


def _hard_records_by_handle(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["handle_id"])].append(row)
    return dict(out)


def _format_ci(ci: Mapping[str, float], digits: int = 3) -> str:
    return f"{ci['mid']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def _behavior_table(task_payload: Mapping[str, Any], task_name: str) -> list[dict[str, Any]]:
    rows = []
    for method in ("raw_squared_uot", "raw_cosine_uot", "raw_cosine_similarity"):
        for calibration in ("raw_cost", "behavior"):
            held = task_payload["heldout_soft_summary"][method][calibration]["behavior"]
            best_key = "calibrated_soft_handles_raw_cost" if calibration == "raw_cost" else "calibrated_soft_handles_behavior"
            best = task_payload[best_key][method]
            rows.append(
                {
                    "task": task_name,
                    "method": method,
                    "calibration": calibration,
                    "k": int(best["k"]),
                    "strength": float(best["strength"]),
                    "sites": list(best["site_ids"]),
                    "same": float(held["same"]),
                    "flip": float(held["flip"]),
                    "wrong_preserve": float(held["wrong_preserve"]),
                    "validity": _behavior_score(held),
                    "shift": float(held["shift"]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            text = ";".join(val) if isinstance(val, list) else str(val)
            vals.append('"' + text.replace('"', '""') + '"')
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_rankings(rows: Sequence[Mapping[str, Any]], *, title: str, out_path: Path, value_key: str = "weight", top_k: int = 8) -> None:
    top = list(rows[:top_k])[::-1]
    labels = [_clean_label(str(row["site_id"])) for row in top]
    values = [float(row[value_key]) for row in top]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(labels, values, color="#4477AA")
    ax.set_title(title)
    ax.set_xlabel(value_key.replace("_", " "))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_dual_quote_rankings(quote_task: Mapping[str, Any], out_path: Path) -> None:
    uot = _selector_rows(quote_task, "raw_cosine_uot")[:8][::-1]
    cos = _selector_rows(quote_task, "raw_cosine_similarity")[:8][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    for ax, rows, title, key in (
        (axes[0], uot, "Cosine-cost UOT", "weight"),
        (axes[1], cos, "Direct cosine ranking", "weight"),
    ):
        labels = [_clean_label(str(row["site_id"])) for row in rows]
        values = [float(row[key]) for row in rows]
        ax.barh(labels, values, color="#4477AA")
        ax.set_title(title)
        ax.set_xlabel("ranking weight")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Quote task: top singleton sites")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_bracket_ablation(
    robust: Mapping[str, Any],
    ci: Mapping[str, Any],
    out_path: Path,
) -> None:
    handles = ["raw_plot_full_top3", "raw_plot_no_final_top2", "raw_plot_final_only"]
    names = ["full top-3", "no final-resid", "final-resid only"]
    metrics = [("same", "same preserve"), ("flip", "different flip"), ("wrong_reject", "wrong-variable reject")]
    x = np.arange(len(handles))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    colors = ["#4477AA", "#66AA55", "#CC6677"]
    for j, (metric, label) in enumerate(metrics):
        mids = [ci[h][metric]["mid"] for h in handles]
        lows = [ci[h][metric]["mid"] - ci[h][metric]["lo"] for h in handles]
        highs = [ci[h][metric]["hi"] - ci[h][metric]["mid"] for h in handles]
        ax.bar(x + (j - 1) * width, mids, width, label=label, color=colors[j], yerr=[lows, highs], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("rate, higher is better")
    ax.set_title("Bracket task: raw PLOT handle ablation with bootstrap intervals")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_method_ablation(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    labels = []
    values = []
    colors = []
    for row in rows:
        if row["method"] == "raw_squared_uot" and row["calibration"] == "raw_cost":
            keep = True
        elif row["method"] == "raw_cosine_uot":
            keep = True
        elif row["method"] == "raw_cosine_similarity" and row["calibration"] == "behavior":
            keep = True
        else:
            keep = False
        if not keep:
            continue
        label = f"{row['task']}\n{row['method'].replace('raw_', '')}\n{row['calibration']}"
        labels.append(label)
        values.append(float(row["validity"]))
        colors.append("#4477AA" if row["task"] == "quote" else "#CC6677")
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(np.arange(len(values)), values, color=colors)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("validity score = average(same, flip, wrong reject)")
    ax.set_title("Ablation: pre-calibration ranking vs full calibrated handles")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    quote: Mapping[str, Any],
    bracket: Mapping[str, Any],
    robust: Mapping[str, Any],
    method_rows: Sequence[Mapping[str, Any]],
    ci: Mapping[str, Any],
    figures: Mapping[str, str],
    args: argparse.Namespace,
) -> None:
    quote_task = quote["soft_runs"]["quote"]
    bracket_task = bracket["soft_runs"]["bracket"]
    quote_best = quote_task["calibrated_soft_handles_behavior"]["raw_cosine_uot"]
    quote_direct = quote_task["calibrated_soft_handles_behavior"]["raw_cosine_similarity"]
    bracket_best = bracket_task["calibrated_soft_handles_behavior"]["raw_cosine_uot"]
    bracket_direct = bracket_task["calibrated_soft_handles_behavior"]["raw_cosine_similarity"]
    robust_soft = robust["soft"]["heldout"]
    hard_held = robust["hard"]["heldout"]

    lines: list[str] = []
    lines.extend(
        [
            "# Raw-Delta PLOT on OpenAI Localized Sparse Circuits",
            "",
            "This report summarizes the latest OpenAI localized-circuit experiments. It is written to be readable without assuming familiarity with every PLOT detail.",
            "",
            "The short conclusion is:",
            "",
            "- Quote task: the full raw-delta PLOT procedure recovers a clean singleton quote-type channel, `0.mlp.resid_delta:460`.",
            "- Bracket task: the same procedure recovers a compact late readout of bracket depth. The result is not just a final-logit artifact: removing `final_resid:1079` still succeeds.",
            "- Direct cosine ranking is a strong simple baseline on these two small one-variable tasks. Here, cosine-UOT and direct cosine often choose the same final handles. The value of the full PLOT procedure is the complete causal workflow: raw effect signatures, matching, top-K handle extraction, calibration, and heldout sensitivity/invariance tests.",
            "",
            f"PLOT reference: [{PLOT_PAPER_URL}]({PLOT_PAPER_URL}). The report follows the paper's Section 4 style: task definition, causal variables, model/candidate sites, learned handles, ablations, and validation.",
            "",
            "## What PLOT Means Here",
            "",
            "A localized circuit gives us a list of candidate neural sites, such as one sparse channel at one hook. PLOT asks whether an abstract causal variable, like quote type or bracket depth, can be matched to one or a few of those neural sites.",
            "",
            "For each base/source pair, we run a swap intervention. A base is the original prompt. A source is another prompt that provides the value to copy into the abstract variable or neural site. We then measure how the output changes.",
            "",
            "The effect signature is the vector of these output changes across many base/source pairs:",
            "",
            "```text",
            "effect signature = phi(y_swap) - phi(y_base)",
            "```",
            "",
            "In these experiments, `phi` is deliberately simple:",
            "",
            "- neural `phi`: the relevant binary logit margin of the localized circuit model;",
            "- abstract `phi`: a signed class output, using `+1` and `-1` for the two possible answers.",
            "",
            "We then compare abstract signatures to neural signatures. The main method is cosine-cost one-sided UOT. We also ablate against direct cosine similarity, squared-cost UOT, raw-cost-only selection, and controls.",
            "",
            "The final PLOT handle is not the raw ranking by itself. Following the PLOT methodology, we keep the top-K sites, renormalize their weights, choose `K` and intervention strength `lambda` on calibration pairs, and then test on heldout pairs.",
            "",
            "## Validation Metrics",
            "",
            "For each candidate handle, we use three plain tests:",
            "",
            "- same-variable preservation: if base and source have the same abstract value, the output should stay the same;",
            "- different-variable flip: if source changes the target abstract variable, the output should flip to the source answer;",
            "- wrong-variable rejection: if source changes irrelevant details, the handle should not behave like a true target-variable swap.",
            "",
            "The best result has same preservation near 1, flip near 1, and wrong-variable preservation near 0.",
            "",
            "## Task 1: Single vs Double Quote Closing",
            "",
            "### Task",
            "",
            "The model sees an unfinished Python-like string and must predict the next closing quote. Examples:",
            "",
            "```text",
            "x = \"hello      -> next token should be \"",
            "x = 'hello      -> next token should be '",
            "print(\"abc     -> next token should be \"",
            "print('abc     -> next token should be '",
            "```",
            "",
            "The abstract causal variable is:",
            "",
            "```text",
            "U = unmatched opening quote type in {single, double}",
            "Y = matching next closing quote",
            "```",
            "",
            "The localized circuit is `csp_yolo1`, using 12 candidate singleton sites from the OpenAI string-closing circuit. The neural margin is `logit(double close quote) - logit(single close quote)`.",
            "",
            f"![Quote ranking]({figures['quote_ranking']})",
            "",
            "### Learned Handle",
            "",
            "The best behavior-calibrated cosine-UOT singleton handle is:",
            "",
            "```text",
            "\n".join(f"{site}" for site in quote_best["site_ids"]),
            "```",
            "",
            f"It uses `K = {quote_best['k']}` and `lambda = {quote_best['strength']}`.",
            "",
            "Heldout behavior:",
            "",
            "| method | sites | same preserve | different flip | wrong preserve | shift |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, best in (("cosine-UOT", quote_best), ("direct cosine", quote_direct)):
        behavior = quote_task["heldout_soft_summary"][best["selector"]]["behavior"]["behavior"]
        lines.append(
            f"| {name} | `{', '.join(best['site_ids'])}` | {behavior['same']:.3f} | {behavior['flip']:.3f} | {behavior['wrong_preserve']:.3f} | {behavior['shift']:.3f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation: this is a clean positive result. The recovered singleton site `0.mlp.resid_delta:460` acts like an internal quote-type channel.",
            "",
            "## Task 2: One vs Two Closing Brackets",
            "",
            "### Task",
            "",
            "The model sees an unfinished Python-like list expression and must decide whether the next closing text should close one level or two levels. Examples from the released samples:",
            "",
            "```text",
            "values =[[5, 3, 11, 3, 12    -> should close with ]]",
            "values =[5, 3, 11, 3, 12     -> should close with ]",
            "```",
            "",
            "The first abstraction is active bracket depth:",
            "",
            "```text",
            "D = active square-bracket depth",
            "Y = one-bracket or two-bracket output",
            "```",
            "",
            "The refined interpretation after the ablation is:",
            "",
            "```text",
            "X -> D_mid -> R_late -> Y",
            "```",
            "",
            "where `D_mid` is an internal parsed-depth state and `R_late` is a late readout expression of that depth decision.",
            "",
            "The localized circuit is `csp_yolo2`. The neural margin is `logit(]]\\n) - logit(]\\n)`.",
            "",
            f"![Bracket ranking]({figures['bracket_ranking']})",
            "",
            "### Raw PLOT Handle",
            "",
            "The behavior-calibrated cosine-UOT handle from the top-12 singleton run is:",
            "",
            "```text",
            "\n".join(f"{site}" for site in bracket_best["site_ids"]),
            "```",
            "",
            f"It uses `K = {bracket_best['k']}` and `lambda = {bracket_best['strength']}`.",
            "",
            "Heldout behavior:",
            "",
            "| method | sites | same preserve | different flip | wrong preserve | shift |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, best in (("cosine-UOT", bracket_best), ("direct cosine", bracket_direct)):
        behavior = bracket_task["heldout_soft_summary"][best["selector"]]["behavior"]["behavior"]
        lines.append(
            f"| {name} | `{', '.join(best['site_ids'])}` | {behavior['same']:.3f} | {behavior['flip']:.3f} | {behavior['wrong_preserve']:.3f} | {behavior['shift']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Final-Residual Ablation",
            "",
            "This ablation tests whether the bracket result is just a final-output artifact. We compare the full top-3 handle, the same handle after removing `final_resid:1079`, and `final_resid:1079` alone.",
            "",
            f"![Bracket ablation]({figures['bracket_ablation']})",
            "",
            "| handle | heldout same | heldout flip | heldout wrong reject | heldout shift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for handle_id in ("raw_plot_full_top3", "raw_plot_no_final_top2", "raw_plot_final_only"):
        lines.append(
            f"| `{handle_id}` | {_format_ci(ci[handle_id]['same'])} | {_format_ci(ci[handle_id]['flip'])} | {_format_ci(ci[handle_id]['wrong_reject'])} | {_format_ci(ci[handle_id]['shift'])} |"
        )

    lines.extend(
        [
            "",
            "The no-final handle still succeeds. Therefore the bracket result is not explained by the final residual site alone. The pair `7.mlp.post_act:4133` and `7.mlp.resid_delta:2041` already carries a valid late depth-readout signal.",
            "",
            "### Earlier Internal-Depth Handles",
            "",
            "Separate hard-handle diagnostics show earlier or more internal depth signals. These are not the final raw-PLOT singleton handle, but they support the refined causal picture.",
            "",
            "| hard handle | same preserve | different flip | wrong preserve | shift |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for handle_id in ("depth_path_1249", "late_depth_signal_core", "late_depth_state_7_mlp_input", "layer1_control_1643"):
        row = hard_held[handle_id]
        wrong = (float(row["wrong_length_preserve_rate"]) + float(row["wrong_content_preserve_rate"])) / 2.0
        lines.append(
            f"| `{handle_id}` | {row['same_depth_preserve_rate']:.3f} | {row['different_depth_flip_rate']:.3f} | {wrong:.3f} | {row['different_depth_mean_source_signed_shift']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Ablations and Simple Baselines",
            "",
            f"![Method ablation]({figures['method_ablation']})",
            "",
            "The table below compares several choices. The validity score is the average of same preservation, different flip, and wrong-variable rejection. Higher is better.",
            "",
            "| task | method | calibration rule | K | lambda | validity | same | flip | wrong preserve | sites |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in method_rows:
        if row["method"] == "raw_cosine_similarity" and row["calibration"] == "raw_cost":
            continue
        sites = ", ".join(row["sites"])
        lines.append(
            f"| {row['task']} | `{row['method']}` | `{row['calibration']}` | {row['k']} | {row['strength']:.1f} | {row['validity']:.3f} | {row['same']:.3f} | {row['flip']:.3f} | {row['wrong_preserve']:.3f} | `{sites}` |"
        )

    lines.extend(
        [
            "",
            "Main ablation takeaways:",
            "",
            "- Direct cosine is a strong baseline here. On both tasks it chooses the same calibrated handle as cosine-UOT, or an equivalent one. These tasks are small and have one target variable, so this is not surprising.",
            "- Raw-cost-only selection is not reliable. In bracket, raw-cost-only cosine selection picks `final_resid:1079` at low strength and fails the flip/wrong-variable tests.",
            "- Squared-cost UOT can be misleading because it is dominated by effect magnitude. It selects different sites and is weaker on bracket unless behavior calibration rescues it partially.",
            "- Control handles fail as expected. For example, `layer1_control_1643` in bracket has flip 0 and wrong-preserve 1.",
            "- The final-only bracket ablation fails, while the no-final internal pair succeeds. This is the strongest evidence that the bracket handle is not just an output-logit shortcut.",
            "",
            "## What Is Nontrivial Here?",
            "",
            "The nontrivial part is not that cosine similarity can rank vectors. The nontrivial part is that the vectors are causal effect signatures built from interventions, and the selected sites are then turned into executable interventions and tested on heldout causal relations.",
            "",
            "For these small one-variable tasks, direct cosine is nearly as good as UOT. So this report should not claim that UOT beats all simple baselines here. The stronger claim is that the raw-delta PLOT workflow gives an automated way to go from a localized circuit to a compact causal handle, with explicit sensitivity and invariance tests. That workflow is the part expected to matter more on larger tasks with multiple abstract variables and many candidate neural sites.",
            "",
            "## Final Claims Supported by the Current Evidence",
            "",
            "Quote:",
            "",
            "```text",
            "PLOT cleanly recovers an internal quote-type residual channel.",
            "Best singleton handle: 0.mlp.resid_delta:460.",
            "```",
            "",
            "Bracket:",
            "",
            "```text",
            "PLOT recovers a compact late depth-readout handle.",
            "The handle is not only final_resid:1079; the internal pair remains valid by itself.",
            "Earlier depth-state handles also exist, but raw output-margin phi naturally prefers late readout sites.",
            "```",
            "",
            "Overall:",
            "",
            "```text",
            "The OpenAI localized-circuit experiments are positive evidence for raw-delta PLOT as an automated causal-handle discovery procedure, with the caveat that direct cosine is a strong baseline in these small one-variable settings.",
            "```",
            "",
            "## Artifact Paths",
            "",
            f"- quote JSON: `{args.quote_json}`",
            f"- bracket top-12 JSON: `{args.bracket_json}`",
            f"- bracket robustness JSON: `{args.bracket_robust_json}`",
            f"- report directory: `{args.out_dir}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    quote = load_json(args.quote_json)
    bracket = load_json(args.bracket_json)
    bracket_full_scan = load_json(args.bracket_full_scan_json)
    robust = load_json(args.bracket_robust_json)

    quote_task = quote["soft_runs"]["quote"]
    bracket_task = bracket["soft_runs"]["bracket"]
    method_rows = _behavior_table(quote_task, "quote") + _behavior_table(bracket_task, "bracket")

    soft_records = {
        handle_id: row["records"]
        for handle_id, row in robust["soft"]["heldout"].items()
        if handle_id in {"raw_plot_full_top3", "raw_plot_no_final_top2", "raw_plot_final_only"}
    }
    ci = _bootstrap_handle_records(soft_records, n_samples=args.bootstrap_samples, seed=args.seed)

    figures = {
        "quote_ranking": "figures/quote_rankings.png",
        "bracket_ranking": "figures/bracket_full_scan_ranking.png",
        "bracket_ablation": "figures/bracket_ablation_ci.png",
        "method_ablation": "figures/method_ablation.png",
    }
    _plot_dual_quote_rankings(quote_task, args.out_dir / figures["quote_ranking"])
    _plot_rankings(
        bracket_full_scan["selector"]["selectors"]["raw_cosine_uot"]["ranked_sites"],
        title="Bracket full singleton scan: cosine-cost UOT",
        out_path=args.out_dir / figures["bracket_ranking"],
        value_key="weight",
        top_k=12,
    )
    _plot_bracket_ablation(robust, ci, args.out_dir / figures["bracket_ablation"])
    _plot_method_ablation(method_rows, args.out_dir / figures["method_ablation"])

    _write_csv(
        args.out_dir / "method_ablation.csv",
        method_rows,
        ["task", "method", "calibration", "k", "strength", "validity", "same", "flip", "wrong_preserve", "shift", "sites"],
    )
    (args.out_dir / "bracket_bootstrap_ci.json").write_text(
        json.dumps(ci, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        args.out_dir / "openai_localized_plot_report.md",
        quote=quote,
        bracket=bracket,
        robust=robust,
        method_rows=method_rows,
        ci=ci,
        figures=figures,
        args=args,
    )
    print(json.dumps({"out_dir": str(args.out_dir), "report": str(args.out_dir / "openai_localized_plot_report.md")}, indent=2))


if __name__ == "__main__":
    main()


