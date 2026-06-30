from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .effect_signatures import build_effect_prompt_pairs, build_effect_signature_table
from .plot_matching import fit_matching
from .run_plot_matching import EXPECTED_SITE_FAMILIES, expected_rank_audit, stage_aware_uot_payload
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap PLOT matching stability over string-closing prompt pairs.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default="csp_yolo1")
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/bootstrap_matching"))
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--sample-pairs", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=5)
    parser.add_argument("--min-abs-margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def _summarize(samples: list[dict[str, Any]], *, audit_key: str = "expected_rank_audit") -> dict[str, Any]:
    top_counts: dict[str, Counter[str]] = defaultdict(Counter)
    expected_top1: dict[str, list[float]] = defaultdict(list)
    expected_mass: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        audit = sample[audit_key]
        for var, row in audit.items():
            top_counts[var][row["top_site"]] += 1
            expected = set(EXPECTED_SITE_FAMILIES.get(var, ()))
            expected_top1[var].append(1.0 if row["top_site"] in expected else 0.0)
            expected_mass[var].append(float(row["expected_family_mass"]))

    summary = {}
    n = max(1, len(samples))
    for var in sorted(top_counts):
        summary[var] = {
            "top_site_counts": dict(top_counts[var]),
            "expected_top1_rate": sum(expected_top1[var]) / n,
            "mean_expected_family_mass": sum(expected_mass[var]) / n,
        }
    return summary


def _write_payload(
    *,
    out_path: Path,
    model_info: dict[str, Any],
    input_pairs: int,
    sample_size: int,
    requested_bootstrap_samples: int,
    min_abs_margin: float,
    seed: int,
    samples: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    summary = _summarize(samples)
    stage_aware_summary = _summarize(samples, audit_key="stage_aware_expected_rank_audit")
    payload = {
        "model_info": model_info,
        "input_pairs": input_pairs,
        "sample_pairs": sample_size,
        "bootstrap_samples_requested": int(requested_bootstrap_samples),
        "bootstrap_samples_completed": len(samples),
        "bootstrap_samples": len(samples),
        "completed": bool(completed),
        "min_abs_margin": float(min_abs_margin),
        "seed": int(seed),
        "summary": summary,
        "stage_aware_summary": stage_aware_summary,
        "samples": samples,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_markdown(
    *,
    out_path: Path,
    model: str,
    input_pairs: int,
    sample_size: int,
    requested_bootstrap_samples: int,
    payload: dict[str, Any],
) -> None:
    lines = ["# Bootstrap PLOT Matching Stability", ""]
    lines.append(f"- model: `{model}`")
    lines.append(f"- input pairs: `{input_pairs}`")
    lines.append(f"- sampled pairs per bootstrap: `{sample_size}`")
    lines.append(f"- bootstrap samples requested: `{requested_bootstrap_samples}`")
    lines.append(f"- bootstrap samples completed: `{payload['bootstrap_samples_completed']}`")
    lines.append(f"- completed: `{payload['completed']}`")
    lines.extend(["", "## Expected-Family Stability", ""])
    for var, row in payload["summary"].items():
        counts = ", ".join(f"`{site}`: {count}" for site, count in sorted(row["top_site_counts"].items()))
        lines.append(
            f"- `{var}`: expected top-1 rate `{row['expected_top1_rate']:.3f}`, "
            f"mean expected-family mass `{row['mean_expected_family_mass']:.3f}`, top sites {counts}"
        )
    lines.extend(["", "## Stage-Aware Expected-Family Stability", ""])
    for var, row in payload["stage_aware_summary"].items():
        counts = ", ".join(f"`{site}`: {count}" for site, count in sorted(row["top_site_counts"].items()))
        lines.append(
            f"- `{var}`: expected top-1 rate `{row['expected_top1_rate']:.3f}`, "
            f"mean expected-family mass `{row['mean_expected_family_mass']:.3f}`, top sites {counts}"
        )
    lines.extend(["", "## Interpretation", ""])
    if payload["completed"]:
        lines.append("This is a local CPU bootstrap stability check over the current prompt-pair pool.")
    else:
        lines.append(
            "This is a partial local CPU bootstrap checkpoint. Rerun or resume before using it as the final stability artifact."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    rng = random.Random(int(args.seed))
    print("loading tokenizer/model", flush=True)
    enc = make_tinypython_encoding(args.circuit_home)
    tokens = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    print("model loaded", flush=True)

    base_pairs = list(build_effect_prompt_pairs(max_pairs=args.max_pairs))
    if not base_pairs:
        raise ValueError("no prompt pairs generated")
    sample_size = min(int(args.sample_pairs), len(base_pairs))
    samples = []
    for sample_idx in range(int(args.bootstrap_samples)):
        sampled_pairs = [base_pairs[rng.randrange(len(base_pairs))] for _ in range(sample_size)]
        table, diagnostics = build_effect_signature_table(
            model,
            enc,
            pairs=sampled_pairs,
            single_token_id=tokens["single"],
            double_token_id=tokens["double"],
            device=device,
            min_abs_margin=args.min_abs_margin,
        )
        result = fit_matching(
            table,
            method="uot",
            cost_mode="centered_cosine",
            epsilon=0.25,
            beta_neural=0.25,
            n_iter=300,
        )
        stage_payload = stage_aware_uot_payload(table, top_k=4)
        samples.append(
            {
                "sample_idx": sample_idx,
                "sampled_pair_ids": [left.pair_id for left, _ in sampled_pairs],
                "kept_pair_ids": diagnostics["kept_pair_ids"],
                "top_matches": result.top_matches(top_k=4),
                "expected_rank_audit": expected_rank_audit(result),
                "stage_aware_top_matches": stage_payload["top_matches"],
                "stage_aware_expected_rank_audit": stage_payload["expected_rank_audit"],
            }
        )
        _write_payload(
            out_path=args.out_dir / "bootstrap_matching.partial.json",
            model_info=model_info,
            input_pairs=len(base_pairs),
            sample_size=sample_size,
            requested_bootstrap_samples=int(args.bootstrap_samples),
            min_abs_margin=float(args.min_abs_margin),
            seed=int(args.seed),
            samples=samples,
            completed=False,
        )
        print(f"finished bootstrap sample {sample_idx + 1}/{args.bootstrap_samples}", flush=True)

    payload = _write_payload(
        out_path=args.out_dir / "bootstrap_matching.json",
        model_info=model_info,
        input_pairs=len(base_pairs),
        sample_size=sample_size,
        requested_bootstrap_samples=int(args.bootstrap_samples),
        min_abs_margin=float(args.min_abs_margin),
        seed=int(args.seed),
        samples=samples,
        completed=True,
    )
    _write_payload(
        out_path=args.out_dir / "bootstrap_matching.partial.json",
        model_info=model_info,
        input_pairs=len(base_pairs),
        sample_size=sample_size,
        requested_bootstrap_samples=int(args.bootstrap_samples),
        min_abs_margin=float(args.min_abs_margin),
        seed=int(args.seed),
        samples=samples,
        completed=True,
    )
    _write_markdown(
        out_path=args.out_dir / "bootstrap_matching.md",
        model=args.model,
        input_pairs=len(base_pairs),
        sample_size=sample_size,
        requested_bootstrap_samples=int(args.bootstrap_samples),
        payload=payload,
    )
    print(f"wrote bootstrap matching report to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
