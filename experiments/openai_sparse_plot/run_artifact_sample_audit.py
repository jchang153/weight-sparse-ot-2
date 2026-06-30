from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact_samples import evaluate_artifact_task_samples, summarize_artifact_sample_records
from .artifacts import DEFAULT_MODEL, DEFAULT_TASK, candidate_viz_paths, load_viz_data
from .runtime import load_sparse_gpt_model, make_tinypython_encoding, quote_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full model on released OpenAI task samples.")
    parser.add_argument("--circuit-home", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--sweep", default="prune_v2")
    parser.add_argument("--k", default="64")
    parser.add_argument("--viz-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/artifact_sample_audit"))
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.cuda else "cpu"
    viz_path = args.viz_path
    if viz_path is None:
        viz_path = candidate_viz_paths(model=args.model, task=args.task, sweeps=(args.sweep,), ks=(args.k,))[0]

    print("loading artifact/model", flush=True)
    viz_data = load_viz_data(viz_path)
    enc = make_tinypython_encoding(args.circuit_home)
    quote_ids = quote_token_ids(enc)
    model, model_info = load_sparse_gpt_model(
        model_name=args.model,
        circuit_home=args.circuit_home,
        cuda=args.cuda,
        flash=False,
        grad_checkpointing=False,
    )
    records = evaluate_artifact_task_samples(
        model=model,
        enc=enc,
        viz_data=viz_data,
        single_token_id=quote_ids["single"],
        double_token_id=quote_ids["double"],
        device=device,
    )
    summary = summarize_artifact_sample_records(records)
    payload = {
        "model": args.model,
        "task": args.task,
        "sweep": args.sweep,
        "k": args.k,
        "viz_path": viz_path,
        "model_info": model_info,
        "quote_token_ids": quote_ids,
        "summary": summary,
        "records": records,
        "notes": (
            "Task-sample rows are right-padded with token id 0, which decodes as a printable token in tinypython. "
            "Rows are trimmed before evaluation. Expected labels use the artifact pairing convention: first half double, second half single."
        ),
    }
    (args.out_dir / "artifact_sample_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# OpenAI Artifact Task-Sample Audit", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- task: `{args.task}`")
    lines.append(f"- viz artifact: `{viz_path}`")
    lines.append(f"- samples: `{summary['num_samples']}`")
    lines.append(f"- full-model paired-label accuracy: `{summary['accuracy']:.3f}`")
    lines.append(f"- mean absolute double-minus-single margin: `{summary['mean_abs_margin']:.3f}`")
    lines.append(f"- samples with |margin| < 1: `{summary['num_margin_lt_1']}`")
    lines.extend(["", "## Important Handling Detail", ""])
    lines.append(
        "The artifact token rows are padded on the right with token id `0`; in the tinypython tokenizer this decodes to a printable token, so the rows must be trimmed before model evaluation."
    )
    lines.append(
        "The released 32 samples are paired: the first 16 are double-quote variants and the last 16 are single-quote counterparts."
    )
    if summary["incorrect_samples"]:
        lines.extend(["", "## Incorrect Or Ambiguous Samples", ""])
        for row in summary["incorrect_samples"]:
            lines.append(
                f"- sample `{row['sample_index']}`: expected `{row['expected_quote']}`, predicted `{row['predicted_quote']}`, margin `{row['double_minus_single_margin']:.4f}`"
            )
            lines.append(f"  tail: `{row['tail']}`")
    (args.out_dir / "artifact_sample_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote artifact sample audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
