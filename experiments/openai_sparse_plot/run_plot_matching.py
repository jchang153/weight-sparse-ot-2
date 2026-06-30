from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch

from .effect_signatures import SIGNATURE_FEATURE_BASES, SIGNED_FEATURES, STATE_FEATURE_BASES
from .effect_signatures import restricted_binary_kl_from_margins
from .plot_matching import MatchingResult, cost_matrix, sinkhorn_one_sided_uot
from .plot_matching import fit_matching
from .schema import EffectSignatureTable


EXPECTED_SITE_FAMILIES: dict[str, tuple[str, ...]] = {
    "OpeningQuoteType": ("0.mlp.post_act:863", "0.mlp.post_act:2790"),
    "StoredQuoteType": ("0.mlp.resid_delta:460", "10.attn.act_in:460"),
    "CopiedQuoteTypeAtFinalPosition": ("10.attn.v:663", "10.attn.resid_delta:83", "final_resid:83"),
    "ClosingQuoteLogitPreference": ("10.attn.resid_delta:83", "final_resid:83"),
    "Output": ("final_resid:83",),
}

ABSTRACT_STAGE: dict[str, str] = {
    "OpeningQuoteType": "opening",
    "StoredQuoteType": "storage",
    "CopiedQuoteTypeAtFinalPosition": "copy",
    "ClosingQuoteLogitPreference": "output_write",
    "Output": "output",
}

SITE_STAGE: dict[str, str] = {
    "0.mlp.post_act:863": "opening",
    "0.mlp.post_act:2790": "opening",
    "0.mlp.resid_delta:460": "storage",
    "0.mlp.resid_delta:985": "storage",
    "10.attn.act_in:460": "storage",
    "10.attn.act_in:985": "attention",
    "10.attn.act_in:1013": "attention",
    "10.attn.q:657": "attention",
    "10.attn.k:657": "attention",
    "10.attn.v:663": "copy",
    "10.attn.resid_delta:83": "output_write",
    "final_resid:83": "output",
}

STAGE_DISTANCE: dict[tuple[str, str], float] = {
    ("opening", "storage"): 0.25,
    ("storage", "opening"): 0.25,
    ("storage", "copy"): 0.20,
    ("copy", "storage"): 0.20,
    ("copy", "output_write"): 0.15,
    ("output_write", "copy"): 0.15,
    ("output_write", "output"): 0.10,
    ("output", "output_write"): 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PLOT matchings over OpenAI sparse effect signatures.")
    parser.add_argument(
        "--table-json",
        type=Path,
        default=Path("eval/openai_sparse_plot/effect_signatures/effect_signature_table.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("eval/openai_sparse_plot/plot_matching"))
    parser.add_argument("--top-k", type=int, default=4)
    return parser.parse_args()


def load_table(path: Path) -> EffectSignatureTable:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return table_from_payload(payload)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def table_from_payload(payload: dict[str, Any]) -> EffectSignatureTable:
    data = payload["table"]
    return EffectSignatureTable.from_sequences(
        abstract_variable_ids=data["abstract_variable_ids"],
        neural_site_ids=data["neural_site_ids"],
        abstract_signatures=data["abstract_signatures"],
        neural_signatures=data["neural_signatures"],
        feature_names=data["feature_names"],
        metadata=data.get("metadata", {}),
    )


def subset_table(table: EffectSignatureTable, indices: Sequence[int], *, suffix: str) -> EffectSignatureTable:
    return EffectSignatureTable.from_sequences(
        abstract_variable_ids=table.abstract_variable_ids,
        neural_site_ids=table.neural_site_ids,
        abstract_signatures=[[row[i] for i in indices] for row in table.abstract_signatures],
        neural_signatures=[[row[i] for i in indices] for row in table.neural_signatures],
        feature_names=[table.feature_names[i] for i in indices],
        metadata={**table.metadata, "subset": suffix},
    )


def shuffled_abstract_table(table: EffectSignatureTable, *, seed: int = 0) -> EffectSignatureTable:
    rng = random.Random(int(seed))
    labels = list(table.abstract_variable_ids)
    rng.shuffle(labels)
    return EffectSignatureTable.from_sequences(
        abstract_variable_ids=labels,
        neural_site_ids=table.neural_site_ids,
        abstract_signatures=table.abstract_signatures,
        neural_signatures=table.neural_signatures,
        feature_names=table.feature_names,
        metadata={**table.metadata, "baseline": "shuffled_abstract_labels", "seed": int(seed)},
    )


def random_top_matches(table: EffectSignatureTable, *, seed: int = 0, top_k: int = 4) -> dict[str, list[tuple[str, float]]]:
    rng = random.Random(int(seed))
    out = {}
    for label in table.abstract_variable_ids:
        cols = list(table.neural_site_ids)
        rng.shuffle(cols)
        chosen = cols[: min(int(top_k), len(cols))]
        out[label] = [(col, 1.0 / len(chosen)) for col in chosen]
    return out


def norm_only_matches(table: EffectSignatureTable, *, top_k: int = 4) -> dict[str, list[tuple[str, float]]]:
    norms = torch.tensor(table.neural_signatures, dtype=torch.float32).norm(dim=1)
    vals, idx = torch.topk(norms, k=min(int(top_k), len(table.neural_site_ids)))
    denom = vals.sum().clamp_min(1e-12)
    ranked = [(table.neural_site_ids[int(i)], float(v / denom)) for v, i in zip(vals, idx)]
    return {label: ranked for label in table.abstract_variable_ids}


def _clean_source_effect_row(source_features: Sequence[float], base_features: Sequence[float], *, align_sign: int) -> tuple[float, ...]:
    state_deltas = []
    for name, source_value, base_value in zip(STATE_FEATURE_BASES, source_features, base_features):
        delta = float(source_value) - float(base_value)
        if name in SIGNED_FEATURES:
            delta *= int(align_sign)
        state_deltas.append(delta)
    kl_reduction = restricted_binary_kl_from_margins(source_features[-1], base_features[-1])
    return tuple(state_deltas) + (kl_reduction,)


def _clean_zero_effect_row(base_features: Sequence[float], reference_features: Sequence[float]) -> tuple[float, ...]:
    state_deltas = tuple(abs(float(a) - float(b)) for a, b in zip(reference_features, base_features))
    return state_deltas + (restricted_binary_kl_from_margins(base_features[-1], reference_features[-1]),)


def _mean_vector(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tuple(float(x) for x in tensor.mean(dim=0))


def wrong_content_length_table(table: EffectSignatureTable, payload: dict[str, Any]) -> EffectSignatureTable | None:
    """Build a wrong-abstraction row from same-quote, different-content clean-run variation."""

    prompt_runs = list(payload.get("diagnostics", {}).get("prompt_runs", {}).values())
    rows = []
    for run in prompt_runs:
        if "clean_features" not in run:
            continue
        content_length = run.get("content_length")
        if content_length is None:
            content_length = len(str(run.get("prompt", "")))
        rows.append(
            {
                "quote_type": run.get("quote_type"),
                "content_length": int(content_length),
                "clean_features": tuple(float(x) for x in run["clean_features"]),
            }
        )
    if len(rows) < 4:
        return None

    source_rows = []
    zero_rows = []
    by_quote: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_quote.setdefault(str(row["quote_type"]), []).append(row)

    quote_means = {
        quote: _mean_vector([row["clean_features"] for row in members])
        for quote, members in by_quote.items()
        if members
    }
    for base in rows:
        same_quote_sources = [
            row
            for row in by_quote.get(str(base["quote_type"]), [])
            if row is not base and row["content_length"] != base["content_length"]
        ]
        for source in same_quote_sources:
            align_sign = 1 if source["content_length"] > base["content_length"] else -1
            source_rows.append(
                _clean_source_effect_row(
                    source["clean_features"],
                    base["clean_features"],
                    align_sign=align_sign,
                )
            )
        reference = quote_means.get(str(base["quote_type"]))
        if reference is not None:
            zero_rows.append(_clean_zero_effect_row(base["clean_features"], reference))
    if not source_rows or not zero_rows:
        return None

    wrong_row = _mean_vector(source_rows) + _mean_vector(zero_rows)
    return EffectSignatureTable.from_sequences(
        abstract_variable_ids=("WrongContentLength",),
        neural_site_ids=table.neural_site_ids,
        abstract_signatures=(wrong_row,),
        neural_signatures=table.neural_signatures,
        feature_names=table.feature_names,
        metadata={**table.metadata, "baseline": "wrong_content_length"},
    )


def expected_rank_audit(result: MatchingResult) -> dict[str, dict[str, Any]]:
    audit = {}
    for row_idx, row_label in enumerate(result.row_labels):
        expected = EXPECTED_SITE_FAMILIES.get(row_label, ())
        vals, idx = torch.sort(result.coupling[row_idx], descending=True)
        ranked_labels = [result.col_labels[int(i)] for i in idx]
        best_rank = None
        best_site = None
        best_weight = None
        expected_mass = 0.0
        for site in expected:
            if site not in result.col_labels:
                continue
            col = result.col_labels.index(site)
            expected_mass += float(result.coupling[row_idx, col])
            rank = ranked_labels.index(site) + 1
            weight = float(result.coupling[row_idx, col])
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_site = site
                best_weight = weight
        audit[row_label] = {
            "expected_sites": list(expected),
            "top_site": ranked_labels[0],
            "top_weight": float(vals[0]),
            "best_expected_rank": best_rank,
            "best_expected_site": best_site,
            "best_expected_weight": best_weight,
            "expected_family_mass": expected_mass,
        }
    return audit


def duplicate_abstract_signature_groups(table: EffectSignatureTable) -> list[list[str]]:
    groups: dict[tuple[float, ...], list[str]] = {}
    for label, row in zip(table.abstract_variable_ids, table.abstract_signatures):
        rounded = tuple(round(float(x), 8) for x in row)
        groups.setdefault(rounded, []).append(label)
    return [labels for labels in groups.values() if len(labels) > 1]


def _result_payload(table: EffectSignatureTable, *, top_k: int) -> dict[str, Any]:
    methods = {}
    for method in ("argmin", "softmax", "sinkhorn", "uot"):
        result = fit_matching(
            table,
            method=method,  # type: ignore[arg-type]
            cost_mode="centered_cosine",
            temperature=0.25,
            epsilon=0.25,
            beta_neural=0.25,
            n_iter=300,
        )
        methods[method] = {
            "cost": result.cost.tolist(),
            "coupling": result.coupling.tolist(),
            "top_matches": result.top_matches(top_k=top_k),
            "expected_rank_audit": expected_rank_audit(result),
        }
    return methods


def _stage_penalty_matrix(table: EffectSignatureTable, *, penalty: float) -> torch.Tensor:
    rows = []
    for abstract_id in table.abstract_variable_ids:
        abstract_stage = ABSTRACT_STAGE.get(abstract_id, "")
        row = []
        for site_id in table.neural_site_ids:
            site_stage = SITE_STAGE.get(site_id, "")
            if abstract_stage and site_stage and abstract_stage == site_stage:
                row.append(0.0)
            elif (abstract_stage, site_stage) in STAGE_DISTANCE:
                row.append(float(STAGE_DISTANCE[(abstract_stage, site_stage)]))
            else:
                row.append(float(penalty))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def stage_aware_uot_payload(table: EffectSignatureTable, *, top_k: int, penalty: float = 0.6) -> dict[str, Any]:
    table.validate()
    base_cost = cost_matrix(
        torch.tensor(table.abstract_signatures, dtype=torch.float32),
        torch.tensor(table.neural_signatures, dtype=torch.float32),
        mode="centered_cosine",
    )
    stage_penalty = _stage_penalty_matrix(table, penalty=penalty)
    cost = base_cost + stage_penalty
    coupling = sinkhorn_one_sided_uot(cost, epsilon=0.25, beta_neural=0.25, n_iter=300)
    result = MatchingResult(
        cost=cost.detach().cpu(),
        coupling=coupling.detach().cpu(),
        method="stage_aware_uot",
        row_labels=table.abstract_variable_ids,
        col_labels=table.neural_site_ids,
    )
    return {
        "stage_penalty": stage_penalty.tolist(),
        "cost": result.cost.tolist(),
        "coupling": result.coupling.tolist(),
        "top_matches": result.top_matches(top_k=top_k),
        "expected_rank_audit": expected_rank_audit(result),
        "note": (
            "Stage-aware UOT is a constrained diagnostic: it adds a small prior "
            "favoring sites at the same abstract computational stage. It should "
            "not be read as unconstrained discovery evidence."
        ),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_payload = load_payload(args.table_json)
    table = table_from_payload(source_payload)
    table.validate()

    output_indices = [i for i, name in enumerate(table.feature_names) if name.endswith("binary_quote_margin")]
    if not output_indices:
        raise ValueError("table has no binary_quote_margin features")
    output_only = subset_table(table, output_indices, suffix="output_margin_only")
    shuffled = shuffled_abstract_table(table, seed=0)
    wrong_content = wrong_content_length_table(table, source_payload)

    payload = {
        "source_table": str(args.table_json),
        "metadata": table.metadata,
        "expected_site_families": EXPECTED_SITE_FAMILIES,
        "duplicate_abstract_signature_groups": duplicate_abstract_signature_groups(table),
        "full": _result_payload(table, top_k=args.top_k),
        "stage_aware": stage_aware_uot_payload(table, top_k=args.top_k),
        "output_margin_only": _result_payload(output_only, top_k=args.top_k),
        "baselines": {
            "random_seed0": random_top_matches(table, seed=0, top_k=args.top_k),
            "neural_signature_norm_only": norm_only_matches(table, top_k=args.top_k),
            "shuffled_abstract_labels": _result_payload(shuffled, top_k=args.top_k)["uot"]["top_matches"],
        },
    }
    if wrong_content is not None:
        payload["wrong_content_length"] = _result_payload(wrong_content, top_k=args.top_k)
    (args.out_dir / "plot_matching.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# PLOT Matching Report", ""]
    lines.append(f"- source table: `{args.table_json}`")
    lines.append(f"- abstract variables: `{len(table.abstract_variable_ids)}`")
    lines.append(f"- neural sites: `{len(table.neural_site_ids)}`")
    lines.append(f"- features: `{len(table.feature_names)}`")
    duplicate_groups = payload["duplicate_abstract_signature_groups"]
    if duplicate_groups:
        rendered_groups = "; ".join(", ".join(f"`{x}`" for x in group) for group in duplicate_groups)
        lines.append(f"- duplicate abstract signatures: {rendered_groups}")
    lines.extend(["", "## Full-Signature UOT Top Matches", ""])
    for row, matches in payload["full"]["uot"]["top_matches"].items():
        rendered = ", ".join(f"`{site}` ({weight:.3f})" for site, weight in matches)
        lines.append(f"- `{row}`: {rendered}")
    lines.extend(["", "## Full-Signature UOT Expected-Family Audit", ""])
    for row, audit in payload["full"]["uot"]["expected_rank_audit"].items():
        lines.append(
            f"- `{row}`: top `{audit['top_site']}`; best expected rank `{audit['best_expected_rank']}` "
            f"via `{audit['best_expected_site']}`; expected-family mass `{audit['expected_family_mass']:.3f}`"
        )
    lines.extend(["", "## Stage-Aware UOT Top Matches", ""])
    for row, matches in payload["stage_aware"]["top_matches"].items():
        rendered = ", ".join(f"`{site}` ({weight:.3f})" for site, weight in matches)
        lines.append(f"- `{row}`: {rendered}")
    lines.extend(["", "## Stage-Aware UOT Expected-Family Audit", ""])
    for row, audit in payload["stage_aware"]["expected_rank_audit"].items():
        lines.append(
            f"- `{row}`: top `{audit['top_site']}`; best expected rank `{audit['best_expected_rank']}` "
            f"via `{audit['best_expected_site']}`; expected-family mass `{audit['expected_family_mass']:.3f}`"
        )
    lines.extend(["", "## Output-Margin-Only UOT Top Matches", ""])
    for row, matches in payload["output_margin_only"]["uot"]["top_matches"].items():
        rendered = ", ".join(f"`{site}` ({weight:.3f})" for site, weight in matches)
        lines.append(f"- `{row}`: {rendered}")
    lines.extend(["", "## Baseline: Neural Signature Norm Only", ""])
    for row, matches in payload["baselines"]["neural_signature_norm_only"].items():
        rendered = ", ".join(f"`{site}` ({weight:.3f})" for site, weight in matches)
        lines.append(f"- `{row}`: {rendered}")
    if wrong_content is not None:
        lines.extend(["", "## Wrong-Abstraction Baseline: Content Length", ""])
        for row, matches in payload["wrong_content_length"]["uot"]["top_matches"].items():
            rendered = ", ".join(f"`{site}` ({weight:.3f})" for site, weight in matches)
            lines.append(f"- `{row}`: {rendered}")
    lines.extend(["", "## Notes", ""])
    lines.append(
        "This is the first effect-signature PLOT fit. It is not yet a full causal-scrubbing validation; it ranks native sites for the next intervention tests."
    )
    (args.out_dir / "plot_matching.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote PLOT matching report to {args.out_dir}")


if __name__ == "__main__":
    main()
