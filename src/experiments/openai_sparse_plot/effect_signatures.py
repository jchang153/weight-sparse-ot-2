from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation import (
    ChannelSite,
    binary_quote_margin,
    encode_prompt,
    quote_positions_for_prompt,
    record_activations,
)
from .interpreted_circuit import PAPER_BACKED_NODE_SPECS
from .schema import EffectSignatureTable, StringClosingExample
from .string_closing import ABSTRACT_VARIABLES, build_matched_pair, string_closing_state


CORE_SITE_IDS: tuple[str, ...] = tuple(spec.node_id for spec in PAPER_BACKED_NODE_SPECS)

STATE_FEATURE_BASES: tuple[str, ...] = (
    "detector_post_act_quote_balance",
    "detector_post_act_quote_mass",
    "quote_type_channel_460",
    "quote_detector_channel_985",
    "attention_q_final_657",
    "attention_k_opening_657",
    "attention_qk_proxy_657",
    "attention_v_opening_663",
    "copied_quote_channel_83",
    "final_preference_channel_83",
    "binary_quote_margin",
)

EFFECT_EXTRA_FEATURE_BASES: tuple[str, ...] = ("binary_quote_restricted_kl",)

SIGNATURE_FEATURE_BASES: tuple[str, ...] = STATE_FEATURE_BASES + EFFECT_EXTRA_FEATURE_BASES

SIGNED_FEATURES: frozenset[str] = frozenset(
    {
        "detector_post_act_quote_balance",
        "quote_type_channel_460",
        "attention_v_opening_663",
        "copied_quote_channel_83",
        "final_preference_channel_83",
        "binary_quote_margin",
    }
)


DEFAULT_EFFECT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("assign", "x = {quote}{content}"),
    ("print", "print({quote}{content}"),
    ("paren_assign", "value = ({quote}{content}"),
    ("handler_arg", "handler(prefix, ({quote}{content}"),
)

DEFAULT_EFFECT_CONTENTS: tuple[str, ...] = (
    "hello",
    "abc",
    "token sequence",
    "path/to/file",
)


@dataclass(frozen=True)
class PromptRun:
    example: StringClosingExample
    token_ids: torch.Tensor
    positions: dict[str, int | str]
    logits: torch.Tensor
    cache: dict[str, torch.Tensor]
    clean_features: tuple[float, ...]
    margin: float
    predicted_quote_type: str


def interpreted_channel_sites(*, include_post_act: bool = True) -> tuple[ChannelSite, ...]:
    sites = []
    for spec in PAPER_BACKED_NODE_SPECS:
        if not include_post_act and spec.node_id.startswith("0.mlp.post_act"):
            continue
        sites.append(ChannelSite.from_node_id(spec.node_id, label=spec.label))
    return tuple(sites)


def build_effect_prompt_pairs(
    *,
    templates: Sequence[tuple[str, str]] = DEFAULT_EFFECT_TEMPLATES,
    contents: Sequence[str] = DEFAULT_EFFECT_CONTENTS,
    max_pairs: int | None = None,
) -> tuple[tuple[StringClosingExample, StringClosingExample], ...]:
    pairs = []
    for template_id, template in templates:
        for content in contents:
            pairs.append(build_matched_pair(template_id=template_id, template=template, content=content, split="fit"))
            if max_pairs is not None and len(pairs) >= int(max_pairs):
                return tuple(pairs)
    return tuple(pairs)


def _cache_value(cache: Mapping[str, torch.Tensor], hook_key: str, position: int, channel: int) -> float:
    return float(cache[hook_key][0, int(position), int(channel)])


def neural_feature_vector(
    *,
    logits: torch.Tensor,
    cache: Mapping[str, torch.Tensor],
    positions: Mapping[str, int | str],
    single_token_id: int,
    double_token_id: int,
) -> tuple[float, ...]:
    opening = int(positions["opening_quote_position"])
    final = int(positions["final_position"])
    double_detector = _cache_value(cache, "0.mlp.post_act", opening, 863)
    single_detector = _cache_value(cache, "0.mlp.post_act", opening, 2790)
    q_final = _cache_value(cache, "10.attn.q", final, 657)
    k_opening = _cache_value(cache, "10.attn.k", opening, 657)
    return (
        double_detector - single_detector,
        double_detector + single_detector,
        _cache_value(cache, "0.mlp.resid_delta", opening, 460),
        _cache_value(cache, "0.mlp.resid_delta", opening, 985),
        q_final,
        k_opening,
        q_final * k_opening,
        _cache_value(cache, "10.attn.v", opening, 663),
        _cache_value(cache, "10.attn.resid_delta", final, 83),
        _cache_value(cache, "final_resid", final, 83),
        binary_quote_margin(logits, single_token_id=single_token_id, double_token_id=double_token_id),
    )


def abstract_feature_vector(
    example: StringClosingExample,
    interventions: Mapping[str, int | float] | None = None,
) -> tuple[float, ...]:
    state = string_closing_state(example, interventions)
    detector_mass = 1.0 if state.opening_quote_type != 0 else 0.0
    qk_proxy = float(state.copied_quote_type * state.stored_quote_type)
    return (
        float(state.opening_quote_type),
        detector_mass,
        float(state.stored_quote_type),
        float(state.stored_quote_type),
        float(state.copied_quote_type),
        float(state.stored_quote_type),
        qk_proxy,
        float(state.copied_quote_type),
        float(state.copied_quote_type),
        float(state.closing_quote_logit_preference),
        float(state.output),
    )


def _concat_mean(rows: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("need at least one signature row")
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tuple(float(x) for x in tensor.mean(dim=0))


def restricted_binary_kl_from_margins(source_margin: float, target_margin: float) -> float:
    """KL over the two closing-quote tokens from source distribution to target distribution.

    For neural logits, `margin` is exactly logit(double quote) - logit(single quote),
    so sigmoid(margin) is the restricted binary probability of double quote.
    For abstract states, the margin is the symbolic output/preference sign and
    therefore gives a smooth proxy distribution rather than a model probability.
    """

    p_double = torch.sigmoid(torch.tensor(float(source_margin), dtype=torch.float64)).clamp(1e-12, 1.0 - 1e-12)
    q_double = torch.sigmoid(torch.tensor(float(target_margin), dtype=torch.float64)).clamp(1e-12, 1.0 - 1e-12)
    p = torch.stack((1.0 - p_double, p_double))
    q = torch.stack((1.0 - q_double, q_double))
    return float((p * (p.log() - q.log())).sum())


def _source_effect_row(
    *,
    after: Sequence[float],
    before: Sequence[float],
    source: Sequence[float],
    source_sign: int,
) -> tuple[float, ...]:
    state_deltas = []
    for name, a, b in zip(STATE_FEATURE_BASES, after, before):
        delta = float(a) - float(b)
        if name in SIGNED_FEATURES:
            delta *= int(source_sign)
        state_deltas.append(delta)
    before_kl = restricted_binary_kl_from_margins(source[-1], before[-1])
    after_kl = restricted_binary_kl_from_margins(source[-1], after[-1])
    return tuple(state_deltas) + (before_kl - after_kl,)


def _zero_effect_row(
    *,
    after: Sequence[float],
    before: Sequence[float],
) -> tuple[float, ...]:
    state_deltas = tuple(abs(float(a) - float(b)) for a, b in zip(after, before))
    return state_deltas + (restricted_binary_kl_from_margins(before[-1], after[-1]),)


def abstract_signature_for_variable(
    variable_id: str,
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
) -> tuple[float, ...]:
    if variable_id not in ABSTRACT_VARIABLES:
        raise ValueError(f"unknown abstract variable: {variable_id}")
    source_rows = []
    zero_rows = []
    for left, right in pairs:
        for base, source in ((left, right), (right, left)):
            before = abstract_feature_vector(base)
            source_features = abstract_feature_vector(source)
            source_state = string_closing_state(source)
            source_value = {
                "OpeningQuoteType": source_state.opening_quote_type,
                "StoredQuoteType": source_state.stored_quote_type,
                "CopiedQuoteTypeAtFinalPosition": source_state.copied_quote_type,
                "ClosingQuoteLogitPreference": source_state.closing_quote_logit_preference,
                "Output": source_state.output,
            }[variable_id]
            after_source = abstract_feature_vector(base, {variable_id: source_value})
            source_rows.append(
                _source_effect_row(
                    after=after_source,
                    before=before,
                    source=source_features,
                    source_sign=source.sign(),
                )
            )
            after_zero = abstract_feature_vector(base, {variable_id: 0})
            zero_rows.append(_zero_effect_row(after=after_zero, before=before))
    return _concat_mean(source_rows) + _concat_mean(zero_rows)


def site_patch_position(site: ChannelSite, positions: Mapping[str, int | str]) -> int:
    opening = int(positions["opening_quote_position"])
    final = int(positions["final_position"])
    if site.hook_key in {"10.attn.q", "10.attn.resid_delta", "final_resid"}:
        return final
    if site.hook_key == "10.attn.act_in" and site.channel == 1013:
        return final
    return opening


def _run_with_zero_patch_and_record(
    model: Any,
    token_ids: torch.Tensor,
    *,
    site: ChannelSite,
    positions: Sequence[int],
    record_sites: Sequence[ChannelSite],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    def _zero_patch(tensor: torch.Tensor) -> torch.Tensor:
        patched = tensor.clone()
        patched[0, list(positions), site.channel] = 0
        return patched

    regex = "^(?:" + "|".join(sorted({s.hook_key.replace(".", r"\.") for s in record_sites})) + ")$"
    with torch.no_grad():
        with hook_recorder(regex=regex, interventions={site.hook_key: _zero_patch}) as ctx:
            logits, _, _ = model(token_ids)
    return logits, {k: v.detach().cpu() for k, v in ctx.items()}


def _run_with_source_patch_and_record(
    model: Any,
    token_ids: torch.Tensor,
    *,
    site: ChannelSite,
    source_cache: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    source_positions: Sequence[int],
    record_sites: Sequence[ChannelSite],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    from .activation import make_channel_patch

    regex = "^(?:" + "|".join(sorted({s.hook_key.replace(".", r"\.") for s in record_sites})) + ")$"
    interventions = {
        site.hook_key: make_channel_patch(
            site,
            source_cache=source_cache,
            positions=positions,
            source_positions=source_positions,
        )
    }
    with torch.no_grad():
        with hook_recorder(regex=regex, interventions=interventions) as ctx:
            logits, _, _ = model(token_ids)
    return logits, {k: v.detach().cpu() for k, v in ctx.items()}


def collect_clean_runs(
    model: Any,
    enc: Any,
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    *,
    sites: Sequence[ChannelSite],
    single_token_id: int,
    double_token_id: int,
    device: str | torch.device = "cpu",
) -> dict[str, PromptRun]:
    runs = {}
    for pair in pairs:
        for example in pair:
            token_ids = encode_prompt(enc, example.prompt, device=device)
            positions = quote_positions_for_prompt(enc, example.prompt)
            logits, cache = record_activations(model, token_ids, sites)
            features = neural_feature_vector(
                logits=logits.detach().cpu(),
                cache=cache,
                positions=positions,
                single_token_id=single_token_id,
                double_token_id=double_token_id,
            )
            margin = features[-1]
            predicted = "double" if margin > 0 else "single"
            runs[example.example_id] = PromptRun(
                example=example,
                token_ids=token_ids,
                positions=positions,
                logits=logits.detach().cpu(),
                cache=cache,
                clean_features=features,
                margin=margin,
                predicted_quote_type=predicted,
            )
    return runs


def filter_correct_pairs(
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    runs: Mapping[str, PromptRun],
    *,
    min_abs_margin: float = 0.0,
) -> tuple[tuple[StringClosingExample, StringClosingExample], ...]:
    kept = []
    for left, right in pairs:
        ok = True
        for example in (left, right):
            run = runs[example.example_id]
            ok = ok and run.predicted_quote_type == example.opening_quote_type
            ok = ok and abs(run.margin) >= float(min_abs_margin)
        if ok:
            kept.append((left, right))
    return tuple(kept)


def neural_signature_for_site(
    model: Any,
    runs: Mapping[str, PromptRun],
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    *,
    site: ChannelSite,
    record_sites: Sequence[ChannelSite],
    single_token_id: int,
    double_token_id: int,
) -> tuple[float, ...]:
    source_rows = []
    zero_rows = []
    for left, right in pairs:
        for base_ex, source_ex in ((left, right), (right, left)):
            base = runs[base_ex.example_id]
            source = runs[source_ex.example_id]
            base_pos = site_patch_position(site, base.positions)
            source_pos = site_patch_position(site, source.positions)
            patched_logits, patched_cache = _run_with_source_patch_and_record(
                model,
                base.token_ids,
                site=site,
                source_cache=source.cache,
                positions=[base_pos],
                source_positions=[source_pos],
                record_sites=record_sites,
            )
            patched_features = neural_feature_vector(
                logits=patched_logits.detach().cpu(),
                cache=patched_cache,
                positions=base.positions,
                single_token_id=single_token_id,
                double_token_id=double_token_id,
            )
            source_rows.append(
                _source_effect_row(
                    after=patched_features,
                    before=base.clean_features,
                    source=source.clean_features,
                    source_sign=source_ex.sign(),
                )
            )

            zero_logits, zero_cache = _run_with_zero_patch_and_record(
                model,
                base.token_ids,
                site=site,
                positions=[base_pos],
                record_sites=record_sites,
            )
            zero_features = neural_feature_vector(
                logits=zero_logits.detach().cpu(),
                cache=zero_cache,
                positions=base.positions,
                single_token_id=single_token_id,
                double_token_id=double_token_id,
            )
            zero_rows.append(_zero_effect_row(after=zero_features, before=base.clean_features))
    return _concat_mean(source_rows) + _concat_mean(zero_rows)


def build_effect_signature_table(
    model: Any,
    enc: Any,
    *,
    pairs: Sequence[tuple[StringClosingExample, StringClosingExample]],
    single_token_id: int,
    double_token_id: int,
    device: str | torch.device = "cpu",
    min_abs_margin: float = 0.0,
) -> tuple[EffectSignatureTable, dict[str, Any]]:
    sites = interpreted_channel_sites(include_post_act=True)
    runs = collect_clean_runs(
        model,
        enc,
        pairs,
        sites=sites,
        single_token_id=single_token_id,
        double_token_id=double_token_id,
        device=device,
    )
    kept_pairs = filter_correct_pairs(pairs, runs, min_abs_margin=min_abs_margin)
    if not kept_pairs:
        raise ValueError("no prompt pairs survived clean correctness filtering")

    abstract_rows = [abstract_signature_for_variable(var, kept_pairs) for var in ABSTRACT_VARIABLES]
    neural_rows = [
        neural_signature_for_site(
            model,
            runs,
            kept_pairs,
            site=site,
            record_sites=sites,
            single_token_id=single_token_id,
            double_token_id=double_token_id,
        )
        for site in sites
    ]
    feature_names = tuple(f"resample.{name}" for name in SIGNATURE_FEATURE_BASES) + tuple(
        f"zero.{name}" for name in SIGNATURE_FEATURE_BASES
    )
    table = EffectSignatureTable.from_sequences(
        abstract_variable_ids=ABSTRACT_VARIABLES,
        neural_site_ids=tuple(site.site_id for site in sites),
        abstract_signatures=abstract_rows,
        neural_signatures=neural_rows,
        feature_names=feature_names,
        metadata={
            "input_pairs": int(len(pairs)),
            "kept_pairs": int(len(kept_pairs)),
            "site_labels": {site.site_id: site.label for site in sites},
            "min_abs_margin": float(min_abs_margin),
            "signature_note": (
                "Rows concatenate source-resampling effects and zero-ablation effects. "
                "Signed source effects are aligned by source quote sign; zero effects use absolute deltas. "
                "Each block also includes restricted binary KL over the two closing quote tokens."
            ),
        },
    )
    diagnostics = {
        "prompt_runs": {
            key: {
                "prompt": run.example.prompt,
                "quote_type": run.example.opening_quote_type,
                "content": run.example.content,
                "content_length": len(run.example.content),
                "template_id": run.example.template_id,
                "pair_id": run.example.pair_id,
                "margin": run.margin,
                "predicted_quote_type": run.predicted_quote_type,
                "positions": run.positions,
                "clean_features": run.clean_features,
            }
            for key, run in runs.items()
        },
        "kept_pair_ids": [left.pair_id for left, _ in kept_pairs],
    }
    return table, diagnostics


def write_effect_signature_table(table: EffectSignatureTable, diagnostics: Mapping[str, Any], *, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.validate()
    (out / "effect_signature_table.json").write_text(
        json.dumps(
            {
                "table": {
                    "abstract_variable_ids": table.abstract_variable_ids,
                    "neural_site_ids": table.neural_site_ids,
                    "abstract_signatures": table.abstract_signatures,
                    "neural_signatures": table.neural_signatures,
                    "feature_names": table.feature_names,
                    "metadata": table.metadata,
                },
                "diagnostics": diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Effect Signature Table",
        "",
        f"- abstract variables: `{len(table.abstract_variable_ids)}`",
        f"- neural sites: `{len(table.neural_site_ids)}`",
        f"- features: `{len(table.feature_names)}`",
        f"- kept prompt pairs: `{table.metadata.get('kept_pairs')}` / `{table.metadata.get('input_pairs')}`",
        "",
        "## Abstract Rows",
        "",
    ]
    for var, row in zip(table.abstract_variable_ids, table.abstract_signatures):
        norm = float(torch.tensor(row).norm())
        lines.append(f"- `{var}` norm `{norm:.4f}`")
    lines.extend(["", "## Neural Rows", ""])
    labels = table.metadata.get("site_labels", {})
    for site, row in zip(table.neural_site_ids, table.neural_signatures):
        norm = float(torch.tensor(row).norm())
        label = labels.get(site)
        lines.append(f"- `{site}` ({label}) norm `{norm:.4f}`")
    (out / "effect_signature_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
