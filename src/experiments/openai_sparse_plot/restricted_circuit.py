from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .activation import binary_quote_margin, encode_prompt


CHANNEL_SITE_RE = re.compile(r"^(?P<hook>[^:]+):(?P<channel>-?\d+)$")


@dataclass(frozen=True)
class RetainedActivationMask:
    """A hook-local activation-channel mask derived from a released circuit artifact."""

    hook_key: str
    retained_channels: tuple[int, ...]


def _as_channel_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    out = []
    for item in value:
        if item == "bias":
            continue
        out.append(int(item))
    return tuple(dict.fromkeys(out))


def retained_masks_from_viz_data(viz_data: Mapping[str, Any]) -> tuple[RetainedActivationMask, ...]:
    circuit_data = viz_data.get("circuit_data", {})
    if not isinstance(circuit_data, Mapping):
        return ()
    masks = []
    for hook_key, raw_channels in sorted(circuit_data.items()):
        if str(hook_key) == "prune_config":
            continue
        channels = _as_channel_tuple(raw_channels)
        if channels:
            masks.append(RetainedActivationMask(hook_key=str(hook_key), retained_channels=channels))
    return tuple(masks)


def retained_masks_from_node_ids(node_ids: Sequence[str]) -> tuple[RetainedActivationMask, ...]:
    grouped: dict[str, list[int]] = {}
    for node_id in node_ids:
        match = CHANNEL_SITE_RE.match(str(node_id))
        if not match:
            raise ValueError(f"node id is not a hook:channel site: {node_id}")
        grouped.setdefault(match.group("hook"), []).append(int(match.group("channel")))
    return tuple(
        RetainedActivationMask(hook_key=hook_key, retained_channels=tuple(dict.fromkeys(channels)))
        for hook_key, channels in sorted(grouped.items())
    )


def make_retain_channel_intervention(mask: RetainedActivationMask):
    retained = torch.tensor(mask.retained_channels, dtype=torch.long)

    def _intervention(tensor: torch.Tensor) -> torch.Tensor:
        patched = torch.zeros_like(tensor)
        valid = retained.to(device=tensor.device)
        valid = valid[(valid >= 0) & (valid < tensor.shape[-1])]
        if valid.numel() == 0:
            return patched
        patched.index_copy_(-1, valid, tensor.index_select(-1, valid))
        return patched

    return _intervention


def _hook_regex(masks: Sequence[RetainedActivationMask]) -> str:
    escaped = [re.escape(mask.hook_key) for mask in masks]
    return "^(?:" + "|".join(sorted(set(escaped))) + ")$"


def run_with_retained_activation_masks(
    model: Any,
    token_ids: torch.Tensor,
    masks: Sequence[RetainedActivationMask],
) -> torch.Tensor:
    from circuit_sparsity.inference.hook_utils import hook_recorder

    interventions = {mask.hook_key: make_retain_channel_intervention(mask) for mask in masks}
    with torch.no_grad():
        with hook_recorder(regex=_hook_regex(masks), interventions=interventions):
            logits, _, _ = model(token_ids)
    return logits


def evaluate_restricted_margin_records(
    model: Any,
    enc: Any,
    prompts: Sequence[str],
    *,
    masks: Sequence[RetainedActivationMask],
    single_token_id: int,
    double_token_id: int,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    records = []
    for prompt in prompts:
        token_ids = encode_prompt(enc, prompt, device=device)
        with torch.no_grad():
            clean_logits, _, _ = model(token_ids)
        restricted_logits = run_with_retained_activation_masks(model, token_ids, masks)
        clean_margin = binary_quote_margin(
            clean_logits.detach().cpu(),
            single_token_id=single_token_id,
            double_token_id=double_token_id,
        )
        restricted_margin = binary_quote_margin(
            restricted_logits.detach().cpu(),
            single_token_id=single_token_id,
            double_token_id=double_token_id,
        )
        expected = "double" if '"' in prompt else "single"
        records.append(
            {
                "prompt": prompt,
                "expected": expected,
                "clean_margin": clean_margin,
                "restricted_margin": restricted_margin,
                "clean_prediction": "double" if clean_margin > 0 else "single",
                "restricted_prediction": "double" if restricted_margin > 0 else "single",
                "restricted_matches_clean": (clean_margin > 0) == (restricted_margin > 0),
                "restricted_matches_expected": ("double" if restricted_margin > 0 else "single") == expected,
            }
        )
    return records


def summarize_margin_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "records": 0,
            "restricted_matches_clean_rate": 0.0,
            "restricted_matches_expected_rate": 0.0,
            "mean_abs_margin_delta": 0.0,
        }
    return {
        "records": len(records),
        "restricted_matches_clean_rate": sum(bool(row["restricted_matches_clean"]) for row in records) / len(records),
        "restricted_matches_expected_rate": sum(bool(row["restricted_matches_expected"]) for row in records)
        / len(records),
        "mean_abs_margin_delta": sum(
            abs(float(row["restricted_margin"]) - float(row["clean_margin"])) for row in records
        )
        / len(records),
    }
