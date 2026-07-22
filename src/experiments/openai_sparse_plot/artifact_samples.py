from __future__ import annotations

from typing import Any, Mapping, Sequence


def trim_trailing_padding(token_ids: Sequence[int], *, pad_token_id: int = 0) -> tuple[int, ...]:
    """Trim right-padding from OpenAI task-sample token rows.

    In the tinypython tokenizer used by the released sparse models, token id
    0 decodes to a printable character. The artifact rows use trailing zeros as
    padding, so local model evaluation must trim them first.
    """

    end = len(token_ids)
    while end > 0 and int(token_ids[end - 1]) == int(pad_token_id):
        end -= 1
    return tuple(int(x) for x in token_ids[:end])


def expected_quote_from_paired_sample_index(index: int, total: int) -> str:
    """Infer quote label from the released task-sample pairing convention."""

    if total <= 0 or total % 2:
        raise ValueError(f"expected an even positive number of paired samples, got {total}")
    if not 0 <= index < total:
        raise IndexError(f"sample index {index} out of range for {total} samples")
    return "double" if index < total // 2 else "single"


def binary_quote_margin(logits: Any, *, single_token_id: int, double_token_id: int) -> float:
    """Return double-minus-single quote logit margin from a final-position vector."""

    return float(logits[int(double_token_id)] - logits[int(single_token_id)])


def prediction_from_margin(margin: float) -> str:
    return "double" if float(margin) > 0 else "single"


def artifact_task_sample_tokens(viz_data: Mapping[str, Any]) -> Any:
    importances = viz_data.get("importances", {})
    if not isinstance(importances, Mapping):
        raise ValueError("viz_data['importances'] is missing or not a mapping")
    task_samples = importances.get("task_samples")
    if not (isinstance(task_samples, tuple) and len(task_samples) == 2):
        raise ValueError("expected importances['task_samples'] to be a (tokens, samples) tuple")
    tokens, _ = task_samples
    return tokens


def evaluate_artifact_task_samples(
    *,
    model: Any,
    enc: Any,
    viz_data: Mapping[str, Any],
    single_token_id: int,
    double_token_id: int,
    device: str = "cpu",
) -> tuple[dict[str, Any], ...]:
    """Evaluate full-model quote preference on released OpenAI task samples."""

    import torch

    tokens = artifact_task_sample_tokens(viz_data)
    rows = []
    total = int(tokens.shape[0])
    with torch.no_grad():
        for index in range(total):
            raw_ids = [int(x) for x in tokens[index].detach().cpu().long().tolist()]
            trimmed = trim_trailing_padding(raw_ids)
            expected = expected_quote_from_paired_sample_index(index, total)
            input_ids = torch.tensor(trimmed, dtype=torch.long, device=device).unsqueeze(0)
            logits, _, _ = model(input_ids)
            final_logits = logits[0, -1].detach().cpu()
            margin = binary_quote_margin(
                final_logits,
                single_token_id=single_token_id,
                double_token_id=double_token_id,
            )
            prediction = prediction_from_margin(margin)
            text = enc.decode(list(trimmed))
            rows.append(
                {
                    "sample_index": index,
                    "raw_length": len(raw_ids),
                    "trimmed_length": len(trimmed),
                    "expected_quote": expected,
                    "predicted_quote": prediction,
                    "correct": prediction == expected,
                    "double_minus_single_margin": margin,
                    "tail": text[-160:],
                }
            )
    return tuple(rows)


def summarize_artifact_sample_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "mean_abs_margin": 0.0,
            "num_margin_lt_1": 0,
            "incorrect_samples": [],
        }
    correct = sum(bool(row.get("correct")) for row in records)
    margins = [float(row.get("double_minus_single_margin", 0.0)) for row in records]
    return {
        "num_samples": n,
        "accuracy": correct / n,
        "mean_abs_margin": sum(abs(x) for x in margins) / n,
        "num_margin_lt_1": sum(abs(x) < 1.0 for x in margins),
        "incorrect_samples": [dict(row) for row in records if not bool(row.get("correct"))],
    }
