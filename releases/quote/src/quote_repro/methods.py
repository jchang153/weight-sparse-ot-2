from __future__ import annotations


def quote_code(quote_type: str) -> float:
    if quote_type == "single":
        return -1.0
    if quote_type == "double":
        return 1.0
    raise ValueError(f"unknown quote type: {quote_type}")


def abstract_quote_delta(source_type: str, base_type: str) -> float:
    return quote_code(source_type) - quote_code(base_type)


def neural_quote_delta(*, swapped_margin: float, base_margin: float) -> float:
    return float(swapped_margin) - float(base_margin)
