from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .schema import QuoteType, StringClosingExample


ABSTRACT_VARIABLES: tuple[str, ...] = (
    "OpeningQuoteType",
    "StoredQuoteType",
    "CopiedQuoteTypeAtFinalPosition",
    "ClosingQuoteLogitPreference",
    "Output",
)

SIGNATURE_FEATURES: tuple[str, ...] = ABSTRACT_VARIABLES

QUOTE_TO_TOKEN: dict[QuoteType, str] = {"single": "'", "double": '"'}
QUOTE_TO_SIGN: dict[QuoteType, int] = {"single": -1, "double": 1}
SIGN_TO_QUOTE: dict[int, QuoteType] = {-1: "single", 1: "double"}


@dataclass(frozen=True)
class StringClosingState:
    opening_quote_type: int
    stored_quote_type: int
    copied_quote_type: int
    closing_quote_logit_preference: int
    output: int

    def vector(self) -> tuple[float, ...]:
        return (
            float(self.opening_quote_type),
            float(self.stored_quote_type),
            float(self.copied_quote_type),
            float(self.closing_quote_logit_preference),
            float(self.output),
        )


DEFAULT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("assign_call", "value = ({quote}{content}"),
    ("print_call", "print({quote}{content}"),
    ("return_call", "return ({quote}{content}"),
    ("dict_value", "config = {{'name': ({quote}{content}"),
    ("function_arg", "handler(prefix, ({quote}{content}"),
)

DEFAULT_CONTENTS: tuple[str, ...] = (
    "alpha",
    "beta_123",
    "hello world",
    "path/to/file",
    "token sequence",
    "short",
    "longer string content",
    "x_y_z",
)


def quote_type_from_sign(value: int | float) -> QuoteType:
    return "double" if float(value) >= 0 else "single"


def _sign_or_zero(value: int | float) -> int:
    numeric = float(value)
    if numeric == 0:
        return 0
    return 1 if numeric > 0 else -1


def string_closing_state(
    example: StringClosingExample,
    interventions: Mapping[str, int | float] | None = None,
) -> StringClosingState:
    """Evaluate the small symbolic string-closing causal model."""

    forced = dict(interventions or {})
    opening_raw = forced.get("OpeningQuoteType", forced.get("OpenQuoteType", example.sign()))
    opening = _sign_or_zero(opening_raw)
    stored_raw = forced.get(
        "StoredQuoteType",
        forced.get("StoredQuoteTypeAtOpeningPosition", opening),
    )
    stored = _sign_or_zero(stored_raw)
    copied_raw = forced.get("CopiedQuoteTypeAtFinalPosition", stored)
    copied = _sign_or_zero(copied_raw)
    preference_raw = forced.get(
        "ClosingQuoteLogitPreference",
        forced.get("ClosingQuoteLogit", copied),
    )
    preference = _sign_or_zero(preference_raw)
    output = _sign_or_zero(forced.get("Output", preference))

    return StringClosingState(
        opening_quote_type=opening,
        stored_quote_type=stored,
        copied_quote_type=copied,
        closing_quote_logit_preference=preference,
        output=output,
    )


def abstract_effect_signature(
    base: StringClosingExample,
    variable_id: str,
    forced_value: int | float,
) -> tuple[float, ...]:
    if variable_id not in ABSTRACT_VARIABLES:
        raise ValueError(f"unknown abstract variable: {variable_id}")
    before = string_closing_state(base).vector()
    after = string_closing_state(base, {variable_id: forced_value}).vector()
    return tuple(a - b for a, b in zip(after, before))


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def build_matched_pair(
    *,
    template_id: str,
    template: str,
    content: str,
    split: str = "unassigned",
) -> tuple[StringClosingExample, StringClosingExample]:
    pair_id = f"{template_id}-{_stable_id(template_id, content)}"
    examples = []
    for quote_type in ("single", "double"):
        quote = QUOTE_TO_TOKEN[quote_type]  # type: ignore[index]
        alt_type: QuoteType = "single" if quote_type == "double" else "double"  # type: ignore[assignment]
        prompt = template.format(quote=quote, content=content)
        examples.append(
            StringClosingExample(
                example_id=f"{pair_id}-{quote_type}",
                prompt=prompt,
                opening_quote_type=quote_type,  # type: ignore[arg-type]
                target_closing_type=quote_type,  # type: ignore[arg-type]
                target_token=quote,
                alt_token=QUOTE_TO_TOKEN[alt_type],
                content=content,
                template_id=template_id,
                split=split,
                pair_id=pair_id,
                opening_quote_position=prompt.find(quote),
                final_position=len(prompt) - 1,
            )
        )
    return examples[0], examples[1]


def build_balanced_dataset(
    *,
    templates: Sequence[tuple[str, str]] = DEFAULT_TEMPLATES,
    contents: Sequence[str] = DEFAULT_CONTENTS,
    seed: int = 0,
) -> tuple[StringClosingExample, ...]:
    examples: list[StringClosingExample] = []
    for template_id, template in templates:
        for content in contents:
            examples.extend(build_matched_pair(template_id=template_id, template=template, content=content))
    rng = random.Random(int(seed))
    rng.shuffle(examples)
    return tuple(examples)


def split_dataset(
    examples: Sequence[StringClosingExample],
    *,
    seed: int = 0,
    fit_frac: float = 0.5,
    calibration_frac: float = 0.2,
    id_test_frac: float = 0.2,
) -> dict[str, tuple[StringClosingExample, ...]]:
    """Pair-preserving deterministic split for string-closing examples."""

    by_pair: dict[str, list[StringClosingExample]] = {}
    for example in examples:
        by_pair.setdefault(example.pair_id, []).append(example)
    pairs = list(by_pair.items())
    random.Random(int(seed)).shuffle(pairs)

    n = len(pairs)
    n_fit = int(round(fit_frac * n))
    n_cal = int(round(calibration_frac * n))
    n_id = int(round(id_test_frac * n))
    boundaries = {
        "fit": (0, n_fit),
        "calibration": (n_fit, n_fit + n_cal),
        "id_test": (n_fit + n_cal, n_fit + n_cal + n_id),
        "template_heldout": (n_fit + n_cal + n_id, n),
    }

    out: dict[str, tuple[StringClosingExample, ...]] = {}
    for split, (lo, hi) in boundaries.items():
        rows = []
        for _, members in pairs[lo:hi]:
            rows.extend(
                StringClosingExample(
                    **{
                        **member.to_dict(),
                        "split": split,
                        "prompt_tokens": tuple(member.prompt_tokens),
                        "token_ids": tuple(member.token_ids),
                    }
                )
                for member in members
            )
        out[split] = tuple(rows)
    return out
