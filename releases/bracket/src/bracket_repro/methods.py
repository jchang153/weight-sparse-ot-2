from __future__ import annotations


def r_from_depth(depth: int) -> int:
    if depth < 1:
        raise ValueError("active depth must be positive")
    return int(depth >= 2)


def abstract_r_delta(source_depth: int, base_depth: int) -> float:
    return float(r_from_depth(source_depth) - r_from_depth(base_depth))


def neural_output_delta(*, swapped_margin: float, base_margin: float) -> float:
    return float(swapped_margin) - float(base_margin)


def abstract_joint_rows(*, delta_t2: float, delta_p_one: float, delta_p_two: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (float(delta_t2), float(delta_p_one), float(delta_p_two)),
        (0.0, float(delta_p_one), float(delta_p_two)),
    )
