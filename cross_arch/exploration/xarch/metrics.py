from __future__ import annotations

import math
from typing import Any, Iterable


def flatten(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        result: list[float] = []
        for child in value:
            result.extend(flatten(child))
        return result
    raise TypeError(f"unsupported tensor surrogate: {type(value)!r}")


def subtract(left: list[float], right: list[float]) -> list[float]:
    if len(left) != len(right):
        raise ValueError("tensor surrogates have different flattened lengths")
    return [a - b for a, b in zip(left, right)]


def mean_square(value: list[float]) -> float:
    if not value:
        raise ValueError("empty tensor surrogate")
    return sum(x * x for x in value) / len(value)


def cosine(left: list[float], right: list[float], epsilon: float = 1e-8) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / max(norm_left * norm_right, epsilon)


def four_condition_metrics(
    *,
    s_clean: Any,
    p_clean: Any,
    sp_clean: Any,
    s_output: Any,
    p_output: Any,
    sp_output: Any,
    target: Any,
) -> dict[str, float]:
    s0, p0, sp0 = map(flatten, [s_clean, p_clean, sp_clean])
    s, p, sp, target_vector = map(flatten, [s_output, p_output, sp_output, target])
    d_s = subtract(s, s0)
    d_p = subtract(p, p0)
    d_sp = subtract(sp, sp0)
    interaction = subtract(subtract(d_sp, d_s), d_p)
    energy = max(mean_square(target_vector), 1e-7)
    target_error = subtract(d_sp, target_vector)
    interaction_error = subtract(interaction, target_vector)
    return {
        "sp_nmse": mean_square(target_error) / energy,
        "interaction_nmse": mean_square(interaction_error) / energy,
        "sp_cosine": cosine(d_sp, target_vector),
        "interaction_cosine": cosine(interaction, target_vector),
        "s_drift_ratio": mean_square(d_s) / energy,
        "p_drift_ratio": mean_square(d_p) / energy,
        "sp_amplitude": math.sqrt(mean_square(d_sp) / energy),
    }

