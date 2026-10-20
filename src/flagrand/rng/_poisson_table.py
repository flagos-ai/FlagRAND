"""Poisson probabilities and Walker alias tables, constructed once per lambda.

Weights are evaluated relative to the mode using the Poisson PMF recurrence.
Each omitted tail has a geometric upper bound below 1e-18 times the mode
weight. This avoids a normal approximation and exp(-lambda) underflow.
"""

import math
from functools import lru_cache


@lru_cache(maxsize=32)
def poisson_alias_table(lam: float, raw_range: int = 1 << 32):
    if not math.isfinite(lam) or lam < 0:
        raise ValueError("Poisson lambda must be finite and non-negative.")
    if lam == 0:
        return [raw_range - 1], [0], 0
    mode = math.floor(lam)
    lower = []
    weight = 1.0
    k = mode
    while k:
        weight *= k / lam
        k -= 1
        lower.append(weight)
        ratio = k / lam
        if ratio < 1 and weight * ratio / (1 - ratio) < 1e-18:
            break
    first = mode - len(lower)
    weights = list(reversed(lower)) + [1.0]
    weight = 1.0
    k = mode
    while True:
        k += 1
        weight *= lam / k
        weights.append(weight)
        ratio = lam / (k + 1)
        if ratio < 1 and weight * ratio / (1 - ratio) < 1e-18:
            break
    size = 1 << (len(weights) - 1).bit_length()
    total = math.fsum(weights)
    probability = [w * size / total for w in weights] + [0.0] * (size - len(weights))
    small = [i for i, p in enumerate(probability) if p < 1.0]
    large = [i for i, p in enumerate(probability) if p >= 1.0]
    threshold = [raw_range - 1] * size
    alias = list(range(size))
    while small and large:
        small_index, large_index = small.pop(), large.pop()
        threshold[small_index] = int(probability[small_index] * raw_range)
        alias[small_index] = large_index
        probability[large_index] -= 1.0 - probability[small_index]
        (small if probability[large_index] < 1.0 else large).append(large_index)
    return threshold, alias, first
