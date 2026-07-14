from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from muxi_matrix.config import Profile


def measure(operation: Callable[[], Any], profile: Profile) -> dict[str, float | int]:
    for _ in range(profile.warmup):
        operation()
    torch.cuda.synchronize()

    pilot_start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    pilot_ms = max((time.perf_counter() - pilot_start) * 1000.0, 0.001)
    calls = max(1, min(1000, int(profile.timing_budget_ms / pilot_ms)))

    total_samples: list[float] = []
    enqueue_samples: list[float] = []
    for _ in range(profile.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(calls):
            operation()
        enqueue_end = time.perf_counter()
        torch.cuda.synchronize()
        end = time.perf_counter()
        enqueue_samples.append((enqueue_end - start) * 1e6 / calls)
        total_samples.append((end - start) * 1e3 / calls)

    return {
        "timing_calls_per_repeat": calls,
        "enqueue_us_median": statistics.median(enqueue_samples),
        "total_ms_median": statistics.median(total_samples),
        "total_ms_min": min(total_samples),
        "total_ms_max": max(total_samples),
    }
