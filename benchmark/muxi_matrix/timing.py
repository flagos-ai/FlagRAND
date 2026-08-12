from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from benchmark.muxi_matrix.config import Profile


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
    device_samples: list[float] = []
    for _ in range(profile.repeats):
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        start = time.perf_counter()
        for _ in range(calls):
            operation()
        enqueue_end = time.perf_counter()
        event_end.record()
        torch.cuda.synchronize()
        end = time.perf_counter()
        enqueue_samples.append((enqueue_end - start) * 1e6 / calls)
        total_samples.append((end - start) * 1e3 / calls)
        device_samples.append(event_start.elapsed_time(event_end) / calls)

    return {
        "timing_calls_per_repeat": calls,
        "enqueue_us_median": statistics.median(enqueue_samples),
        "total_ms_median": statistics.median(total_samples),
        "total_ms_min": min(total_samples),
        "total_ms_max": max(total_samples),
        "device_ms_median": statistics.median(device_samples),
        "device_ms_min": min(device_samples),
        "device_ms_max": max(device_samples),
    }
