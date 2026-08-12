from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from benchmark.ascend_910c.config import Profile

TIMING_SCOPE = (
    "persistent generator and preallocated output; each measured call includes "
    "Python public API, enqueue, NPU execution, and the repeat's final synchronize; "
    "generator/output creation and destruction are excluded"
)


def measure(operation: Callable[[], Any], profile: Profile) -> dict[str, Any]:
    for _ in range(profile.warmup):
        operation()
    torch.npu.synchronize()

    pilot_start = time.perf_counter()
    operation()
    torch.npu.synchronize()
    pilot_ms = max((time.perf_counter() - pilot_start) * 1000.0, 0.001)
    calls = max(1, min(1000, int(profile.timing_budget_ms / pilot_ms)))

    total_samples: list[float] = []
    enqueue_samples: list[float] = []
    device_samples: list[float] = []
    event_errors: list[str] = []
    for _ in range(profile.repeats):
        torch.npu.synchronize()
        event_start, event_end = _make_events(event_errors)
        if event_start is not None:
            event_start.record()
        start = time.perf_counter()
        for _ in range(calls):
            operation()
        enqueue_end = time.perf_counter()
        if event_end is not None:
            event_end.record()
        torch.npu.synchronize()
        end = time.perf_counter()
        enqueue_samples.append((enqueue_end - start) * 1e6 / calls)
        total_samples.append((end - start) * 1e3 / calls)
        if event_start is not None and event_end is not None:
            try:
                device_samples.append(float(event_start.elapsed_time(event_end)) / calls)
            except Exception as exc:  # Event timing is diagnostic, never the formal metric.
                event_errors.append(f"{type(exc).__name__}: {exc}")

    result: dict[str, Any] = {
        "timing_scope": TIMING_SCOPE,
        "formal_metric": "total_ms_median",
        "timing_calls_per_repeat": calls,
        "enqueue_us_median": statistics.median(enqueue_samples),
        "total_ms_median": statistics.median(total_samples),
        "total_ms_min": min(total_samples),
        "total_ms_max": max(total_samples),
    }
    if device_samples:
        result.update(
            device_ms_median=statistics.median(device_samples),
            device_ms_min=min(device_samples),
            device_ms_max=max(device_samples),
        )
    if event_errors:
        result["device_event_warning"] = event_errors[0]
    return result


def _make_events(errors: list[str]) -> tuple[Any | None, Any | None]:
    try:
        return (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        return None, None
