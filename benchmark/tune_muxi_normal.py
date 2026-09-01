from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import torch

from benchmark.muxi_matrix.config import ACCEPTANCE_DISTRIBUTION_SIZES, Profile
from benchmark.muxi_matrix.mcrand import McrandGenerator
from benchmark.muxi_matrix.timing import measure
from flagrand import curand

CONFIGS = tuple((block, warps) for block in (128, 256, 512, 1024) for warps in (2, 4, 8))
DISTRIBUTIONS = ("normal_f32", "normal_f64", "lognormal_f32", "lognormal_f64")
DTYPES = {
    "normal_f32": torch.float32,
    "normal_f64": torch.float64,
    "lognormal_f32": torch.float32,
    "lognormal_f64": torch.float64,
}
PROFILE = Profile(name="muxi_tune", sizes=(), warmup=8, repeats=5, timing_budget_ms=20.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep MetaX Philox normal/lognormal launch configurations.")
    parser.add_argument("--distributions", default=",".join(DISTRIBUTIONS))
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()
    distributions = tuple(name.strip() for name in args.distributions.split(",") if name.strip())

    for distribution in distributions:
        shapes = ACCEPTANCE_DISTRIBUTION_SIZES[distribution]
        baselines = {n: _measure_mcrand(distribution, n) for n in shapes}
        rows = []
        for block, warps in CONFIGS:
            timings = {
                n: _measure_flagrand(distribution, n, block_size=block, num_warps=warps) for n in shapes
            }
            speedups = [baselines[n] / timings[n] for n in shapes]
            rows.append(
                {
                    "block": block,
                    "warps": warps,
                    "api_speedup": sum(speedups) / len(speedups),
                    "min_speedup": min(speedups),
                    "max_speedup": max(speedups),
                }
            )
        rows.sort(key=lambda row: row["api_speedup"], reverse=True)
        for rank, row in enumerate(rows[: args.top], start=1):
            print(
                f"TUNE {distribution:<15} rank={rank} BLOCK={row['block']:<4} "
                f"warps={row['warps']} api={row['api_speedup']:.3f}x "
                f"min={row['min_speedup']:.3f}x max={row['max_speedup']:.3f}x"
            )
    return 0


def _measure_mcrand(distribution: str, n: int) -> float:
    output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
    generator = McrandGenerator(seed=1234)
    try:
        result = measure(lambda: generator.generate(distribution, output), PROFILE)
    finally:
        generator.close()
    return float(result["total_ms_median"])


def _measure_flagrand(distribution: str, n: int, *, block_size: int, num_warps: int) -> float:
    output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
    generator = curand.create_generator(seed=1234)
    operation = _operation(distribution)
    result = measure(
        lambda: operation(
            generator,
            output,
            mean=0.0,
            stddev=1.0,
            block_size=block_size,
            num_warps=num_warps,
        ),
        PROFILE,
    )
    return float(result["total_ms_median"])


def _operation(distribution: str) -> Callable[..., Any]:
    return {
        "normal_f32": curand.generate_normal,
        "normal_f64": curand.generate_normal_double,
        "lognormal_f32": curand.generate_lognormal,
        "lognormal_f64": curand.generate_lognormal_double,
    }[distribution]


if __name__ == "__main__":
    raise SystemExit(main())
