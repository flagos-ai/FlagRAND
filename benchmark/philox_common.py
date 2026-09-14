"""Public Philox invocation, statistical checks and record fields shared by HIP/CUDA matrices."""

from __future__ import annotations

import traceback
from typing import Any

import torch

from flagrand import curand

DTYPES = {
    "raw32": torch.int32,
    "uniform_f32": torch.float32,
    "uniform_f64": torch.float64,
    "normal_f32": torch.float32,
    "normal_f64": torch.float64,
    "lognormal_f32": torch.float32,
    "lognormal_f64": torch.float64,
}


def make_flagrand_generator(seed: int, *, offset: int = 0) -> object:
    return curand.create_generator(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10, seed=seed, offset=offset)


def invoke_flagrand(distribution: str, generator: object, output: torch.Tensor) -> None:
    if distribution == "raw32":
        curand.generate(generator, output)
    elif distribution == "uniform_f32":
        curand.generate_uniform(generator, output)
    elif distribution == "uniform_f64":
        curand.generate_uniform_double(generator, output)
    elif distribution == "normal_f32":
        curand.generate_normal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "normal_f64":
        curand.generate_normal_double(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f32":
        curand.generate_lognormal(generator, output, mean=0.0, stddev=1.0)
    elif distribution == "lognormal_f64":
        curand.generate_lognormal_double(generator, output, mean=0.0, stddev=1.0)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")


def validate_output(output: torch.Tensor, replay: torch.Tensor, distribution: str) -> dict[str, Any]:
    sample = output[: min(output.numel(), 1 << 20)].detach().cpu()
    values = sample.to(torch.float64)
    metrics = _tensor_metrics(values)
    checks = {
        "reproducible": bool(torch.equal(output, replay)),
        "finite": bool(torch.isfinite(values).all().item()),
    }
    if distribution == "raw32":
        checks["not_all_zero"] = bool(torch.any(sample != 0).item())
    elif distribution.startswith("uniform"):
        checks["range"] = metrics["min"] > 0.0 and metrics["max"] <= 1.0
        if sample.numel() >= 4096:
            checks.update(
                mean=abs(metrics["mean"] - 0.5) < 0.04,
                variance=abs(metrics["variance"] - (1.0 / 12.0)) < 0.01,
            )
    elif distribution.startswith("normal"):
        if sample.numel() >= 4096:
            checks.update(mean=abs(metrics["mean"]) < 0.04, std=0.96 < metrics["std"] < 1.04)
    else:
        log_values = torch.log(values)
        checks["positive"] = metrics["min"] > 0.0
        if sample.numel() >= 4096:
            checks.update(
                log_mean=abs(float(log_values.mean().item())) < 0.04,
                log_std=0.96 < float(log_values.std(unbiased=False).item()) < 1.04,
            )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "sample_n": sample.numel(),
        "checks": checks,
        "metrics": metrics,
    }


def _base_record(backend: str, distribution: str, n: int) -> dict[str, Any]:
    return {
        "backend": backend,
        "generator": "philox4x32_10",
        "distribution": distribution,
        "dtype": str(DTYPES[distribution]).removeprefix("torch."),
        "n": n,
        "status": "pending",
    }


def _throughput(record: dict[str, Any], element_size: int, n: int) -> dict[str, float]:
    seconds = float(record["total_ms_median"]) / 1000.0
    return {
        "giga_elements_per_second": n / seconds / 1e9,
        "effective_gb_per_second": n * element_size / seconds / 1e9,
    }


def _tensor_metrics(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "variance": float(values.var(unbiased=False).item()),
    }


def _capture_error(record: dict[str, Any], exc: Exception) -> None:
    record.update(
        status="error",
        error_type=type(exc).__name__,
        error=str(exc),
        traceback_tail=traceback.format_exc().splitlines()[-20:],
    )
