from __future__ import annotations

import traceback
from typing import Any

import torch

from benchmark.corex_matrix.baseline import CorexCurandGenerator, corex_curand_library_path
from benchmark.corex_matrix.config import COREX_CURAND_DISTRIBUTIONS, Profile
from benchmark.muxi_matrix.timing import measure
from flagrand import curand

DTYPES = {
    "raw32": torch.int32,
    "uniform_f32": torch.float32,
    "uniform_f64": torch.float64,
    "normal_f32": torch.float32,
    "lognormal_f32": torch.float32,
}


def warm_flagrand_compilation_paths(seed: int) -> dict[str, Any]:
    probes = (
        ("raw32", 1, 0),
        ("raw32", 1, 1),
        ("raw32", 1, 2),
        ("raw32", 1, 3),
        ("raw32", 4, 0),
        ("uniform_f32", 1, 0),
        ("uniform_f32", 4, 0),
        ("uniform_f64", 1, 0),
        ("uniform_f64", 1, 1),
        ("uniform_f64", 1, 2),
        ("uniform_f64", 1, 3),
        ("uniform_f64", 2, 0),
        ("normal_f32", 2, 0),
        ("normal_f32", 4, 0),
        ("lognormal_f32", 2, 0),
        ("lognormal_f32", 4, 0),
    )
    for distribution, n, offset in probes:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        invoke_flagrand(distribution, make_flagrand_generator(seed, offset=offset), output)
    torch.cuda.synchronize()
    return {"status": "pass", "probe_count": len(probes)}


def run_flagrand_case(distribution: str, n: int, seed: int, profile: Profile) -> dict[str, Any]:
    record = _base_record("flagrand_public", distribution, n)
    record.update(seed=seed, implementation_path=_implementation_path(distribution))
    try:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        replay = torch.empty_like(output)
        invoke_flagrand(distribution, make_flagrand_generator(seed), output)
        invoke_flagrand(distribution, make_flagrand_generator(seed), replay)
        torch.cuda.synchronize()
        record["validation"] = validate_output(output, replay, distribution)
        if record["validation"]["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = make_flagrand_generator(seed)
        record.update(measure(lambda: invoke_flagrand(distribution, timing_generator, timing_output), profile))
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def run_corex_curand_case(distribution: str, n: int, seed: int, profile: Profile) -> dict[str, Any]:
    record = _base_record("corex_curand", distribution, n)
    record["seed"] = seed
    if distribution not in COREX_CURAND_DISTRIBUTIONS:
        record.update(
            status="unsupported",
            reason="CoreX cuRAND returns CURAND_STATUS_DOUBLE_PRECISION_REQUIRED for float64",
        )
        return record
    try:
        record["library_path"] = corex_curand_library_path()
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        replay = torch.empty_like(output)
        with CorexCurandGenerator(seed=seed) as generator:
            generator.generate(distribution, output)
        with CorexCurandGenerator(seed=seed) as generator:
            generator.generate(distribution, replay)
        torch.cuda.synchronize()
        record["validation"] = validate_output(output, replay, distribution)
        if record["validation"]["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = CorexCurandGenerator(seed=seed)
        try:
            record.update(measure(lambda: timing_generator.generate(distribution, timing_output), profile))
        finally:
            timing_generator.close()
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def annotate_speedup(flagrand_record: dict[str, Any], baseline_record: dict[str, Any]) -> None:
    if flagrand_record["status"] != "pass" or baseline_record["status"] != "pass":
        return
    flagrand_record["speedup_vs_corex_curand"] = float(baseline_record["total_ms_median"]) / float(
        flagrand_record["total_ms_median"]
    )
    flagrand_record["device_speedup_vs_corex_curand"] = float(baseline_record["device_ms_median"]) / float(
        flagrand_record["device_ms_median"]
    )
    baseline_record["speedup_reference_for"] = "flagrand_public"


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
    elif distribution == "lognormal_f32":
        curand.generate_lognormal(generator, output, mean=0.0, stddev=1.0)


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
        if distribution == "uniform_f64" and sample.numel() >= 4096:
            mantissa = sample.view(torch.int64) & ((1 << 52) - 1)
            low29_nonzero_fraction = float(((mantissa & ((1 << 29) - 1)) != 0).double().mean().item())
            checks["full_binary64_resolution"] = low29_nonzero_fraction > 0.99
            metrics["low29_nonzero_fraction"] = low29_nonzero_fraction
    elif distribution == "normal_f32":
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


def _implementation_path(distribution: str) -> str:
    if distribution == "uniform_f64":
        return "corex_triton_philox_integer_ieee754_binary64"
    return "corex_triton_philox_fused_distribution"


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
