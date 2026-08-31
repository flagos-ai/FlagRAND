from __future__ import annotations

import traceback
from typing import Any

import torch

from benchmark.nvidia_matrix.config import ALL_DISTRIBUTIONS, Profile
from benchmark.nvidia_matrix.curand import NvidiaCurandGenerator, nvidia_curand_library_path
from benchmark.nvidia_matrix.timing import measure
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


def warm_flagrand_compilation_paths(seed: int) -> dict[str, Any]:
    probes = []
    for distribution in ALL_DISTRIBUTIONS:
        n = 8 if distribution.startswith(("normal", "lognormal")) else 7
        for offset in (0, 1, 2, 3):
            output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
            invoke_flagrand(distribution, make_flagrand_generator(seed, offset=offset), output)
            probes.append((distribution, n, offset))
    torch.cuda.synchronize()
    return {"status": "pass", "probe_count": len(probes)}


def run_native_compatibility_preflight(seed: int) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for offset in range(8):
        expected = torch.empty(17, device="cuda", dtype=torch.int32)
        actual = torch.empty_like(expected)
        with NvidiaCurandGenerator(seed=seed, offset=offset) as generator:
            generator.generate("raw32", expected)
        invoke_flagrand("raw32", make_flagrand_generator(seed, offset=offset), actual)
        checks[f"raw32_offset_{offset}"] = bool(torch.equal(expected, actual))
    torch.cuda.synchronize()
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "curand_library": nvidia_curand_library_path(),
    }


def run_flagrand_case(distribution: str, n: int, seed: int, profile: Profile) -> dict[str, Any]:
    record = _base_record("flagrand_public", distribution, n)
    record.update(seed=seed, implementation_path="nvidia_triton_philox")
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


def run_nvidia_curand_case(distribution: str, n: int, seed: int, profile: Profile) -> dict[str, Any]:
    record = _base_record("nvidia_curand", distribution, n)
    record.update(seed=seed, library_path=nvidia_curand_library_path())
    try:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        replay = torch.empty_like(output)
        with NvidiaCurandGenerator(seed=seed) as generator:
            generator.generate(distribution, output)
        with NvidiaCurandGenerator(seed=seed) as generator:
            generator.generate(distribution, replay)
        torch.cuda.synchronize()
        record["validation"] = validate_output(output, replay, distribution)
        if record["validation"]["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = NvidiaCurandGenerator(seed=seed)
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
    flagrand_record["speedup_vs_nvidia_curand"] = float(baseline_record["total_ms_median"]) / float(
        flagrand_record["total_ms_median"]
    )
    flagrand_record["device_speedup_vs_nvidia_curand"] = float(baseline_record["device_ms_median"]) / float(
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
