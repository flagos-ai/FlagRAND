from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import torch

from flagrand import curand
from muxi_matrix.config import ALL_DISTRIBUTIONS, Profile
from muxi_matrix.timing import measure


DTYPES = {
    "raw32": torch.int32,
    "uniform_f32": torch.float32,
    "uniform_f64": torch.float64,
    "normal_f32": torch.float32,
    "normal_f64": torch.float64,
    "lognormal_f32": torch.float32,
    "lognormal_f64": torch.float64,
}

_PHILOX_GOLDEN_SEED_0 = (
    1713891541, -1135104948, -513161843, -1694442536,
    -2075847199, 253345875, -259166550, -2080868880,
    107075191, 639715526, -1232967847, 2000445424,
    -257357124, -1092682108, 291018424, -1200667194,
)


def run_preflight(seed: int) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    golden = torch.empty(len(_PHILOX_GOLDEN_SEED_0), device="cuda", dtype=torch.int32)
    invoke("raw32", make_generator(0), golden)
    expected = torch.tensor(_PHILOX_GOLDEN_SEED_0, device="cuda", dtype=torch.int32)
    checks["raw32_seed0_golden_vector"] = bool(torch.equal(golden, expected))

    split_checks: dict[str, bool] = {}
    for distribution in ALL_DISTRIBUTIONS:
        dtype = DTYPES[distribution]
        bulk = torch.empty(4096, device="cuda", dtype=dtype)
        first = torch.empty(2048, device="cuda", dtype=dtype)
        second = torch.empty(2048, device="cuda", dtype=dtype)
        invoke(distribution, make_generator(seed), bulk)
        split_generator = make_generator(seed)
        invoke(distribution, split_generator, first)
        invoke(distribution, split_generator, second)
        split_checks[distribution] = bool(torch.equal(bulk, torch.cat((first, second))))
    checks["bulk_equals_split"] = split_checks
    torch.cuda.synchronize()

    passed = checks["raw32_seed0_golden_vector"] and all(split_checks.values())
    return {"status": "pass" if passed else "fail", "checks": checks}


def run_flagrand_case(
    distribution: str,
    n: int,
    seed: int,
    profile: Profile,
) -> dict[str, Any]:
    record = _base_record("flagrand_public", distribution, n, DTYPES[distribution])
    record["seed"] = seed
    try:
        output = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        invoke(distribution, make_generator(seed), output)
        replay = torch.empty_like(output)
        invoke(distribution, make_generator(seed), replay)
        torch.cuda.synchronize()

        validation = validate_output(output, replay, distribution)
        record["validation"] = validation
        if validation["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = make_generator(seed)
        record.update(measure(lambda: invoke(distribution, timing_generator, timing_output), profile))
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def run_torch_references(n: int, profile: Profile) -> list[dict[str, Any]]:
    cases: tuple[tuple[str, torch.dtype, Callable[[torch.Tensor], Any]], ...] = (
        ("uniform_f32", torch.float32, lambda out: out.uniform_()),
        ("uniform_f64", torch.float64, lambda out: out.uniform_()),
        ("normal_f32", torch.float32, lambda out: out.normal_(0.0, 1.0)),
        ("normal_f64", torch.float64, lambda out: out.normal_(0.0, 1.0)),
        ("lognormal_f32", torch.float32, lambda out: out.log_normal_(0.0, 1.0)),
        ("lognormal_f64", torch.float64, lambda out: out.log_normal_(0.0, 1.0)),
    )
    records = [
        _run_torch_case("torch_rng_reference", name, dtype, n, operation, profile)
        for name, dtype, operation in cases
    ]
    for dtype_name, dtype in (
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("float32", torch.float32),
        ("float64", torch.float64),
    ):
        records.append(
            _run_torch_case(
                "zero_write_reference",
                f"zero_{dtype_name}",
                dtype,
                n,
                lambda out: out.zero_(),
                profile,
            )
        )
    return records


def make_generator(seed: int) -> object:
    return curand.create_generator(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10, seed=seed)


def invoke(distribution: str, generator: object, output: torch.Tensor) -> None:
    operations = {
        "raw32": curand.generate,
        "uniform_f32": curand.generate_uniform,
        "uniform_f64": curand.generate_uniform_double,
        "normal_f32": lambda gen, out: curand.generate_normal(gen, out, mean=0.0, stddev=1.0),
        "normal_f64": lambda gen, out: curand.generate_normal_double(gen, out, mean=0.0, stddev=1.0),
        "lognormal_f32": lambda gen, out: curand.generate_lognormal(gen, out, mean=0.0, stddev=1.0),
        "lognormal_f64": lambda gen, out: curand.generate_lognormal_double(gen, out, mean=0.0, stddev=1.0),
    }
    operations[distribution](generator, output)


def validate_output(output: torch.Tensor, replay: torch.Tensor, distribution: str) -> dict[str, Any]:
    metrics = _tensor_metrics(output)
    checks = {"reproducible": bool(torch.equal(output, replay)), "nonempty": output.numel() > 0}
    if distribution == "raw32":
        checks["not_all_zero"] = bool(torch.any(output != 0).item())
    elif distribution.startswith("uniform"):
        checks.update(
            finite=bool(torch.isfinite(output).all().item()),
            range=metrics["min"] > 0.0 and metrics["max"] <= 1.0,
            mean_sanity=abs(metrics["mean"] - 0.5) < 0.08,
        )
    elif distribution.startswith("normal"):
        checks.update(
            finite=bool(torch.isfinite(output).all().item()),
            mean_sanity=abs(metrics["mean"]) < 0.10,
            std_sanity=0.80 < metrics["std"] < 1.20,
        )
    else:
        log_output = torch.log(output.double())
        checks.update(
            finite=bool(torch.isfinite(output).all().item()),
            positive=metrics["min"] > 0.0,
            log_mean_sanity=abs(float(log_output.mean().item())) < 0.10,
            log_std_sanity=0.80 < float(log_output.std(unbiased=False).item()) < 1.20,
        )
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks, "metrics": metrics}


def _run_torch_case(
    backend: str,
    distribution: str,
    dtype: torch.dtype,
    n: int,
    operation: Callable[[torch.Tensor], Any],
    profile: Profile,
) -> dict[str, Any]:
    record = _base_record(backend, distribution, n, dtype)
    record["comparison_scope"] = (
        "platform_reference_not_algorithm_equivalent"
        if backend == "torch_rng_reference"
        else "write_reference_not_theoretical_bandwidth"
    )
    try:
        output = torch.empty(n, device="cuda", dtype=dtype)
        operation(output)
        torch.cuda.synchronize()
        record["validation"] = {"status": "pass", "metrics": _tensor_metrics(output)}
        record.update(measure(lambda: operation(output), profile))
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def _base_record(backend: str, distribution: str, n: int, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "backend": backend,
        "generator": (
            "philox4x32_10"
            if backend == "flagrand_public"
            else "torch_default" if backend == "torch_rng_reference" else "not_applicable"
        ),
        "distribution": distribution,
        "n": n,
        "dtype": str(dtype).removeprefix("torch."),
        "status": "pending",
    }


def _throughput(record: dict[str, Any], element_size: int, n: int) -> dict[str, float]:
    seconds = float(record["total_ms_median"]) / 1000.0
    return {
        "giga_elements_per_second": n / seconds / 1e9,
        "effective_gb_per_second": n * element_size / seconds / 1e9,
    }


def _tensor_metrics(output: torch.Tensor) -> dict[str, float]:
    values = output.double()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _capture_error(record: dict[str, Any], exc: Exception) -> None:
    record.update(
        status="error",
        error_type=type(exc).__name__,
        error=str(exc),
        traceback_tail=traceback.format_exc().splitlines()[-12:],
    )
