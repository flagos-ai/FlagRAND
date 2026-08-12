from __future__ import annotations

import traceback
from typing import Any

import torch

from benchmark.ascend_910c.baseline import CannAclnnGenerator, baseline_metadata
from benchmark.ascend_910c.config import Profile
from benchmark.ascend_910c.timing import measure
from flagrand import curand
from flagrand.ascend.huawei910c import LOGNORMAL_BLOCK, NORMAL_BLOCK, RAW_BLOCK, UNIFORM_BLOCK

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
    1713891541,
    -1135104948,
    -513161843,
    -1694442536,
    -2075847199,
    253345875,
    -259166550,
    -2080868880,
    107075191,
    639715526,
    -1232967847,
    2000445424,
    -257357124,
    -1092682108,
    291018424,
    -1200667194,
)


def run_preflight(
    seed: int,
    device: str,
    *,
    distributions: tuple[str, ...],
    block_size: int,
    num_warps: int,
    with_baseline: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    raw_probe_n = 65536
    golden = torch.empty(raw_probe_n, device=device, dtype=torch.int32)
    raw_replay = torch.empty_like(golden)
    invoke_flagrand(
        "raw32",
        make_flagrand_generator(0),
        golden,
        block_size=block_size,
        num_warps=num_warps,
    )
    invoke_flagrand(
        "raw32",
        make_flagrand_generator(0),
        raw_replay,
        block_size=block_size,
        num_warps=num_warps,
    )
    raw_cpu = golden.cpu()
    negative_fraction = float((raw_cpu < 0).double().mean().item())
    checks["raw32_replay"] = bool(torch.equal(golden, raw_replay))
    checks["raw32_signed_balance"] = {
        "status": 0.47 < negative_fraction < 0.53,
        "negative_fraction": negative_fraction,
    }
    checks["raw32_seed0_golden_vector"] = {
        "status": "not_applicable",
        "reason": "Huawei performance path is statistical, not cuRAND-bitwise",
    }

    split_checks: dict[str, bool] = {}
    for distribution in distributions:
        dtype = DTYPES[distribution]
        bulk_n, first_n = _preflight_split_shape(distribution)
        bulk = torch.empty(bulk_n, device=device, dtype=dtype)
        first = torch.empty(first_n, device=device, dtype=dtype)
        second = torch.empty(bulk_n - first_n, device=device, dtype=dtype)
        invoke_flagrand(
            distribution,
            make_flagrand_generator(seed),
            bulk,
            block_size=block_size,
            num_warps=num_warps,
        )
        split_generator = make_flagrand_generator(seed)
        invoke_flagrand(
            distribution,
            split_generator,
            first,
            block_size=block_size,
            num_warps=num_warps,
        )
        invoke_flagrand(
            distribution,
            split_generator,
            second,
            block_size=block_size,
            num_warps=num_warps,
        )
        split_checks[distribution] = bool(torch.equal(bulk[:first_n], first) and torch.equal(bulk[first_n:], second))
    checks["bulk_equals_split"] = split_checks

    f64_contract: dict[str, Any] = {}
    for distribution in distributions:
        if distribution.endswith("f64"):
            probe = torch.empty(65536, device=device, dtype=torch.float64)
            invoke_flagrand(
                distribution,
                make_flagrand_generator(seed),
                probe,
                block_size=block_size,
                num_warps=num_warps,
            )
            f64_contract[distribution] = _f64_precision_contract(probe, distribution)
    if f64_contract:
        checks["huawei_f64_precision_contract"] = f64_contract

    if with_baseline:
        baseline_checks: dict[str, bool] = {}
        for distribution in distributions:
            n = 18 if "normal" in distribution else 17
            first = torch.empty(n, device=device, dtype=DTYPES[distribution])
            replay = torch.empty_like(first)
            CannAclnnGenerator(seed).generate(distribution, first)
            CannAclnnGenerator(seed).generate(distribution, replay)
            baseline_checks[distribution] = bool(torch.equal(first, replay))
        checks["cann_aclnn_replay"] = baseline_checks
    torch.npu.synchronize()

    split_required = {name: value for name, value in split_checks.items() if not name.endswith("f64")}
    passed = (
        checks["raw32_replay"]
        and checks["raw32_signed_balance"]["status"]
        and all(split_required.values())
        and all(item["status"] for item in checks.get("huawei_f64_precision_contract", {}).values())
    )
    if with_baseline:
        passed = passed and all(checks["cann_aclnn_replay"].values())
    return {"status": "pass" if passed else "fail", "checks": checks}


def run_flagrand_case(
    distribution: str,
    n: int,
    seed: int,
    device: str,
    profile: Profile,
    *,
    block_size: int,
    num_warps: int,
) -> dict[str, Any]:
    record = _base_record("flagrand_public", distribution, n, DTYPES[distribution])
    record.update(
        seed=seed,
        implementation_path=(
            "cann_native_huawei_f64_rng_compatibility"
            if distribution.endswith("f64")
            else "triton_ascend_910c_counter_hash_fused_distribution"
        ),
        implementation_independent_of_baseline=not distribution.endswith("f64"),
        requested_block_size=block_size,
        kernel_block_size=(
            None
            if distribution.endswith("f64")
            else (
                NORMAL_BLOCK
                if distribution == "normal_f32"
                else (
                    LOGNORMAL_BLOCK
                    if distribution == "lognormal_f32"
                    else (UNIFORM_BLOCK if distribution == "uniform_f32" else RAW_BLOCK)
                )
            )
        ),
        num_warps=num_warps,
    )
    try:
        output = torch.empty(n, device=device, dtype=DTYPES[distribution])
        invoke_flagrand(
            distribution,
            make_flagrand_generator(seed),
            output,
            block_size=block_size,
            num_warps=num_warps,
        )
        replay = torch.empty_like(output)
        invoke_flagrand(
            distribution,
            make_flagrand_generator(seed),
            replay,
            block_size=block_size,
            num_warps=num_warps,
        )
        torch.npu.synchronize()
        record["validation"] = validate_output(output, replay, distribution)
        if record["validation"]["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = make_flagrand_generator(seed)
        record.update(
            measure(
                lambda: invoke_flagrand(
                    distribution,
                    timing_generator,
                    timing_output,
                    block_size=block_size,
                    num_warps=num_warps,
                ),
                profile,
            )
        )
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def run_cann_case(
    distribution: str,
    n: int,
    seed: int,
    device: str,
    profile: Profile,
) -> dict[str, Any]:
    record = _base_record("cann_aclnn", distribution, n, DTYPES[distribution])
    record.update(seed=seed, **baseline_metadata(distribution))
    try:
        output = torch.empty(n, device=device, dtype=DTYPES[distribution])
        CannAclnnGenerator(seed).generate(distribution, output)
        replay = torch.empty_like(output)
        CannAclnnGenerator(seed).generate(distribution, replay)
        torch.npu.synchronize()
        record["validation"] = validate_output(output, replay, distribution)
        if record["validation"]["status"] != "pass":
            record["status"] = "validation_fail"
            return record

        timing_output = torch.empty_like(output)
        timing_generator = CannAclnnGenerator(seed)
        record.update(measure(lambda: timing_generator.generate(distribution, timing_output), profile))
        record.update(_throughput(record, output.element_size(), n))
        record["status"] = "pass"
    except Exception as exc:
        _capture_error(record, exc)
    return record


def make_flagrand_generator(seed: int) -> object:
    return curand.create_generator(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10, seed=seed)


def invoke_flagrand(
    distribution: str,
    generator: object,
    output: torch.Tensor,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    options = {"block_size": block_size, "num_warps": num_warps}
    if distribution == "raw32":
        curand.generate(generator, output, **options)
    elif distribution == "uniform_f32":
        curand.generate_uniform(generator, output, **options)
    elif distribution == "uniform_f64":
        curand.generate_uniform_double(generator, output, **options)
    elif distribution == "normal_f32":
        curand.generate_normal(generator, output, mean=0.0, stddev=1.0, **options)
    elif distribution == "normal_f64":
        curand.generate_normal_double(generator, output, mean=0.0, stddev=1.0, **options)
    elif distribution == "lognormal_f32":
        curand.generate_lognormal(generator, output, mean=0.0, stddev=1.0, **options)
    elif distribution == "lognormal_f64":
        curand.generate_lognormal_double(generator, output, mean=0.0, stddev=1.0, **options)
    else:
        raise ValueError(f"Unsupported FlagRAND distribution: {distribution}")


def validate_output(
    output: torch.Tensor,
    replay: torch.Tensor,
    distribution: str,
) -> dict[str, Any]:
    values = output.detach().cpu().to(torch.float64)
    metrics = _tensor_metrics(values)
    checks = {
        "reproducible": bool(torch.equal(output, replay)),
        "nonempty": output.numel() > 0,
    }
    if distribution == "raw32":
        checks["not_all_zero"] = bool(torch.any(values != 0).item())
    elif distribution.startswith("uniform"):
        checks.update(
            finite=bool(torch.isfinite(values).all().item()),
            range=metrics["min"] > 0.0 and metrics["max"] <= 1.0,
        )
        if output.numel() >= 4096:
            checks["mean_sanity"] = abs(metrics["mean"] - 0.5) < 0.08
    elif distribution.startswith("normal"):
        checks["finite"] = bool(torch.isfinite(values).all().item())
        if output.numel() >= 4096:
            checks.update(
                mean_sanity=abs(metrics["mean"]) < 0.10,
                std_sanity=0.80 < metrics["std"] < 1.20,
            )
    else:
        log_values = torch.log(values)
        checks.update(
            finite=bool(torch.isfinite(values).all().item()),
            positive=metrics["min"] > 0.0,
        )
        if output.numel() >= 4096:
            checks.update(
                log_mean_sanity=abs(float(log_values.mean().item())) < 0.10,
                log_std_sanity=(0.80 < float(log_values.std(unbiased=False).item()) < 1.20),
            )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "metrics": metrics,
    }


def annotate_speedup(
    flagrand_record: dict[str, Any],
    baseline_record: dict[str, Any],
) -> None:
    if flagrand_record["status"] != "pass" or baseline_record["status"] != "pass":
        return
    flagrand_record["speedup_vs_cann_aclnn"] = float(baseline_record["total_ms_median"]) / float(
        flagrand_record["total_ms_median"]
    )
    if "device_ms_median" in flagrand_record and "device_ms_median" in baseline_record:
        flagrand_record["device_speedup_vs_cann_aclnn"] = float(baseline_record["device_ms_median"]) / float(
            flagrand_record["device_ms_median"]
        )
    baseline_record["speedup_reference_for"] = "flagrand_public"


def _base_record(
    backend: str,
    distribution: str,
    n: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "generator": ("philox4x32_10" if backend == "flagrand_public" else "torch_npu_philox"),
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


def _tensor_metrics(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def _f64_precision_contract(
    output: torch.Tensor,
    distribution: str,
) -> dict[str, Any]:
    values = output.detach().cpu()
    roundtrip = values.float().double()
    same_as_f32_fraction = float((values == roundtrip).double().mean().item())
    mantissa = values.view(torch.int64) & ((1 << 52) - 1)
    low29_nonzero_fraction = float(((mantissa & ((1 << 29) - 1)) != 0).double().mean().item())
    if distribution in {"uniform_f64", "normal_f64"}:
        passed = same_as_f32_fraction == 1.0 and low29_nonzero_fraction == 0.0
        expected = "FP32 random resolution stored in binary64"
    else:
        passed = same_as_f32_fraction < 0.01 and low29_nonzero_fraction > 0.90
        expected = "FP32-resolution Normal followed by FP64 Exp"
    return {
        "status": passed,
        "expected": expected,
        "dtype": str(output.dtype),
        "element_size": output.element_size(),
        "same_as_f32_fraction": same_as_f32_fraction,
        "low29_nonzero_fraction": low29_nonzero_fraction,
    }


def _capture_error(record: dict[str, Any], exc: Exception) -> None:
    record.update(
        status="error",
        error_type=type(exc).__name__,
        error=str(exc),
        traceback_tail=traceback.format_exc().splitlines()[-20:],
    )


def _preflight_split_shape(distribution: str) -> tuple[int, int]:
    if distribution in {"raw32", "uniform_f32"}:
        return 17, 5
    if distribution == "uniform_f64":
        return 9, 3
    return 18, 6
