from __future__ import annotations

from typing import Any

import torch

from benchmark.nvidia_matrix.config import ALL_DISTRIBUTIONS, Profile
from benchmark.nvidia_matrix.curand import NvidiaCurandGenerator, nvidia_curand_library_path
from benchmark.nvidia_matrix.timing import measure
from benchmark.philox_common import (
    DTYPES,
    _base_record,
    _capture_error,
    _throughput,
    invoke_flagrand,
    make_flagrand_generator,
    validate_output,
)


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
    mismatches: dict[str, dict[str, Any]] = {}
    raw_probes = [(f"raw32_offset_{offset}", offset, 17) for offset in range(8)]
    raw_probes.append(("raw32_aligned_fast_path", 0, 32))
    for name, offset, n in raw_probes:
        expected = torch.empty(n, device="cuda", dtype=torch.int32)
        actual = torch.empty_like(expected)
        with NvidiaCurandGenerator(seed=seed, offset=offset) as generator:
            generator.generate("raw32", expected)
        invoke_flagrand("raw32", make_flagrand_generator(seed, offset=offset), actual)
        matches = bool(torch.equal(expected, actual))
        checks[name] = matches
        if not matches:
            mismatches[name] = {
                "nvidia_curand_head": expected[:8].cpu().tolist(),
                "flagrand_head": actual[:8].cpu().tolist(),
            }

    uniform_probes = []
    for distribution in ("uniform_f32", "uniform_f64"):
        uniform_probes.extend((f"{distribution}_offset_{offset}", distribution, offset, 18) for offset in range(4))
        uniform_probes.append((f"{distribution}_aligned_fast_path", distribution, 0, 64))
    for name, distribution, offset, n in uniform_probes:
        expected = torch.empty(n, device="cuda", dtype=DTYPES[distribution])
        actual = torch.empty_like(expected)
        with NvidiaCurandGenerator(seed=seed, offset=offset) as generator:
            generator.generate(distribution, expected)
        invoke_flagrand(distribution, make_flagrand_generator(seed, offset=offset), actual)
        matches = bool(torch.equal(expected, actual))
        checks[name] = matches
        if not matches:
            head_n = 8
            raw_n = head_n * (2 if distribution == "uniform_f64" else 1)
            raw = torch.empty(raw_n, device="cuda", dtype=torch.int32)
            with NvidiaCurandGenerator(seed=seed, offset=offset) as generator:
                generator.generate("raw32", raw)
            mismatches[name] = {
                "nvidia_curand_head": expected[:head_n].cpu().tolist(),
                "flagrand_head": actual[:head_n].cpu().tolist(),
                "nvidia_curand_ieee_hex": _ieee_hex_head(expected, head_n),
                "flagrand_ieee_hex": _ieee_hex_head(actual, head_n),
                "raw32_head": raw.cpu().tolist(),
            }
    torch.cuda.synchronize()
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "mismatches": mismatches,
        "curand_library": nvidia_curand_library_path(),
    }


def _ieee_hex_head(values: torch.Tensor, n: int) -> list[str]:
    if values.dtype == torch.float32:
        words = values[:n].view(torch.int32).cpu().tolist()
        return [f"0x{word & 0xFFFFFFFF:08x}" for word in words]
    words = values[:n].view(torch.int64).cpu().tolist()
    return [f"0x{word & 0xFFFFFFFFFFFFFFFF:016x}" for word in words]


def run_flagrand_case(distribution: str, n: int, seed: int, profile: Profile) -> dict[str, Any]:
    record = _base_record("flagrand_public", distribution, n)
    implementation_path = (
        "nvidia_vendor_philox_group4_f64" if distribution.endswith("_f64") else "nvidia_vendor_philox_direct_32"
    )
    record.update(seed=seed, implementation_path=implementation_path)
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
