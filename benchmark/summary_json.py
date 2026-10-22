"""FlagOS-compatible ``summary.json`` generation for FlagRAND benchmarks."""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

DTYPE_MAP = {
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "bfloat16": "bf16",
    "int16": "int16",
    "int32": "int32",
    "int8": "int8",
    "uint8": "uint8",
    "int64": "int64",
    "bool": "bool",
    "complex64": "cf64",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}

DELIVERY_OPERATORS = (
    ("PHILOX4_32_10", ("raw32",)),
    ("RAW_BITS", ("raw32",)),
    ("UNIFORM", ("uniform_f32", "uniform_f64")),
    ("NORMAL", ("normal_f32", "normal_f64")),
    ("LOGNORMAL", ("lognormal_f32", "lognormal_f64")),
)


def write_summary_json(
    run_directory: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    summary = build_summary(manifest, records)
    (run_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_summary(
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    timestamp: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flagrand_records = [record for record in records if record.get("backend") == "flagrand_public"]
    distributions = _distribution_order(manifest, flagrand_records)
    baseline = _baseline_backend(manifest)
    api_results = {
        distribution: {
            "customized": True,
            "accuracy": _accuracy_result(
                [record for record in flagrand_records if record.get("distribution") == distribution]
            ),
            "performance": _performance_result(distribution, flagrand_records, records, baseline),
        }
        for distribution in distributions
    }
    return {
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "env": environment or collect_environment(manifest),
        "result": aggregate_delivery_results(api_results),
    }


def aggregate_delivery_results(api_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {
        delivery_name: _aggregate_operator_result(
            delivery_name,
            [(api_name, api_results[api_name]) for api_name in api_names if api_name in api_results],
        )
        for delivery_name, api_names in DELIVERY_OPERATORS
    }
    for name, family in _aggregate_families(api_results).items():
        performance = family["performance"]
        result[name] = {
            "customized": family["customized"],
            "accuracy": family["accuracy"],
            "performance": {
                "status": performance["status"],
                "data": {
                    dtype: {key: item[key] for key in ("result", "details", "speedup")}
                    for dtype, item in performance["data"].items()
                },
                "test_case": performance["test_case"],
            },
        }
    return result


def build_performance_statistics(api_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep API geometric means and family statistics alongside the Philox-format summary."""
    result = {}
    for name, family in _aggregate_families(api_results).items():
        performance = family["performance"]
        result[name] = {
            key: value for key, value in performance.items() if key not in {"status", "data", "test_case"}
        }
        result[name]["dtypes"] = {
            dtype: {
                "geometric_mean": item.get("geometric_mean"),
                "shape_parameter_count": len(item["details"]),
            }
            for dtype, item in performance["data"].items()
        }
    return result


def _aggregate_families(api_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from benchmark.rng_delivery.config import APIS, ENGINES, POISSON_ENGINES

    result = {}
    groups = {engine.upper(): tuple(f"{engine}_{api}" for api in APIS + ("poisson",)) for engine in ENGINES}
    groups["POISSON"] = tuple(f"{engine}_poisson" for engine in POISSON_ENGINES)
    for name, operators in groups.items():
        selected = [(op, api_results[op]) for op in operators if op in api_results]
        if selected:
            result[name] = _aggregate_family_result(name, selected, operators)
    return result


def _aggregate_family_result(name, api_results, expected_operators):
    """Keep arithmetic speedup and add equal-API geometric means explicitly."""
    accuracy = _aggregate_accuracy([result["accuracy"] for _, result in api_results])
    data = {}
    api_means = {}
    statuses = []
    complete_shapes = True
    for operator, result in api_results:
        performance = result["performance"]
        statuses.append(performance["status"])
        complete_shapes = complete_shapes and performance.get("complete_shape_set", True)
        ratios = []
        for dtype, item in performance.get("data", {}).items():
            target = data.setdefault(dtype, {"result": "Unknown", "details": {}, "speedup": 0.0})
            for shape, point in item["details"].items():
                target["details"][f"{operator}:{shape}"] = point
                ratios.append(float(point["speedup"]))
        api_means[operator] = math.exp(sum(map(math.log, ratios)) / len(ratios)) if ratios else None
    for item in data.values():
        ratios = [float(point["speedup"]) for point in item["details"].values()]
        if ratios:
            item.update(
                result="OK",
                speedup=sum(ratios) / len(ratios),
                geometric_mean=math.exp(sum(map(math.log, ratios)) / len(ratios)),
            )
    complete = (
        set(api_means) == set(expected_operators)
        and all(value is not None for value in api_means.values())
        and complete_shapes
        and all(status == "Passed" for status in statuses)
    )
    overall = math.exp(sum(math.log(value) for value in api_means.values()) / len(api_means)) if complete else None
    status = (
        "Failed"
        if any(s in {"Failed", "Error", "Timeout"} for s in statuses)
        else (
            "Skipped"
            if any(s == "Skipped" for s in statuses)
            else "Passed" if statuses and all(s == "Passed" for s in statuses) else "NotFound"
        )
    )
    return {
        "customized": True,
        "accuracy": accuracy,
        "performance": {
            "status": status,
            "data": data,
            "test_case": name,
            "aggregation": "speedup: arithmetic over recorded points; geometric_mean: equal weight per API",
            "api_geometric_means": api_means,
            "geometric_mean": overall,
            "complete_api_family": complete,
            "expected_apis": list(expected_operators),
            "observed_apis": list(api_means),
            "target_geometric_mean": 0.95,
            "target_scope": "each API geometric mean; Poisson includes all configured lambda-shape pairs",
            "api_meets_target": {op: value >= 0.95 if value is not None else None for op, value in api_means.items()},
            "meets_target": all(value >= 0.95 for value in api_means.values()) if complete else None,
        },
    }


def _aggregate_operator_result(
    delivery_name: str,
    api_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "customized": True,
        "accuracy": _aggregate_accuracy([result["accuracy"] for _, result in api_results]),
        "performance": _aggregate_performance(delivery_name, api_results),
    }


def _aggregate_accuracy(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(int(result.get("passed", 0)) for result in results)
    skipped = sum(int(result.get("skipped", 0)) for result in results)
    failed = sum(int(result.get("failed", 0)) for result in results)
    details: dict[str, dict[str, list[str]]] = {}
    for result in results:
        for category, reasons in result.get("details", {}).items():
            target = details.setdefault(str(category), {})
            for reason, cases in reasons.items():
                target.setdefault(str(reason), []).extend(str(case) for case in cases)

    statuses = {str(result.get("status", "NotFound")) for result in results}
    if failed or statuses & {"Failed", "Error", "Timeout"}:
        status = "Failed"
    elif passed:
        status = "Passed"
    elif skipped:
        status = "Skipped"
    else:
        status = "NotFound"
    return {
        "total": sum(int(result.get("total", 0)) for result in results),
        "skipped": skipped,
        "failed": failed,
        "passed": passed,
        "details": details,
        "status": status,
    }


def _aggregate_performance(
    delivery_name: str,
    api_results: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    data: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    reasons: list[str] = []
    for api_name, operator_result in api_results:
        performance = operator_result["performance"]
        status = str(performance.get("status", "NotFound"))
        statuses.append(status)
        if status != "Passed" and performance.get("reason"):
            reasons.append(f"{api_name}: {performance['reason']}")
        for dtype, dtype_result in performance.get("data", {}).items():
            target = data.setdefault(str(dtype), {"result": "Unknown", "details": {}, "speedup": 0.0})
            target["details"].update(dtype_result.get("details", {}))

    for dtype_result in data.values():
        details = dtype_result["details"]
        if details:
            dtype_result["result"] = "OK"
            dtype_result["speedup"] = sum(float(item["speedup"]) for item in details.values()) / len(details)

    if any(status in {"Failed", "Error", "Timeout"} for status in statuses):
        status = "Failed"
    elif "Passed" in statuses:
        status = "Passed"
    elif "Skipped" in statuses:
        status = "Skipped"
    else:
        status = "NotFound"
    result: dict[str, Any] = {
        "status": status,
        "data": data,
        "test_case": delivery_name,
    }
    if status != "Passed" and reasons:
        result["reason"] = "; ".join(reasons)
    return result


def collect_environment(manifest: dict[str, Any]) -> dict[str, Any]:
    import torch
    import triton

    from flagrand import __version__ as flagrand_version
    from flagrand.runtime import device as runtime_device

    platform_name = str(manifest["platform"])
    source = manifest.get("environment", {})
    if platform_name == "huawei_ascend_910c":
        versions = source["versions"]
        torch_version = versions["torch"]
        device_name = source["device_name"]
        device_count = torch.npu.device_count()
        cuda_available = torch.cuda.is_available()
    else:
        torch_version = source["torch"]
        device_info = source.get("device", "N/A")
        device_name = device_info.get("name", "N/A") if isinstance(device_info, dict) else device_info
        device_count = int(source.get("device_count", torch.cuda.device_count()))
        cuda_available = torch.cuda.is_available()

    try:
        flagtree_version: str | None = importlib.metadata.version("flagtree")
    except importlib.metadata.PackageNotFoundError:
        flagtree_version = None

    os_release = platform.freedesktop_os_release()
    return {
        "architecture": platform.machine(),
        "os_name": os_release["ID"],
        "os_release": os_release["VERSION_ID"],
        "python": platform.python_version(),
        "torch": {
            "version": torch_version,
            "cuda_available": cuda_available,
            "device_name": device_name,
            "device_count": device_count,
        },
        "flagtree": flagtree_version,
        "triton": {
            "version": triton.__version__,
            "has_config": hasattr(triton, "Config"),
        },
        "flagrand": {
            "version": flagrand_version,
            "vendor": runtime_device.vendor_name,
            "device": runtime_device.name,
        },
    }


def _distribution_order(
    manifest: dict[str, Any],
    flagrand_records: list[dict[str, Any]],
) -> list[str]:
    configured = list(manifest.get("distributions", ()))
    observed = [str(record["distribution"]) for record in flagrand_records]
    return list(dict.fromkeys(configured + observed))


def _baseline_backend(manifest: dict[str, Any]) -> str:
    platform_name = str(manifest["platform"])
    if platform_name == "nvidia_h20_cuda":
        return "nvidia_curand"
    if platform_name == "iluvatar_bi_v150_corex":
        return "corex_curand"
    if platform_name == "huawei_ascend_910c":
        return "cann_aclnn"
    return str(manifest.get("acceptance_baseline", "ppu_acrand"))


def _accuracy_result(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed: list[str] = []
    skipped: dict[str, list[str]] = {}
    failed: dict[str, list[str]] = {}
    for record in records:
        case = _case_name(record)
        status = record.get("status")
        if status == "pass" and record.get("validation", {}).get("status") == "pass":
            passed.append(case)
        elif status == "unsupported":
            skipped.setdefault(str(record.get("reason", "Unknown")), []).append(case)
        else:
            failed.setdefault(_failure_reason(record), []).append(case)

    details: dict[str, Any] = {}
    if failed:
        details["failed"] = failed
    elif skipped:
        details["skipped"] = skipped
    if failed:
        status = "Failed"
    elif passed:
        status = "Passed"
    elif skipped:
        status = "Skipped"
    else:
        status = "NotFound"
    return {
        "total": len(records),
        "skipped": sum(len(cases) for cases in skipped.values()),
        "failed": sum(len(cases) for cases in failed.values()),
        "passed": len(passed),
        "details": details,
        "status": status,
    }


def _performance_result(
    distribution: str,
    flagrand_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    baseline_backend: str,
) -> dict[str, Any]:
    implementation = [record for record in flagrand_records if record.get("distribution") == distribution]
    if not implementation:
        return {"status": "NotFound", "data": {}, "test_case": distribution}

    baselines = {
        int(record["n"]): record
        for record in all_records
        if record.get("backend") == baseline_backend and record.get("distribution") == distribution
    }
    data: dict[str, dict[str, Any]] = {}
    missing_baselines: dict[str, list[str]] = {}
    failed_cases: list[str] = []
    for record in implementation:
        dtype = _dtype_name(str(record["dtype"]))
        dtype_result = data.setdefault(dtype, {"result": "Unknown", "details": {}, "speedup": 0})
        if record.get("status") != "pass":
            failed_cases.append(_case_name(record))
            continue
        baseline = baselines.get(int(record["n"]))
        if baseline is None or baseline.get("status") != "pass":
            reason = (
                "baseline not run"
                if baseline is None
                else str(baseline.get("reason") or baseline.get("error") or baseline["status"])
            )
            missing_baselines.setdefault(reason, []).append(_case_name(record))
            continue
        gems = float(record["total_ms_median"])
        base = float(baseline["total_ms_median"])
        shape = f"({int(record['n'])},)"
        dtype_result["details"][shape] = {
            "base": base,
            "gems": gems,
            "speedup": base / gems,
        }

    for dtype_result in data.values():
        details = dtype_result["details"]
        if details:
            dtype_result["result"] = "OK"
            dtype_result["speedup"] = sum(item["speedup"] for item in details.values()) / len(details)

    result: dict[str, Any] = {
        "status": "Passed",
        "data": data,
        "test_case": distribution,
    }
    if failed_cases:
        result["status"] = "Failed"
        result["reason"] = "; ".join(failed_cases)
    elif missing_baselines:
        result["status"] = "Skipped"
        reasons = "; ".join(f"{reason} ({len(cases)} shapes)" for reason, cases in missing_baselines.items())
        result["reason"] = f"{baseline_backend}: {reasons}"
    return result


def _dtype_name(dtype: str) -> str:
    normalized = dtype.removeprefix("torch.")
    return DTYPE_MAP.get(normalized, f"torch.{normalized}")


def _case_name(record: dict[str, Any]) -> str:
    return f"{record['distribution']}:({int(record['n'])},)"


def _failure_reason(record: dict[str, Any]) -> str:
    if record.get("error"):
        return str(record["error"])
    validation = record.get("validation", {})
    failed_checks = [name for name, passed in validation.get("checks", {}).items() if passed is False]
    if failed_checks:
        return "failed checks: " + ",".join(failed_checks)
    return str(record.get("status", "Unknown"))
