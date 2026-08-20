from __future__ import annotations

from benchmark.summary_json import build_summary

DELIVERY_NAMES = ["PHILOX4_32_10", "RAW_BITS", "UNIFORM", "NORMAL", "LOGNORMAL"]

ENVIRONMENT = {
    "architecture": "x86_64",
    "os_name": "ubuntu",
    "os_release": "22.04",
    "python": "3.10.18",
    "torch": {
        "version": "2.7.1+corex.4.4.0",
        "cuda_available": True,
        "device_name": "Iluvatar BI-V150",
        "device_count": 4,
    },
    "flagtree": None,
    "triton": {"version": "3.1.0+corex.4.4.0", "has_config": True},
    "flagrand": {"version": "0.1.0", "vendor": "iluvatar", "device": "cuda"},
}


def test_summary_matches_flagos_schema_and_latency_semantics() -> None:
    records = [
        _record("flagrand_public", "uniform_f32", "float32", 4096, 0.20),
        _record("corex_curand", "uniform_f32", "float32", 4096, 0.05),
    ]
    summary = build_summary(
        {"platform": "iluvatar_bi_v150_corex", "distributions": ("uniform_f32",)},
        records,
        timestamp="2026-08-18 12:34:56",
        environment=ENVIRONMENT,
    )

    assert set(summary) == {"timestamp", "env", "result"}
    assert summary["timestamp"] == "2026-08-18 12:34:56"
    assert summary["env"] == ENVIRONMENT
    assert list(summary["result"]) == DELIVERY_NAMES
    operator = summary["result"]["UNIFORM"]
    assert set(operator) == {"customized", "accuracy", "performance"}
    assert operator["accuracy"] == {
        "total": 1,
        "skipped": 0,
        "failed": 0,
        "passed": 1,
        "details": {},
        "status": "Passed",
    }
    assert operator["performance"] == {
        "status": "Passed",
        "data": {
            "fp32": {
                "result": "OK",
                "details": {"(4096,)": {"base": 0.05, "gems": 0.20, "speedup": 0.25}},
                "speedup": 0.25,
            }
        },
        "test_case": "UNIFORM",
    }


def test_summary_marks_unsupported_baseline_without_fake_measurements() -> None:
    records = [
        _record("flagrand_public", "uniform_f64", "float64", 65536, 0.18),
        {
            "backend": "corex_curand",
            "distribution": "uniform_f64",
            "dtype": "float64",
            "n": 65536,
            "status": "unsupported",
            "reason": "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
        },
    ]
    summary = build_summary(
        {"platform": "iluvatar_bi_v150_corex", "distributions": ("uniform_f64",)},
        records,
        timestamp="2026-08-18 12:34:56",
        environment=ENVIRONMENT,
    )

    operator = summary["result"]["UNIFORM"]
    assert operator["accuracy"]["status"] == "Passed"
    assert operator["performance"]["status"] == "Skipped"
    assert operator["performance"]["data"] == {"fp64": {"result": "Unknown", "details": {}, "speedup": 0}}
    assert "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" in operator["performance"]["reason"]


def test_summary_groups_accuracy_failures_by_reason() -> None:
    record = _record("flagrand_public", "normal_f32", "float32", 8192, 0.3)
    record["status"] = "validation_fail"
    record["validation"] = {
        "status": "fail",
        "checks": {"finite": True, "mean": False, "std": False},
    }
    summary = build_summary(
        {"platform": "iluvatar_bi_v150_corex", "distributions": ("normal_f32",)},
        [record],
        timestamp="2026-08-18 12:34:56",
        environment=ENVIRONMENT,
    )

    accuracy = summary["result"]["NORMAL"]["accuracy"]
    assert accuracy["status"] == "Failed"
    assert accuracy["failed"] == 1
    assert accuracy["details"] == {"failed": {"failed checks: mean,std": ["normal_f32:(8192,)"]}}


def test_summary_selects_the_platform_acceptance_baseline() -> None:
    platforms = (
        ({"platform": "huawei_ascend_910c"}, "cann_aclnn"),
        ({"platform": "PPU", "acceptance_baseline": "ppu_acrand"}, "ppu_acrand"),
    )
    for manifest, baseline in platforms:
        manifest["distributions"] = ("raw32",)
        records = [
            _record("flagrand_public", "raw32", "int32", 1024, 0.25),
            _record(baseline, "raw32", "int32", 1024, 0.5),
        ]
        summary = build_summary(
            manifest,
            records,
            timestamp="2026-08-18 12:34:56",
            environment=ENVIRONMENT,
        )

        for delivery_name in ("PHILOX4_32_10", "RAW_BITS"):
            performance = summary["result"][delivery_name]["performance"]
            assert performance["status"] == "Passed"
            assert performance["data"]["int32"]["details"]["(1024,)"] == {
                "base": 0.5,
                "gems": 0.25,
                "speedup": 2.0,
            }


def test_summary_aggregates_distribution_dtypes_under_delivery_name() -> None:
    records = [
        _record("flagrand_public", "uniform_f32", "float32", 4096, 0.20),
        _record("corex_curand", "uniform_f32", "float32", 4096, 0.10),
        _record("flagrand_public", "uniform_f64", "float64", 65536, 0.40),
        _record("corex_curand", "uniform_f64", "float64", 65536, 0.20),
    ]
    summary = build_summary(
        {
            "platform": "iluvatar_bi_v150_corex",
            "distributions": ("uniform_f32", "uniform_f64"),
        },
        records,
        timestamp="2026-08-18 12:34:56",
        environment=ENVIRONMENT,
    )

    uniform = summary["result"]["UNIFORM"]
    assert uniform["accuracy"]["total"] == 2
    assert uniform["accuracy"]["passed"] == 2
    assert uniform["performance"]["status"] == "Passed"
    assert set(uniform["performance"]["data"]) == {"fp32", "fp64"}
    assert uniform["performance"]["data"]["fp32"]["speedup"] == 0.5
    assert uniform["performance"]["data"]["fp64"]["speedup"] == 0.5


def _record(backend: str, distribution: str, dtype: str, n: int, total_ms: float) -> dict[str, object]:
    return {
        "backend": backend,
        "distribution": distribution,
        "dtype": dtype,
        "n": n,
        "status": "pass",
        "total_ms_median": total_ms,
        "validation": {"status": "pass", "checks": {"finite": True}},
    }
