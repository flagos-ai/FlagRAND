from __future__ import annotations

import json

from tools.run_tests import (
    get_ops_from_inventory,
    get_ops_to_test,
    normalise_performance,
    parse_accuracy_data,
    parse_performance_data,
    supported_operator_ids,
)


def test_registry_uses_flaggems_maturity_stages() -> None:
    catalog = get_ops_from_inventory()
    selected = get_ops_to_test(
        catalog,
        ops_arg=None,
        op_list_file=None,
        stages_arg="stable",
        start=None,
    )

    assert [operator["id"] for operator in selected] == [operator["id"] for operator in catalog]
    assert all(operator["stages"] == [{"stable": "0.1"}] for operator in catalog)


def test_accuracy_result_is_parsed_like_flaggems(tmp_path) -> None:
    result_file = tmp_path / "accuracy_result.json"
    result_file.write_text(
        json.dumps(
            {
                "test_a[float32]": {"result": "passed", "params": {"dtype": "float32"}},
                "test_b[float64]": {
                    "result": "skipped",
                    "params": {"dtype": "float64"},
                    "reason": "unsupported",
                },
            }
        ),
        encoding="utf-8",
    )

    result = parse_accuracy_data(result_file)

    assert result["status"] == "Passed"
    assert result["total"] == 2
    assert result["passed"] == 1
    assert result["skipped"] == 1


def test_platform_capabilities_come_from_benchmark_configs() -> None:
    assert supported_operator_ids("iluvatar") == {
        "raw32",
        "uniform_f32",
        "uniform_f64",
        "normal_f32",
        "lognormal_f32",
    }
    assert supported_operator_ids("ascend-cann900") == supported_operator_ids("thead")


def test_performance_records_are_normalised_to_flaggems_schema(tmp_path) -> None:
    raw = normalise_performance(
        "uniform_f32",
        [
            {
                "backend": "flagrand_public",
                "distribution": "uniform_f32",
                "dtype": "float32",
                "n": 65536,
                "status": "pass",
                "total_ms_median": 0.05,
                "speedup_vs_corex_curand": 1.5,
            }
        ],
    )
    result_file = tmp_path / "performance_result.json"
    result_file.write_text(json.dumps(raw), encoding="utf-8")

    result = parse_performance_data("uniform_f32", result_file)

    assert result["status"] == "Passed"
    assert result["data"]["fp32"]["speedup"] == 1.5
    assert result["data"]["fp32"]["details"]["(65536,)"] == {
        "base": 0.07500000000000001,
        "gems": 0.05,
        "speedup": 1.5,
    }


def test_missing_native_baseline_is_reported_as_skipped(tmp_path) -> None:
    raw = normalise_performance(
        "uniform_f64",
        [
            {
                "backend": "flagrand_public",
                "distribution": "uniform_f64",
                "dtype": "float64",
                "n": 65536,
                "status": "pass",
                "total_ms_median": 0.08,
            },
            {
                "backend": "corex_curand",
                "distribution": "uniform_f64",
                "dtype": "float64",
                "n": 65536,
                "status": "unsupported",
                "reason": "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
            },
        ],
    )
    result_file = tmp_path / "performance_result.json"
    result_file.write_text(json.dumps(raw), encoding="utf-8")

    result = parse_performance_data("uniform_f64", result_file)

    assert result["status"] == "Skipped"
    assert result["reason"] == "corex_curand: CURAND_STATUS_DOUBLE_PRECISION_REQUIRED (1 shapes)"
