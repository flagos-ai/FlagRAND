"""CPU checks for family identities, unmodified means and Poisson parameter rows."""

import json
import math

from benchmark.rng_delivery.config import OPERATORS, parameter_values, shapes_for
from benchmark.summary_json import aggregate_delivery_results, build_performance_statistics
from tools.run_tests import (
    RunnerConfig,
    _performance_command,
    get_ops_from_inventory,
    get_ops_to_test,
    normalise_performance,
    parse_performance_data,
)


def test_family_aliases_select_all_25_without_repeating_poisson():
    selected = get_ops_to_test(
        get_ops_from_inventory(),
        ops_arg="xorwow,mrg32k3a,mtgp32,poisson",
        op_list_file=None,
        stages_arg="stable",
        start=None,
    )
    assert len(selected) == 25
    assert {item["id"] for item in selected} == set(OPERATORS)
    assert len(shapes_for("raw32", "matrix")) == 44
    assert len(shapes_for("normal_f64", "matrix")) == 42
    assert 1023 in shapes_for("poisson", "matrix")
    assert len(parameter_values("poisson")) == 5


def test_poisson_parameter_rows_survive_normalization(tmp_path):
    records = []
    for lam, speedup in ((0.1, 0.5), (1000, 2.0)):
        records.append(
            {
                "backend": "flagrand_public",
                "distribution": "mtgp32_poisson",
                "dtype": "int32",
                "n": 1024,
                "lambda": lam,
                "status": "pass",
                "total_ms_median": 0.1,
                "speedup_vs_mcrand": speedup,
            }
        )
    path = tmp_path / "performance_result.json"
    path.write_text(json.dumps(normalise_performance("mtgp32_poisson", records)))
    item = parse_performance_data("mtgp32_poisson", path)["data"]["int32"]
    assert set(item["details"]) == {"(1024,);lambda=0.1", "(1024,);lambda=1000"}
    assert item["speedup"] == 1.25
    assert item["geometric_mean"] == 1.0


def test_family_geometric_mean_weights_poisson_once():
    from benchmark.rng_delivery.config import APIS

    api_results = {}
    for api in APIS + ("poisson",):
        ratio = 0.5 if api == "poisson" else 2.0
        details = {
            f"point{i}": {"base": ratio, "gems": 1.0, "speedup": ratio} for i in range(5 if api == "poisson" else 1)
        }
        api_results[f"mtgp32_{api}"] = {
            "accuracy": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "details": {}, "status": "Passed"},
            "performance": {
                "status": "Passed",
                "complete_shape_set": True,
                "data": {"int32": {"details": details, "speedup": ratio, "result": "OK"}},
            },
        }
    result = build_performance_statistics(api_results)["MTGP32"]
    assert math.isclose(result["geometric_mean"], (2.0**7 * 0.5) ** (1 / 8))
    assert result["api_meets_target"]["mtgp32_poisson"] is False
    assert result["meets_target"] is False
    summary = aggregate_delivery_results(api_results)["MTGP32"]["performance"]
    assert len(summary["data"]["int32"]["details"]) == 12
    assert summary["data"]["int32"]["speedup"] == 16.5 / 12
    assert result["dtypes"]["int32"]["shape_parameter_count"] == 12
    api_results["mtgp32_raw32"]["performance"]["complete_shape_set"] = False
    assert build_performance_statistics(api_results)["MTGP32"]["geometric_mean"] is None


def test_family_summary_fields_match_philox_and_keep_all_parameter_points():
    from benchmark.rng_delivery.config import OPERATOR_GROUPS

    api_results = {}
    for api in ("raw32",) + OPERATORS:
        dtype = "fp64" if api.endswith("f64") else "fp32" if api.endswith("f32") else "int32"
        shapes = ("(1024,);lambda=0.1", "(1024,);lambda=1000") if api.endswith("poisson") else ("(1024,)",)
        api_results[api] = {
            "accuracy": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "details": {}, "status": "Passed"},
            "performance": {
                "status": "Passed",
                "complete_shape_set": True,
                "data": {
                    dtype: {
                        "result": "OK",
                        "details": {shape: {"base": 2.0, "gems": 1.0, "speedup": 2.0} for shape in shapes},
                        "speedup": 2.0,
                    }
                },
            },
        }
    summary = aggregate_delivery_results(api_results)
    philox = summary["PHILOX4_32_10"]
    philox_dtype = philox["performance"]["data"]["int32"]
    for name, operators in OPERATOR_GROUPS.items():
        family = summary[name.upper()]
        assert family.keys() == philox.keys()
        assert family["accuracy"].keys() == philox["accuracy"].keys()
        assert family["performance"].keys() == philox["performance"].keys()
        for dtype, item in family["performance"]["data"].items():
            assert item.keys() == philox_dtype.keys()
            expected = {
                f"{api}:{shape}": point
                for api in operators
                for api_dtype, data in api_results[api]["performance"]["data"].items()
                if api_dtype == dtype
                for shape, point in data["details"].items()
            }
            assert item["details"] == expected


def test_new_api_uses_fixed_matrix_and_has_no_performance_threshold_exit(tmp_path):
    config = RunnerConfig(tmp_path, "metax", "acceptance", True, 4800)
    command = _performance_command(0, "xorwow_raw32", tmp_path, config)
    assert command[2] == "benchmark.run_rng_delivery"
    assert command[command.index("--profile") + 1] == "matrix"
    assert "--strict-acceptance" not in command
