from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from tools.run_tests import (
    PLATFORM_ALIASES,
    RunnerConfig,
    _accuracy_test_paths,
    _performance_command,
    get_ops_from_inventory,
    get_ops_to_test,
    normalise_performance,
    parse_accuracy_data,
    parse_performance_data,
    run_performance,
    run_accuracy,
    supported_operator_ids,
    worker_env,
)


def test_accuracy_retains_completed_case_when_next_case_aborts(tmp_path, monkeypatch) -> None:
    import tools.run_tests as runner

    plugin = Path(__file__).resolve().parents[1] / 'conftest.py'
    (tmp_path / 'conftest.py').write_text(plugin.read_text())
    test_file = tmp_path / 'test_abrupt.py'
    test_file.write_text('''import os
import pytest
@pytest.mark.raw32
def test_00_pass():
    assert 1 == 1
@pytest.mark.raw32
def test_01_abort():
    os._exit(86)
''')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, '_accuracy_test_paths', lambda *_: [test_file])
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    config = RunnerConfig(output_dir=tmp_path/'results', platform='iluvatar', profile='acceptance',
                          dump_output=True, timeout=30)
    result = run_accuracy(0, {'id':'raw32'}, config)
    assert result['status'] == 'Error'
    assert result['exit_code'] == 86
    assert result['passed'] == 1
    assert result['total'] == 1
    records = json.loads((config.output_dir/result['data_file']).read_text())
    assert next(iter(records.values()))['result'] == 'passed'


def test_acceptance_profile_is_translated_for_each_backend(tmp_path) -> None:
    cases = {
        "nvidia": ("matrix", set()),
        "hygon": ("matrix", set()),
        "metax": (
            "full",
            {
                "--skip-torch-reference",
                "--with-mcrand",
                "--acceptance-profile",
                "--acceptance-baseline",
                "mcrand",
                "--strict-acceptance",
            },
        ),
        "thead": ("full", {"--skip-torch-reference", "--acceptance-profile"}),
        "ascend-cann900": (
            "full",
            {"--allow-non-cann9", "--allow-unverified-910c", "--acceptance-profile"},
        ),
        "iluvatar": ("matrix", set()),
    }

    for platform, (expected_profile, expected_flags) in cases.items():
        config = RunnerConfig(
            output_dir=tmp_path,
            platform=platform,
            profile="acceptance",
            dump_output=False,
            timeout=60,
        )
        command = _performance_command(0, "uniform_f32", tmp_path / platform, config)

        profile_index = command.index("--profile")
        assert command[profile_index + 1] == expected_profile
        assert expected_flags.issubset(command)


def test_ascend_worker_maps_masked_physical_device_to_logical_zero(tmp_path) -> None:
    config = RunnerConfig(
        output_dir=tmp_path,
        platform="ascend-cann900",
        profile="smoke",
        dump_output=False,
        timeout=60,
    )

    env = worker_env(config.platform, 6)
    command = _performance_command(6, "normal_f32", tmp_path, config)

    assert env["ASCEND_RT_VISIBLE_DEVICES"] == "6"
    assert env["NPU_VISIBLE_DEVICES"] == "6"
    assert command[command.index("--device") + 1] == "0"


def test_nvidia_accuracy_combines_shared_cuda_and_native_curand_tests() -> None:
    operator = next(item for item in get_ops_from_inventory() if item["id"] == "uniform_f64")

    paths = [path.name for path in _accuracy_test_paths(operator, "nvidia")]

    assert paths == ["test_philox_cuda.py", "test_philox_nvidia.py"]


def test_iluvatar_child_keeps_the_agents_physical_device_mask(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setenv("ILUVATAR_VISIBLE_DEVICES", "0")
    for logical, physical in [(0, "1"), (1, "3")]:
        env = worker_env("iluvatar", logical)
        assert env["CUDA_VISIBLE_DEVICES"] == env["ILUVATAR_VISIBLE_DEVICES"] == physical


def test_managed_phase_remains_in_the_node_workers_process_group(tmp_path) -> None:
    from tools.run_tests import run_cmd
    result = tmp_path / "pgrp.txt"
    config = RunnerConfig(tmp_path, "iluvatar", "smoke", False, 10)
    env = dict(os.environ, FLAGRAND_MANAGED_PROCESS_GROUP="1")
    code = run_cmd("raw32", "performance", [sys.executable, "-c",
        "import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpgrp()))", str(result)], env=env, config=config)
    assert code == 0
    assert int(result.read_text()) == os.getpgrp()


def test_metax_accuracy_combines_shared_cuda_and_platform_dispatch_tests() -> None:
    operator = next(item for item in get_ops_from_inventory() if item["id"] == "normal_f32")

    paths = [path.name for path in _accuracy_test_paths(operator, "metax")]

    assert paths == ["test_philox_cuda.py", "test_philox_metax.py"]


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
    assert supported_operator_ids("metax") == supported_operator_ids("thead")
    assert supported_operator_ids("nvidia") == supported_operator_ids("thead")


def test_metax_platform_aliases_use_the_canonical_runner_name() -> None:
    assert {PLATFORM_ALIASES[name] for name in ("metax", "muxi", "maca")} == {"metax"}


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


def test_nvidia_performance_records_use_native_curand_speedup(tmp_path) -> None:
    raw = normalise_performance(
        "normal_f64",
        [
            {
                "backend": "flagrand_public",
                "distribution": "normal_f64",
                "dtype": "float64",
                "n": 65536,
                "status": "pass",
                "total_ms_median": 0.08,
                "speedup_vs_nvidia_curand": 1.25,
            }
        ],
    )
    result_file = tmp_path / "nvidia_performance_result.json"
    result_file.write_text(json.dumps(raw), encoding="utf-8")

    result = parse_performance_data("normal_f64", result_file)

    assert result["status"] == "Passed"
    assert result["data"]["fp64"]["speedup"] == 1.25


def test_metax_performance_records_use_native_mcrand_speedup(tmp_path) -> None:
    raw = normalise_performance(
        "normal_f32",
        [
            {
                "backend": "flagrand_public",
                "distribution": "normal_f32",
                "dtype": "float32",
                "n": 65536,
                "status": "pass",
                "total_ms_median": 0.08,
                "speedup_vs_mcrand": 1.2,
            }
        ],
    )
    result_file = tmp_path / "metax_performance_result.json"
    result_file.write_text(json.dumps(raw), encoding="utf-8")

    result = parse_performance_data("normal_f32", result_file)

    assert result["status"] == "Passed"
    assert result["data"]["fp32"]["speedup"] == 1.2


def test_performance_preserves_raw_benchmark_artifacts(tmp_path, monkeypatch) -> None:
    config = RunnerConfig(
        output_dir=tmp_path,
        platform="nvidia",
        profile="acceptance",
        dump_output=False,
        timeout=60,
    )

    def fake_run_cmd(*args, **kwargs) -> int:
        run_dir = tmp_path / "raw32" / "performance_raw" / "matrix_run"
        run_dir.mkdir(parents=True)
        record = {
            "backend": "flagrand_public",
            "distribution": "raw32",
            "dtype": "int32",
            "n": 4096,
            "status": "pass",
            "total_ms_median": 0.01,
            "speedup_vs_nvidia_curand": 1.1,
        }
        (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        for name in ("manifest.json", "records.csv", "summary.json", "summary.md"):
            (run_dir / name).write_text("artifact", encoding="utf-8")
        return 0

    monkeypatch.setattr("tools.run_tests.run_cmd", fake_run_cmd)

    result = run_performance(0, {"id": "raw32"}, config)

    assert result["status"] == "Passed"
    run_dir = tmp_path / "raw32" / "performance_raw" / "matrix_run"
    assert {path.name for path in run_dir.iterdir()} == {
        "manifest.json",
        "records.jsonl",
        "records.csv",
        "summary.json",
        "summary.md",
    }


def test_nonzero_performance_exit_is_failed_when_records_are_empty(tmp_path, monkeypatch) -> None:
    config = RunnerConfig(
        output_dir=tmp_path,
        platform="ascend-cann900",
        profile="smoke",
        dump_output=False,
        timeout=60,
    )

    def fake_run_cmd(*args, **kwargs) -> int:
        run_dir = tmp_path / "normal_f32" / "performance_raw" / "failed_run"
        run_dir.mkdir(parents=True)
        (run_dir / "records.jsonl").write_text("", encoding="utf-8")
        return 3

    monkeypatch.setattr("tools.run_tests.run_cmd", fake_run_cmd)

    result = run_performance(6, {"id": "normal_f32"}, config)

    assert result["status"] == "Failed"
    assert result["exit_code"] == 3
    assert result["reason"] == "benchmark command exited with 3"


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
    assert result["data"] == {"fp64": {"result": "Unknown", "details": {}, "speedup": 0.0}}
