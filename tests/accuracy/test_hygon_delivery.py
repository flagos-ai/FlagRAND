import json

from benchmark.hygon_matrix.config import ALL_DISTRIBUTIONS, MATRIX_SHAPE_MAP
from benchmark.nvidia_matrix.config import MATRIX_SHAPE_MAP as NVIDIA_SHAPES
from benchmark.summary_json import build_summary
from tools.run_tests import (
    RunnerConfig,
    _accuracy_test_paths,
    _performance_command,
    get_ops_from_inventory,
    supported_operator_ids,
    worker_env,
)


def test_hygon_uses_seven_api_matrix_and_unified_runner(tmp_path):
    assert MATRIX_SHAPE_MAP == NVIDIA_SHAPES
    assert supported_operator_ids("hygon") == set(ALL_DISTRIBUTIONS)
    config = RunnerConfig(tmp_path, "hygon", "acceptance", True, 60)
    command = _performance_command(0, "normal_f64", tmp_path, config)
    assert command[command.index("-m") + 1] == "benchmark.run_hygon"
    assert command[command.index("--profile") + 1] == "matrix"
    for op in get_ops_from_inventory():
        assert [p.name for p in _accuracy_test_paths(op, "hygon")] == ["test_philox_cuda.py", "test_philox_hygon.py"]


def test_hygon_children_keep_the_selected_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    env = worker_env("hygon", 1)
    assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "6"


def test_hygon_shared_summary_uses_native_baseline_and_shape_mean():
    records = []
    for distribution in ALL_DISTRIBUTIONS:
        dtype = "int32" if distribution == "raw32" else ("float64" if distribution.endswith("f64") else "float32")
        for n, base, gems in [(16, 0.04, 0.02), (4096, 0.03, 0.06)]:
            for backend, time in [("flagrand_public", gems), ("hygon_hiprand", base)]:
                records.append(
                    dict(
                        backend=backend,
                        distribution=distribution,
                        n=n,
                        dtype=dtype,
                        status="pass",
                        validation={"status": "pass"},
                        total_ms_median=time,
                    )
                )
    result = build_summary(
        {"platform": "hygon_dtk_hip", "acceptance_baseline": "hygon_hiprand", "distributions": ALL_DISTRIBUTIONS},
        records,
        environment={"test": True},
    )
    assert set(result) == {"timestamp", "env", "result"}
    assert set(result["result"]) == {"PHILOX4_32_10", "RAW_BITS", "UNIFORM", "NORMAL", "LOGNORMAL"}
    for item in result["result"].values():
        assert item["accuracy"]["status"] == item["performance"]["status"] == "Passed"
        for data in item["performance"]["data"].values():
            assert data["speedup"] == 1.25
            assert data["details"]["(4096,)"] == {"base": 0.03, "gems": 0.06, "speedup": 0.5}
    json.dumps(result, allow_nan=False)
