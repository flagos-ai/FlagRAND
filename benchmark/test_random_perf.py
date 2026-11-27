"""Opt-in end-to-end performance tests for supported hardware runners."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.nvidia_matrix.config import ALL_DISTRIBUTIONS
from benchmark.rng_delivery.config import OPERATORS
from tools.run_tests import RunnerConfig, detect_platform, run_performance, supported_operator_ids

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OPERATOR_MARKS = (
    pytest.mark.raw32,
    pytest.mark.uniform_f32,
    pytest.mark.uniform_f64,
    pytest.mark.normal_f32,
    pytest.mark.normal_f64,
    pytest.mark.lognormal_f32,
    pytest.mark.lognormal_f64,
    *(getattr(pytest.mark, name) for name in OPERATORS),
)
OPERATOR_CASES = [pytest.param(mark.name, marks=mark) for mark in OPERATOR_MARKS]
pytestmark = [pytest.mark.performance]


def _performance_command(platform: str, profile: str, distribution: str) -> list[str]:
    if platform in {"muxi", "maca"}:
        platform = "metax"
    if platform == "nvidia":
        command = [sys.executable, "-m", "benchmark.run_nvidia"]
    elif platform == "metax":
        command = [
            sys.executable,
            "-m",
            "benchmark.run_muxi",
            "--skip-torch-reference",
            "--with-mcrand",
        ]
    elif platform == "ppu":
        command = [sys.executable, "-m", "benchmark.run_ppu", "--skip-torch-reference"]
    elif platform == "ascend_910c":
        command = [
            sys.executable,
            "-m",
            "benchmark.run_ascend_910c",
            "--device",
            os.environ.get("FLAGRAND_DEVICE", "0"),
        ]
    else:
        raise ValueError(f"Unsupported FLAGRAND_PERF_PLATFORM={platform!r}")

    command.extend(("--distributions", distribution))

    if profile == "acceptance":
        if platform == "nvidia":
            command.extend(("--profile", "matrix"))
        else:
            command.extend(("--profile", "full", "--acceptance-profile"))
        if platform == "ppu":
            command.append("--strict-acceptance")
        elif platform == "metax":
            command.extend(("--acceptance-baseline", "mcrand", "--strict-acceptance"))
        elif platform == "ascend_910c":
            command.extend(("--speedup-min", os.environ.get("FLAGRAND_SPEEDUP_MIN", "0.95")))
    else:
        command.extend(("--profile", profile))
    return command


def _run_legacy_performance(distribution: str) -> None:
    platform = os.environ.get("FLAGRAND_PERF_PLATFORM")
    if not platform:
        pytest.skip("set FLAGRAND_PERF_PLATFORM=nvidia, metax, ppu, or ascend_910c on a hardware runner")
    profile = os.environ.get("FLAGRAND_PERF_PROFILE", "smoke")
    command = _performance_command(platform, profile, distribution)
    LOGGER.info("Running: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=int(os.environ.get("FLAGRAND_PERF_TIMEOUT_SECONDS", "3600")),
    )
    LOGGER.info("%s", result.stdout)
    assert result.returncode == 0, result.stdout
    if platform == "ppu" and profile == "acceptance":
        assert "Acceptance status: pass" in result.stdout, result.stdout
    if platform in {"metax", "muxi", "maca"} and profile == "acceptance":
        assert "Acceptance status: pass" in result.stdout, result.stdout
    if platform == "ascend_910c" and profile == "acceptance":
        pattern = rf"^API pass\s+{re.escape(distribution)}\s"
        assert re.search(pattern, result.stdout, re.MULTILINE), result.stdout


@pytest.mark.parametrize("distribution", OPERATOR_CASES)
def test_hardware_performance_api(distribution, request, record_property, tmp_path, monkeypatch):
    if request.config.getoption("level") is None and request.config.getoption("record") is None:
        # Keep the direct, environment-driven Philox performance entry point.
        if distribution not in ALL_DISTRIBUTIONS:
            pytest.skip("family benchmarks run through --level core or tools/run_tests.py")
        _run_legacy_performance(distribution)
        return

    platform = detect_platform()
    if distribution not in supported_operator_ids(platform):
        pytest.skip(f"{platform} does not implement {distribution}")
    profile = os.environ.get("FLAGRAND_PERF_PROFILE", "smoke")
    timeout = int(os.environ.get("FLAGRAND_PERF_TIMEOUT_SECONDS", "3600"))
    # optest cancels the entire pytest process group on timeout.
    monkeypatch.setenv("FLAGRAND_MANAGED_PROCESS_GROUP", "1")
    config = RunnerConfig(tmp_path, platform, profile, True, timeout)
    result = run_performance(0, {"id": distribution}, config)
    path = tmp_path / distribution / "performance_result.json"
    record = json.loads(path.read_text(encoding="utf-8"))[distribution] if path.exists() else {}
    for detail in record.get("details", []):
        detail["dtype"] = "torch." + detail["dtype"].removeprefix("torch.")
    record_property("flagrand_performance", record)
    if result.get("status") == "Skipped":
        pytest.skip(record["reason"])
    if result.get("status") != "Passed":
        stderr = tmp_path / distribution / "performance_stderr.log"
        details = stderr.read_text(errors="replace") if stderr.exists() else ""
        pytest.fail(f"{result.get('status')}: {record.get('reason') or result.get('reason', '')}\n{details}")
    reference = request.config.getoption("base_data")
    if reference:
        from benchmark.external_baseline import apply_external_baseline

        chip = os.environ.get("FLAGOS_GPU_DEVICE")
        if not chip:
            from flagrand.runtime import device

            chip = device.torch_device_fn.get_device_name(0)
        apply_external_baseline(distribution, record["details"], reference, platform, chip)
