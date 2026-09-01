"""Opt-in end-to-end performance tests for supported hardware runners."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_CASES = [
    pytest.param("raw32", marks=pytest.mark.raw32),
    pytest.param("uniform_f32", marks=pytest.mark.uniform_f32),
    pytest.param("uniform_f64", marks=pytest.mark.uniform_f64),
    pytest.param("normal_f32", marks=pytest.mark.normal_f32),
    pytest.param("normal_f64", marks=pytest.mark.normal_f64),
    pytest.param("lognormal_f32", marks=pytest.mark.lognormal_f32),
    pytest.param("lognormal_f64", marks=pytest.mark.lognormal_f64),
]
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


@pytest.mark.parametrize("distribution", OPERATOR_CASES)
def test_hardware_performance_api(distribution: str) -> None:
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
