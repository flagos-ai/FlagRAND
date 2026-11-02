"""Host-only checks for automatic PPU native API discovery and result reuse."""

import json
import subprocess
from unittest.mock import Mock

import pytest

from benchmark.ppu_matrix.probe import CALL_MARKER, ENV_MARKER, RESULT_MARKER
from benchmark.rng_delivery.native import discover_native_support


def test_native_support_is_discovered_once_without_an_input_file(monkeypatch):
    output = "\n".join(
        (
            CALL_MARKER + json.dumps({"function": "acrandGetVersion", "status": 0}),
            ENV_MARKER + json.dumps({"native_library": "/sdk/libacrand.so", "native_version": 1400}),
            CALL_MARKER + json.dumps({"function": "acrandGenerate", "status": 0}),
            RESULT_MARKER + json.dumps({"status": "pass"}),
        )
    )
    run = Mock(return_value=subprocess.CompletedProcess([], 0, output, ""))
    monkeypatch.setattr("benchmark.rng_delivery.native.subprocess.run", run)
    support = {}
    first = discover_native_support(support, "xorwow", "raw32", "/sdk/libacrand.so")
    second = discover_native_support(support, "xorwow", "raw32", "/sdk/libacrand.so")
    assert first is second
    assert first["status"] == "pass"
    assert first["returncode"] == 0
    assert first["calls"][-1] == {"function": "acrandGenerate", "status": 0}
    assert support["environment"]["native_version"] == 1400
    run.assert_called_once()
    assert run.call_args.args[0][-2:] == ["--library", "/sdk/libacrand.so"]


@pytest.mark.parametrize(
    "stderr, expected",
    [
        ("[ALINPU FATAL] Unimplemented API: genDefaultU32", "unsupported"),
        ("acrandGenerate returned NotSupported", "unsupported"),
        ("Segmentation fault", "error"),
        ("TypeError: unsupported operand type", "error"),
    ],
)
def test_native_abort_retains_exit_code_and_explicit_support_status(monkeypatch, stderr, expected):
    calls = CALL_MARKER + json.dumps({"function": "acrandCreateGenerator", "status": 0})
    monkeypatch.setattr(
        "benchmark.rng_delivery.native.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], -6, calls, stderr)),
    )
    row = discover_native_support({}, "mtgp32", "raw32", "/sdk/libacrand.so")
    assert row["status"] == expected
    assert row["returncode"] == -6
    assert row["stderr"] == stderr
    assert row["calls"] == [{"function": "acrandCreateGenerator", "status": 0}]


def test_native_support_reuses_an_optional_observation(monkeypatch):
    run = Mock(side_effect=AssertionError("a supplied observation should be reused"))
    monkeypatch.setattr("benchmark.rng_delivery.native.subprocess.run", run)
    observation = {"status": "unsupported", "reason": "Unimplemented API", "returncode": -6}
    support = {"support": {"mtgp32": {"poisson": observation}}}
    assert discover_native_support(support, "mtgp32", "poisson", "/sdk/libacrand.so") is observation
    run.assert_not_called()


def test_native_timeout_is_an_error(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired(["python"], 60, output=b"partial output"))
    monkeypatch.setattr("benchmark.rng_delivery.native.subprocess.run", run)
    row = discover_native_support({}, "mrg32k3a", "normal_f64", "/sdk/libacrand.so")
    assert row["status"] == "error"
    assert row["returncode"] is None
    assert row["stdout"] == "partial output"
    assert "timed out" in row["reason"]
