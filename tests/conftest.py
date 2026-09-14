"""FlagGems-compatible JSON recording for FlagRAND accuracy tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flagrand-record")
    group.addoption("--record", choices=("json",), help="Record test cases in FlagGems JSON format")
    group.addoption("--output", help="Path written by --record json")
    group.addoption("--ref", choices=("cpu",), help="Accepted for FlagGems runner compatibility")


def pytest_configure(config: pytest.Config) -> None:
    config._flagrand_records = {}  # type: ignore[attr-defined]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if report.when not in {"setup", "call"}:
        return
    records = item.config._flagrand_records  # type: ignore[attr-defined]
    if report.when == "setup" and report.passed:
        return
    params = {}
    if hasattr(item, "callspec"):
        params = {key: _json_value(value) for key, value in item.callspec.params.items()}
    record = {"result": report.outcome, "params": params}
    if not report.passed:
        record["reason"] = report.longreprtext
    records[item.nodeid] = record
    _write_records(item.config)


def pytest_sessionfinish(session: pytest.Session) -> None:
    _write_records(session.config)


def _write_records(config: pytest.Config) -> None:
    if config.getoption("record") != "json":
        return
    output = config.getoption("output")
    if not output:
        raise pytest.UsageError("--record json requires --output")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config._flagrand_records, indent=2, ensure_ascii=False),  # type: ignore[attr-defined]
        encoding="utf-8",
    )
