"""FlagGems-compatible JSON recording for FlagRAND accuracy tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tools.run_tests import ROOT, _accuracy_test_paths, detect_platform, supported_operator_ids


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flagrand-record")
    group.addoption("--quick", action="store_true", help="Run one parameter set per selected accuracy test")
    group.addoption("--ref", choices=("cpu",), help="Accepted for FlagGems runner compatibility")


def pytest_sessionstart(session):
    config = session.config
    config._flagrand_accuracy_paths = None
    config._flagrand_skip_reason = None
    operator = config._flagrand_operators.get(config.option.markexpr)
    if not (
        operator and config.getoption("record") and any(Path(arg).resolve() == ROOT / "tests" for arg in config.args)
    ):
        return
    platform = detect_platform()
    config._flagrand_accuracy_paths = set(_accuracy_test_paths(operator, platform))
    if operator["id"] not in supported_operator_ids(platform):
        config._flagrand_skip_reason = f"{platform} does not implement {operator['id']}"


def pytest_ignore_collect(collection_path, config):
    paths = config._flagrand_accuracy_paths
    if paths is not None and collection_path.name.startswith("test_") and collection_path.suffix == ".py":
        return bool(config._flagrand_skip_reason) or collection_path not in paths
    return None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("quick"):
        return
    seen = set()
    selected, deselected = [], []
    for item in items:
        if not item.path.is_relative_to(Path(__file__).parent):
            selected.append(item)
            continue
        key = (item.parent.nodeid, item.originalname)
        if key in seen:
            deselected.append(item)
        else:
            seen.add(key)
            selected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    if report.failed:
        collector.config._flagrand_results.record(
            "accuracy", report.nodeid, {"result": "failed", "reason": report.longreprtext}
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]):
    outcome = yield
    report = outcome.get_result()
    if item.config.getoption("record") != "json" or (report.when != "call" and report.passed):
        return
    params = {}
    if hasattr(item, "callspec"):
        params = {key: _json_value(value) for key, value in item.callspec.params.items()}
    record = {"result": report.outcome, "params": params}
    if item.user_properties:
        record["observations"] = {key: _json_value(value) for key, value in item.user_properties}
    if not report.passed:
        record["reason"] = report.longreprtext
    item.config._flagrand_results.record("accuracy", item.nodeid, record)


def pytest_sessionfinish(session: pytest.Session) -> None:
    config = session.config
    if config._flagrand_skip_reason and session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        config._flagrand_results.record(
            "accuracy", config.option.markexpr, {"result": "skipped", "reason": config._flagrand_skip_reason}
        )
        session.exitstatus = pytest.ExitCode.OK
