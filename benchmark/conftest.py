"""Record measured per-operator benchmark data for flagos-optest."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("flagrand-benchmark")
    group.addoption("--level", choices=("core",), help="Run the core operator benchmarks")
    group.addoption("--base-data", type=Path, help="NVIDIA reference latency data")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    config = item.config
    if config.getoption("record") != "json" or not hasattr(item, "callspec"):
        return
    if report.when != "call" and report.passed:
        return
    operator = item.callspec.params.get("distribution")
    if operator not in config._flagrand_operators:
        return
    record = dict(item.user_properties).get("flagrand_performance", {})
    record.update(test_case=item.nodeid, result=report.outcome)
    if not report.passed:
        record["reason"] = (
            report.longrepr[2] if report.skipped and isinstance(report.longrepr, tuple) else report.longreprtext
        )
    config._flagrand_results.record("performance", operator, record)
