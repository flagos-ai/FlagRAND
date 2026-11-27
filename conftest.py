"""Pytest options shared by FlagRAND's accuracy and performance suites."""

import json
from pathlib import Path

import pytest
import yaml

from tools.run_tests import get_ops_from_inventory


class ResultWriter:
    """Keep central single-suite files flat and preserve both suites in a combined invocation."""

    def __init__(self, output):
        self.output = output
        self.suites = {}

    def record(self, suite, case, result):
        if self.output:
            self.suites.setdefault(suite, {})[case] = result
            self.flush()

    def flush(self):
        if not self.output:
            return
        data = next(iter(self.suites.values())) if len(self.suites) == 1 else self.suites
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def pytest_addoption(parser):
    group = parser.getgroup("flagrand")
    group.addoption("--record", choices=("json",), help="Record FlagOS operator results")
    group.addoption("--output", type=Path, help="Destination for --record json")
    group.addoption("--collect-marks", type=Path, help="Write the collected operator markers")


def pytest_configure(config):
    config._flagrand_operators = {op["id"]: op for op in get_ops_from_inventory()}
    for name in config._flagrand_operators:
        config.addinivalue_line("markers", f"{name}: public RNG operator")
    recording = config.getoption("record") == "json"
    config._flagrand_results = ResultWriter(config.getoption("output") if recording else None)
    if recording and not config.getoption("output"):
        raise pytest.UsageError("--record json requires --output")


def pytest_collection_finish(session):
    writer = session.config._flagrand_results
    if writer.output:
        root = Path(__file__).parent
        for suite, directory in (("accuracy", root / "tests"), ("performance", root / "benchmark")):
            if any(item.path.is_relative_to(directory) for item in session.items):
                writer.suites.setdefault(suite, {})
    target = session.config.getoption("collect_marks")
    if target is None:
        return
    operators = session.config._flagrand_operators
    rows = []
    for item in session.items:
        marks = sorted({mark.name for mark in item.iter_markers() if mark.name in operators})
        if marks:
            rows.append({"nodeid": item.nodeid, "marks": marks})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(rows, sort_keys=False), encoding="utf-8")


def pytest_sessionfinish(session):
    session.config._flagrand_results.flush()
