"""A combined pytest invocation must keep both report types."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_mixed_suites_preserve_results_and_ignore_unrelated_benchmark_parameters(tmp_path):
    root = Path(__file__).resolve().parents[2]
    for subdir in ("benchmark", "tests"):
        (tmp_path / subdir).mkdir()
    for path in ("conftest.py", "benchmark/conftest.py", "tests/conftest.py"):
        (tmp_path / path).write_text((root / path).read_text())
    (tmp_path / "tests/test_accuracy.py").write_text("def test_ok(): pass")
    (tmp_path / "benchmark/test_bench.py").write_text(
        "import pytest\n"
        '@pytest.mark.parametrize("batch", [1, 2])\n'
        "def test_unrelated(batch): pass\n"
        '@pytest.mark.parametrize("distribution", ["raw32"])\n'
        'def test_rng(distribution): pytest.skip("no device")\n'
    )
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "benchmark",
            "--quick",
            "--level",
            "core",
            "--record",
            "json",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(root)},
        check=True,
    )
    data = json.loads(output.read_text())
    assert data["accuracy"]["tests/test_accuracy.py::test_ok"]["result"] == "passed"
    assert set(data["performance"]) == {"raw32"}
    assert data["performance"]["raw32"]["result"] == "skipped"
