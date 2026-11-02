"""Host checks for automatic MTGP32 baseline preparation and provenance."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from benchmark.ppu_matrix import build


@pytest.fixture
def clean_build_environment(monkeypatch):
    monkeypatch.delenv(build.LIBRARY_ENV, raising=False)
    monkeypatch.delenv(build.BUILD_RECORD_ENV, raising=False)


def test_build_is_shared_with_children_and_record_matches_library(tmp_path, monkeypatch, clean_build_environment):
    # Execute a small stand-in builder in a child interpreter. This checks
    # subprocess paths and environment propagation without requiring a PPU SDK.
    root = tmp_path / "fresh checkout"
    (root / "tools").mkdir(parents=True)
    (root / "tools/build_ppu_mtgp32_baseline.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "for name in ('source', 'output', 'work'): p.add_argument('--' + name, type=Path)\n"
        "a = p.parse_args()\n"
        "a.output.write_bytes(b'host test library')\n"
        "a.work.mkdir(parents=True)\n"
        "(a.work / 'build.json').write_text(json.dumps({'library': str(a.output)}))\n"
    )
    monkeypatch.setattr(build, "ROOT", root)
    directory = tmp_path / "test results/native-build/mtgp32"
    library = build.prepare_mtgp32_baseline(directory)
    assert Path(library).read_bytes() == b"host test library"
    assert build.baseline_build_record()["library"] == library

    # The child must reuse the parent's library instead of rebuilding per API.
    child = subprocess.check_output(
        [
            os.sys.executable,
            "-c",
            "import json; from benchmark.ppu_matrix.build import "
            "prepare_mtgp32_baseline, baseline_build_record; "
            "print(json.dumps([prepare_mtgp32_baseline(), baseline_build_record()]))",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    assert json.loads(child) == [library, {"library": library}]
    assert build.prepare_mtgp32_baseline(tmp_path / "unused") == library
    assert not (tmp_path / "unused").exists()


def test_explicit_library_has_no_invented_build_record(tmp_path, monkeypatch, clean_build_environment):
    library = str(tmp_path / "sdk-experiment.so")
    monkeypatch.setenv(build.LIBRARY_ENV, library)
    assert build.prepare_mtgp32_baseline(tmp_path / "unused") == library
    assert build.baseline_build_record() is None
    assert not (tmp_path / "unused").exists()


def test_compiler_error_is_reported_without_selecting_another_baseline(tmp_path, monkeypatch, clean_build_environment):
    def compile_error(command, **kwargs):
        kwargs["stdout"].write("SDK compiler diagnostic\n")
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(build.subprocess, "run", compile_error)
    with pytest.raises(subprocess.CalledProcessError) as error:
        build.prepare_mtgp32_baseline(tmp_path)
    assert error.value.returncode == 2
    assert (tmp_path / "build.log").read_text() == "SDK compiler diagnostic\n"
    assert build.LIBRARY_ENV not in os.environ
    assert build.BUILD_RECORD_ENV not in os.environ
