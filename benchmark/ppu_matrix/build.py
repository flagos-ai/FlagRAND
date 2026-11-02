"""Build the repository's MTGP32 SDK baseline before benchmark timing."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIBRARY_ENV = "FLAGRAND_PPU_MTGP32_DEVICE_LIBRARY"
BUILD_RECORD_ENV = "FLAGRAND_PPU_MTGP32_BUILD_RECORD"


def prepare_mtgp32_baseline(build_directory=None):
    """Build once per runner, then share the library with its child processes.

    An explicit library is still supported for SDK experiments. The normal
    test entry point builds from source in its own result directory, so a new
    checkout never needs a library or support record from a previous machine.
    """
    if os.environ.get(LIBRARY_ENV):
        return os.environ[LIBRARY_ENV]

    directory = Path(build_directory or ROOT / "build/ppu-mtgp32").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libflagrand_mtgp32_device.so"
    work = directory / "compiler"
    command = [
        sys.executable,
        str(ROOT / "tools/build_ppu_mtgp32_baseline.py"),
        "--source",
        str(ROOT / "benchmark/ppu_matrix/mtgp32_device.hg"),
        "--output",
        str(library),
        "--work",
        str(work),
    ]
    log = directory / "build.log"
    print(f"[PPU] Building MTGP32 SDK baseline; log: {log}", flush=True)
    with log.open("w") as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    os.environ[LIBRARY_ENV] = str(library)
    os.environ[BUILD_RECORD_ENV] = str(work / "build.json")
    return str(library)


def baseline_build_record():
    """Return the build record belonging to the selected library, when supplied."""
    import json

    record_path = os.environ.get(BUILD_RECORD_ENV)
    if not record_path:
        return None
    record = json.loads(Path(record_path).read_text())
    if Path(record["library"]).resolve() != Path(os.environ[LIBRARY_ENV]).resolve():
        return None
    return record
