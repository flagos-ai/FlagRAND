"""Native RNG providers and automatic PPU API discovery for the common matrix."""

import json
import re
import subprocess
import sys
from functools import lru_cache

from benchmark.ppu_matrix.probe import CALL_MARKER, ENV_MARKER, RESULT_MARKER


@lru_cache(maxsize=None)
def native_provider(platform, engine=None):
    if platform == "thead":
        if engine == "mtgp32":
            from benchmark.ppu_matrix.mtgp32_device import (
                BASELINE_KIND,
                Mtgp32DeviceGenerator,
                mtgp32_device_library_path,
                mtgp32_device_version,
            )

            return BASELINE_KIND, Mtgp32DeviceGenerator, mtgp32_device_library_path, mtgp32_device_version
        from benchmark.ppu_matrix.baseline import AcrandGenerator, acrand_library_path, acrand_version

        return "ppu_acrand", AcrandGenerator, acrand_library_path, acrand_version
    from benchmark.muxi_matrix.mcrand import McrandGenerator, mcrand_library_path, mcrand_version

    return "mcrand", McrandGenerator, mcrand_library_path, mcrand_version


def native_support_row(support, engine, api):
    return support.get("support", {}).get(engine, {}).get(api)


def discover_native_support(support, engine, api, library_path):
    """Observe each missing engine/API once, without an external prerequisite."""
    existing = native_support_row(support, engine, api)
    if existing is not None:
        return existing
    command = [
        sys.executable,
        "-m",
        "benchmark.ppu_matrix.probe",
        "--engine",
        engine,
        "--api",
        api,
        "--library",
        library_path,
    ]
    row = {"engine": engine, "api": api, "command": command}
    try:
        child = subprocess.run(command, capture_output=True, text=True, timeout=60)
        row.update(returncode=child.returncode, stdout=child.stdout, stderr=child.stderr)
    except subprocess.TimeoutExpired as error:
        row.update(returncode=None, error="Native API observation timed out after 60 seconds")
        row["stdout"] = (
            error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout or ""
        )
        row["stderr"] = (
            error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr or ""
        )
    calls, environment, result = [], {"native_library": library_path, "native_version": None, "prefix": "acrand"}, {}
    for line in row["stdout"].splitlines():
        if line.startswith(CALL_MARKER):
            calls.append(json.loads(line[len(CALL_MARKER) :]))
        elif line.startswith(ENV_MARKER):
            environment.update(json.loads(line[len(ENV_MARKER) :]))
        elif line.startswith(RESULT_MARKER):
            result = json.loads(line[len(RESULT_MARKER) :])
    row.update(calls=calls, environment=environment)
    messages = "\n".join((row["stdout"], row["stderr"]))
    unsupported = re.search(
        r"\bUnimplemented(?:\s+API)?\b|\bNotSupported\b|\bACRAND_STATUS_NOT_SUPPORTED\b", messages, re.I
    )
    if row["returncode"] == 0 and result.get("status") == "pass":
        row["status"] = "pass"
    elif unsupported is not None:
        row.update(status="unsupported", reason=unsupported.group(0))
    else:
        row.update(
            status="error",
            reason=result.get("error") or row.get("error") or f"Native child exited {row['returncode']}",
        )
    support.setdefault("environment", {}).update(environment)
    support.setdefault("support", {}).setdefault(engine, {})[api] = row
    return row
