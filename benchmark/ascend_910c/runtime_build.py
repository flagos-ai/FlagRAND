"""Compile the optional CANN Runtime C baseline before timing."""

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def prepare_runtime_baseline(build_directory=None, cann_root=None):
    directory = Path(build_directory or ROOT / "build/ascend-runtime").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cann = Path(
        cann_root
        or os.environ.get("FLAGRAND_ASCEND_CANN_ROOT")
        or os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    source = Path(__file__).with_name("runtime_rng.cpp")
    library = directory / "libflagrand_cann_runtime_rng.so"
    command = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-shared",
        str(source),
        f"-I{cann / 'include'}",
        f"-I{cann / 'include/aclnnop'}",
        f"-L{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        "-lopapi",
        "-lascendcl",
        "-o",
        str(library),
    ]
    log = directory / "build.log"
    print(f"[910C] Building CANN Runtime C baseline; log: {log}", flush=True)
    with log.open("w") as output:
        subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True)
    (directory / "build.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "library": str(library),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                "command": command,
            },
            indent=2,
        )
        + "\n"
    )
    return str(library)
