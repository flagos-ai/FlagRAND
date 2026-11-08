"""Build the repository-owned CoreX SDK corrected references outside timing."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmark.corex_matrix.sdk_host_extract import extract_host

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
LIBRARY_ENV = "FLAGRAND_COREX_CORRECTED_LIBRARY"
RECORD_ENV = "FLAGRAND_COREX_CORRECTED_BUILD_RECORD"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(directory, sdk_root, compiler, arch):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    host = extract_host(sdk_root / "include", directory)
    device = SOURCE / "corrected_device.cu"
    library = directory / "libflagrand_corex_corrected.so"
    host_object, device_object = directory / "host.o", directory / "device.o"
    device_options = [
        str(compiler),
        "-x",
        "ivcore",
        f"--cuda-path={sdk_root}",
        f"--offload-arch={arch}",
        "-std=c++17",
        "-O3",
        "-fPIC",
    ]
    commands = [
        ("compiler-version", [str(compiler), "--version"]),
        ("host", [str(compiler), "-x", "c++", "-std=c++17", "-O3", "-fPIC", "-c", str(host), "-o", str(host_object)]),
        ("device-driver", [*device_options, "-###", "-c", str(device), "-o", str(device_object)]),
        ("device", [*device_options, "-c", str(device), "-o", str(device_object)]),
        (
            "link",
            [
                str(compiler),
                "-shared",
                str(host_object),
                str(device_object),
                f"-L{sdk_root / 'lib64'}",
                "-lcudart",
                "-o",
                str(library),
            ],
        ),
    ]
    record = {
        "baseline_identity": "CoreX SDK corrected references: integer MRG and SDK raw + CPU Poisson alias",
        "original_sdk_host_generate_called": False,
        "sdk_root": str(sdk_root),
        "arch": arch,
        "compiler": str(compiler),
        "compiler_sha256": sha256(compiler),
        "library": str(library),
        "source_sha256": {
            str(p): sha256(p)
            for p in (
                device,
                SOURCE / "integer_mrg.cuh",
                SOURCE / "SDK-LICENSE.txt",
                SOURCE / "sdk_host_extract.py",
                Path(__file__),
                host,
            )
        },
        "headers": [{"path": str(p), "sha256": sha256(p)} for p in sorted((sdk_root / "include").glob("curand*.h"))],
        "host_extraction": json.loads((directory / "host-extraction.json").read_text()),
        "stages": [],
    }
    for name, command in commands:
        result = subprocess.run(command, text=True, capture_output=True)
        (directory / f"{name}.stdout.txt").write_text(result.stdout)
        (directory / f"{name}.stderr.txt").write_text(result.stderr)
        record["stages"].append({"stage": name, "command": command, "returncode": result.returncode})
        record["status"] = "compile_error" if result.returncode else "building"
        (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"CoreX corrected reference {name} returned {result.returncode}; see {directory}")
    record.update(status="compiled", library_sha256=sha256(library))
    (directory / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return str(library)


def prepare_corrected_baseline(build_directory=None):
    if os.environ.get(LIBRARY_ENV):
        return os.environ[LIBRARY_ENV]
    directory = Path(build_directory or ROOT / "build/corex-corrected").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "build.log"
    command = [sys.executable, "-m", "benchmark.corex_matrix.corrected_build", "--output-dir", str(directory)]
    print(f"[CoreX] Building SDK corrected references; log: {log}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    library = directory / "libflagrand_corex_corrected.so"
    os.environ[LIBRARY_ENV] = str(library)
    os.environ[RECORD_ENV] = str(directory / "build.json")
    return str(library)


def baseline_build_record():
    path = os.environ.get(RECORD_ENV)
    return json.loads(Path(path).read_text()) if path else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, default=Path(os.environ.get("COREX_HOME", "/usr/local/corex")))
    parser.add_argument("--compiler", type=Path, default=os.environ.get("FLAGRAND_COREX_COMPILER"))
    parser.add_argument("--arch", default=os.environ.get("FLAGRAND_COREX_ARCH", "ivcore11"))
    args = parser.parse_args()
    library = build(args.output_dir, args.sdk_root, args.compiler or args.sdk_root / "bin/clang++", args.arch)
    print(json.dumps({"library": library, "status": "compiled"}))


if __name__ == "__main__":
    main()
