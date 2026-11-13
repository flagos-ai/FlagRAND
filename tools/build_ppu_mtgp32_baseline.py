#!/usr/bin/env python3
"""Build the SDK device baseline as a shared library with HGGC's own stages.

The installed host splitter needs the same GCC compatibility define that was
validated in the 2026-10-24 r5 experiment. Device cc1 flags are left unchanged.
HGGC does not accept -shared. Its full link plan supplies the device-link
registration object and SDK libraries; only the final host link adds -shared.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def parse_commands(text):
    commands = []
    environment = {}
    for line in text.splitlines():
        if line.startswith("#$ "):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", line[3:]):
                name, value = shlex.split(line[3:], posix=True)[0].split("=", 1)
                environment[name] = value
                continue
            tokens = shlex.split(line[3:], posix=False)
            commands.append([token[1:-1] if token[0] in "\"'" and token[-1] == token[0] else token for token in tokens])
    return commands, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--compiler", default=os.environ.get("FLAGRAND_PPU_HGCC", "hgcc"))
    parser.add_argument("--arch", default=os.environ.get("FLAGRAND_PPU_ARCH", "ppu001"))
    args = parser.parse_args()
    source, library, work = args.source.resolve(), args.output.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    library.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.compiler,
        "--dryrun",
        "-O3",
        "-std=c++14",
        f"-arch={args.arch}",
        "-keep",
        "-Xcompiler",
        "-std=c++14",
        "-Xcompiler",
        "-fPIC",
        str(source),
        "-o",
        str(library),
    ]
    dryrun = subprocess.run(command, cwd=work, capture_output=True, text=True)
    (work / "dryrun.stdout").write_text(dryrun.stdout)
    (work / "dryrun.stderr").write_text(dryrun.stderr)
    record = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library": str(library),
        "command": command,
        "dryrun_returncode": dryrun.returncode,
        "device_cc1_modified": False,
        "shared_link": "HGGC full device-link plan; -shared added only to the final host compiler command",
        "stages": [],
    }

    def save():
        (work / "build.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    if dryrun.returncode:
        sys.stderr.write(dryrun.stderr)
        return dryrun.returncode
    commands, dryrun_environment = parse_commands(dryrun.stdout + "\n" + dryrun.stderr)
    record["environment_overrides"] = dryrun_environment
    if not commands:
        raise RuntimeError("HGGC dryrun returned no stage commands")
    record["planned_stages"] = len(commands)
    for index, original in enumerate(commands):
        argv = list(original)
        if "-cc1" in argv and "-split-host-code" in argv:
            argv.insert(argv.index("-cc1") + 1, "-D__HGGC_GCC_X86IMM_BUILTINS_H__")
        if index == len(commands) - 1:
            # The installed HGGC dry-run ends with g++ linking the host object,
            # device registration object, and -lhggcrt1. Keep that entire argv;
            # -Xcompiler has already supplied -fPIC to both host object stages.
            argv.insert(1, "-shared")
        result = subprocess.run(argv, cwd=work, capture_output=True, text=True,
                                env={**os.environ, **dryrun_environment})
        prefix = f"stage-{index:02d}"
        (work / (prefix + ".stdout")).write_text(result.stdout)
        (work / (prefix + ".stderr")).write_text(result.stderr)
        record["stages"].append(
            {
                "original_argv": original,
                "executed_argv": argv,
                "returncode": result.returncode,
                "stdout": prefix + ".stdout",
                "stderr": prefix + ".stderr",
            }
        )
        save()
        print(f"stage {index + 1}/{len(commands)}: {Path(argv[0]).name} rc={result.returncode}", flush=True)
        if result.returncode:
            sys.stderr.write(result.stderr)
            return result.returncode
    record["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    # The -keep preprocessed source records the include paths actually consumed
    # by this build, including resolved SDK header variants.
    names = {
        "hggc_runtime.h",
        "hggc_library_types.h",
        "hgrand.h",
        "hgrand_kernel.h",
        "hgrand_mtgp32.h",
        "hgrand_mtgp32_kernel.h",
        "hgrand_mtgp32_host.h",
        "hgrand_mtgp32dc_p_11213.h",
        "hgrand_uniform.h",
        "hgrand_normal.h",
        "hgrand_normal_static.h",
        "hgrand_lognormal.h",
        "hgrand_poisson.h",
    }
    seen = {}
    for preprocessed in sorted(work.rglob("*")):
        if not preprocessed.is_file() or preprocessed.suffix not in (".ii", ".i", ".hggci", ".cpp"):
            continue
        for line in preprocessed.read_text(errors="replace").splitlines():
            match = re.match(r'^#\s+(?:line\s+)?\d+\s+"([^"]+)"', line)
            if match and Path(match.group(1)).name in names:
                header = Path(match.group(1)).resolve()
                seen[str(header)] = {
                    "path": match.group(1),
                    "resolved_path": str(header),
                    "sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                    "observed_in": str(preprocessed),
                }
    record["device_headers"] = list(seen.values())
    record["header_provenance"] = "actual line markers from HGGC -keep preprocessed sources"
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
