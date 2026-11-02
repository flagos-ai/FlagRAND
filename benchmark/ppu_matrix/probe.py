"""Observe one native RNG API in a child process, including its C return codes."""

import argparse
import ctypes as C
import json
from pathlib import Path

CALL_MARKER = "PPU_NATIVE_CALL="
ENV_MARKER = "PPU_NATIVE_ENV="
RESULT_MARKER = "PPU_NATIVE_RESULT="


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--api", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    result = {"engine": args.engine, "api": args.api, "n": 1024, "status": "error", "calls": []}
    handle = None
    lib = None

    def call(name, *values):
        status = int(getattr(lib, name)(*values))
        observation = {"function": name, "status": status}
        result["calls"].append(observation)
        print(CALL_MARKER + json.dumps(observation), flush=True)
        if status:
            raise RuntimeError(f"{name} returned ACRAND status {status}")

    try:
        import torch

        from benchmark.ppu_matrix.baseline import ACRAND_TYPES, FUNCTIONS, load_library

        lib = load_library(args.library)
        version = C.c_int()
        call("acrandGetVersion", C.byref(version))
        environment = {
            "native_library": args.library,
            "native_library_resolved": str(Path(args.library).resolve()),
            "native_version": version.value,
            "prefix": "acrand",
        }
        print(ENV_MARKER + json.dumps(environment), flush=True)
        result["environment"] = environment
        dtype = (
            torch.int32
            if args.api in {"raw32", "poisson"}
            else (torch.float64 if args.api.endswith("f64") else torch.float32)
        )
        output = torch.empty(1024, dtype=dtype, device="cuda")
        created = C.c_void_p()
        call("acrandCreateGenerator", C.byref(created), ACRAND_TYPES[args.engine])
        handle = created
        call("acrandSetPseudoRandomGeneratorSeed", handle, C.c_ulonglong(73))
        if args.engine != "mtgp32":
            call("acrandSetGeneratorOffset", handle, C.c_ulonglong(0))
        call("acrandSetStream", handle, C.c_void_p(int(torch.cuda.current_stream().cuda_stream)))
        name, scalar = FUNCTIONS[args.api]
        values = [handle, C.c_void_p(output.data_ptr()), C.c_size_t(output.numel())]
        if args.api == "poisson":
            values.append(C.c_double(10.0))
        elif scalar is not None:
            values.extend((scalar(0.0), scalar(1.0)))
        call(name, *values)
        torch.cuda.synchronize()
        result["status"] = "pass"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if handle is not None:
            try:
                call("acrandDestroyGenerator", handle)
            except Exception as error:
                result.update(status="error", error=f"{type(error).__name__}: {error}")
        print(RESULT_MARKER + json.dumps(result), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
