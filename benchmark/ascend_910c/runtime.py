"""Direct Runtime random tasks with a device-managed counter."""

import ctypes as C
from functools import lru_cache
from time import perf_counter

import torch
import torch_npu

from .runtime_build import prepare_runtime_baseline

APIS = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3}


@lru_cache(maxsize=None)
def load_library(path):
    library = C.CDLL(path)
    library.runtime_rng_last_operation.restype = C.c_char_p
    library.runtime_rng_recent_error.restype = C.c_char_p
    library.runtime_rng_bind.argtypes = [
        C.c_int,
        C.c_uint64,
        C.c_uint64,
        C.c_void_p,
        C.c_void_p,
        C.POINTER(C.c_void_p),
    ]
    library.runtime_rng_generate.argtypes = [C.c_void_p, C.c_void_p]
    library.runtime_rng_synchronize.argtypes = [C.c_void_p]
    library.runtime_rng_reset.argtypes = [C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64]
    library.runtime_rng_read_counter.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
    library.runtime_rng_task_types.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
    library.runtime_rng_task_types.restype = None
    library.runtime_rng_workspace_bytes.argtypes = [C.c_void_p]
    library.runtime_rng_workspace_bytes.restype = C.c_uint64
    library.runtime_rng_destroy.argtypes = [C.c_void_p]
    return library


def check(library, status):
    if status:
        raise RuntimeError(
            f"{library.runtime_rng_last_operation().decode()}: CANN status {status}; "
            f"{library.runtime_rng_recent_error()}"
        )


class NativeRuntimeGenerator:
    """Bind raw32 or an FP32 distribution without initializing global ACL."""

    def __init__(self, output, api, seed, lam=None, build_dir=None):
        self.output, self.api, self.seed = output, api, seed
        self.device_index = output.device.index
        self.library_path = prepare_runtime_baseline(build_dir)
        self.library = load_library(self.library_path)
        self.read_stream = torch_npu._C._npu_getCurrentRawStream
        self.stream = torch.npu.current_stream(output.device)
        self.stream_handle = self.read_stream(self.device_index)
        self.handle = C.c_void_p()
        start = perf_counter()
        check(
            self.library,
            self.library.runtime_rng_bind(
                APIS[api], output.numel(), seed, output.data_ptr(), self.stream_handle, C.byref(self.handle)
            ),
        )
        self.setup_and_first_call_ms = (perf_counter() - start) * 1000
        self.workspace_bytes = self.library.runtime_rng_workspace_bytes(self.handle)
        output.record_stream(self.stream)

    def generate(self):
        handle = self.read_stream(self.device_index)
        if handle != self.stream_handle:
            current = torch.npu.current_stream(self.output.device)
            current.wait_stream(self.stream)
            self.stream, self.stream_handle = current, self.read_stream(self.device_index)
            self.output.record_stream(current)
        check(self.library, self.library.runtime_rng_generate(self.handle, self.stream_handle))
        return self.output

    def synchronize(self):
        self.read_stream(self.device_index)
        check(self.library, self.library.runtime_rng_synchronize(self.handle))

    def reset(self, seed=None, offset=0):
        self.synchronize()
        if seed is not None:
            self.seed = seed
        check(self.library, self.library.runtime_rng_reset(self.handle, self.seed, offset, 0))

    def counter(self):
        self.synchronize()
        value = (C.c_uint64 * 2)()
        check(self.library, self.library.runtime_rng_read_counter(self.handle, value))
        return list(value)

    def task_types(self):
        value = (C.c_int * 3)()
        self.library.runtime_rng_task_types(self.handle, value)
        return dict(zip(("template", "call_before", "call_after"), value))

    def close(self):
        self.synchronize()
        check(self.library, self.library.runtime_rng_destroy(self.handle))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def baseline_metadata(api, dtype):
    function = "normal" if api in ("normal", "lognormal") else "uniform"
    timing_scope = (
        "ctypes, current-stream/task-queue processing, local task copy, Runtime random task submission "
        "and execution, vendor-internal parameter transfer, batch synchronization"
    )
    components = [f"aclrtRandomNumAsync({function})"]
    if api == "lognormal":
        components.append("aclnnInplaceExp")
        timing_scope += "; Exp workspace query, fresh executor and execute included"
    return {
        "provider": "CANN Runtime native C",
        "dtype": str(dtype).removeprefix("torch."),
        "components": components,
        "comparison_scope": "same_distribution_not_same_generator_or_bitstream",
        "timing_scope": timing_scope + "; output/generator setup and reset excluded",
        "offset_policy": "device counter advances natively; reset offset is counter[0], counter[1]=0",
        "counter_observation": "CANN 9.1 raw32/uniform32/normal32 samples advance ceil(N/4) blocks",
        "state_upload": "no per-call host counter upload",
        "execution_route": "Runtime random task; Exp uses ACLNN two-phase operator when requested",
        "precision_scope": "uint32 bits in int32 storage" if api == "raw32" else "native float32",
        "raw_reference_scope": (
            "32-bit output with observed low-bit resolution and both high-bit values; narrow min/max "
            "requests did not constrain observed values, so no integer interval contract is asserted"
            if api == "raw32"
            else None
        ),
        "uniform_endpoint_contract": "Runtime endpoint inclusion is not asserted" if api == "uniform" else None,
        "task_copy_reason": "CANN 9.1 writes the RT datatype enum into the submitted task; ACL template retained",
    }
