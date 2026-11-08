"""Direct CANN C calls with preallocated output and workspace."""

import ctypes as C
from functools import lru_cache
from time import perf_counter

import torch

from .build import prepare_native_baseline

APIS = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3, "poisson": 5}
DTYPES = {torch.int32: 0, torch.float32: 1, torch.float64: 2}


@lru_cache(maxsize=None)
def load_library(path):
    library = C.CDLL(path)
    library.cann_rng_last_operation.restype = C.c_char_p
    library.cann_rng_recent_error.restype = C.c_char_p
    library.cann_rng_bind.argtypes = [
        C.c_int,
        C.c_int,
        C.c_int64,
        C.c_int64,
        C.c_int64,
        C.c_double,
        C.c_void_p,
        C.c_void_p,
        C.POINTER(C.c_void_p),
    ]
    library.cann_rng_set_state.argtypes = [C.c_void_p, C.c_int64, C.c_int64]
    library.cann_rng_reserve_calls.argtypes = [C.c_void_p, C.c_int64]
    library.cann_rng_generate.argtypes = [C.c_void_p]
    library.cann_rng_synchronize.argtypes = [C.c_void_p]
    library.cann_rng_destroy.argtypes = [C.c_void_p]
    library.cann_rng_workspace_bytes.argtypes = [C.c_void_p]
    library.cann_rng_workspace_bytes.restype = C.c_uint64
    return library


def check(library, status):
    if status != 0:
        raise RuntimeError(
            f"{library.cann_rng_last_operation().decode()}: CANN status {status}\n" f"{library.cann_rng_recent_error()}"
        )


class NativeCannGenerator:
    """Bind one contiguous NPU output on the current stream.

    Each timed batch may enqueue at most ``calls_per_batch`` calls before
    ``synchronize()``. Graph calls keep a pinned seed/counter slot per upload.
    The vendor algorithms are distribution references, not engine equivalents.
    """

    def __init__(self, output, api, seed, lam=None, build_dir=None, *, calls_per_batch=1024, offset_step=10):
        self.output = output
        self.api = api
        self.seed = seed
        if api == "raw32":
            self.offset_step = (output.numel() + 3) // 4
        elif output.dtype == torch.float64:
            self.offset_step = (output.numel() + 1) // 2
        else:
            self.offset_step = offset_step
        self.library_path = prepare_native_baseline(build_dir)
        self.library = load_library(self.library_path)
        self.handle = C.c_void_p()
        self.stream = torch.npu.current_stream(output.device)
        start = perf_counter()
        check(
            self.library,
            self.library.cann_rng_bind(
                APIS[api],
                DTYPES[output.dtype],
                output.numel(),
                seed,
                self.offset_step,
                10.0 if lam is None else lam,
                output.data_ptr(),
                self.stream.npu_stream,
                C.byref(self.handle),
            ),
        )
        check(self.library, self.library.cann_rng_reserve_calls(self.handle, calls_per_batch))
        self.setup_and_first_call_ms = (perf_counter() - start) * 1000
        self.workspace_bytes = self.library.cann_rng_workspace_bytes(self.handle)

    def generate(self):
        check(self.library, self.library.cann_rng_generate(self.handle))
        return self.output

    def reset(self, seed=None, offset=0):
        if seed is not None:
            self.seed = seed
        check(self.library, self.library.cann_rng_set_state(self.handle, self.seed, offset))

    def synchronize(self):
        check(self.library, self.library.cann_rng_synchronize(self.handle))

    def close(self):
        check(self.library, self.library.cann_rng_destroy(self.handle))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def baseline_metadata(api, dtype):
    graph = api in ("raw32", "poisson") or dtype == torch.float64
    components = {
        "raw32": ["StatelessRandomUniformFullIntV2"],
        "uniform": ["aclnnInplaceUniform"],
        "normal": ["aclnnInplaceNormal"],
        "lognormal": ["aclnnInplaceNormal", "aclnnInplaceExp"],
        "poisson": ["StatelessRandomPoisson"],
    }[api]
    if dtype == torch.float64:
        components = {
            "uniform": ["StatelessRandomUniformV2"],
            "normal": ["StatelessRandomNormalV2"],
            "lognormal": ["StatelessRandomNormalV2", "aclnnInplaceExp"],
        }[api]
    if api == "raw32":
        offset_policy = "counter[0] advances ceil(N/4) Philox blocks; counter[1]=0"
    elif dtype == torch.float64:
        offset_policy = "counter[0] advances ceil(N/2) Philox blocks; counter[1]=0"
    else:
        offset_policy = "explicit per-call offset increment (default 10)"
    if graph:
        timing_scope = "ctypes call, 16-byte seed/counter upload, CompileAndExecute, batch synchronization"
        if api == "lognormal":
            timing_scope += "; Exp workspace query, fresh executor and execute included"
    else:
        timing_scope = "ctypes call, workspace query and fresh executor, execute, batch synchronization"
    return {
        "provider": "CANN native C",
        "dtype": str(dtype).removeprefix("torch."),
        "components": components,
        "comparison_scope": "same_distribution_not_same_generator_or_bitstream",
        "timing_scope": timing_scope + "; setup and reset excluded",
        "offset_policy": offset_policy + "; no cross-shape split-continuity claim",
        "execution_route": (
            "KERNEL_AICPU observed on CANN 9.1; binary64 Exp also executes on AI CPU"
            if graph
            else "ACLNN two-phase operator"
        ),
        "raw_reference_scope": (
            "FullIntV2 complete int32 range; low-bit resolution validated separately" if api == "raw32" else None
        ),
        "state_upload": (
            "16 bytes per call on the same stream; pinned slots allocated before timing" if graph else None
        ),
        "algorithm": "Philox (alg=1), key=[seed]" if graph and api != "poisson" else None,
        "precision_scope": (
            "binary64 graph distribution; not the FP32-grid ACLNN sample converted to binary64"
            if dtype == torch.float64
            else "native output dtype"
        ),
        "graph_compile_option": "ACL_OP_JIT_COMPILE=enable" if graph else None,
    }
