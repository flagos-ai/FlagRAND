"""DTK hipRAND host API, using persistent generators and the current Torch HIP stream."""

from __future__ import annotations

import ctypes as C
import os
from functools import lru_cache
from pathlib import Path

import torch

# Official ROCm 6.3.3 hiprand.h:135-139; these differ from cuRAND/mcRAND.
# https://github.com/ROCm/hipRAND/blob/rocm-6.3.3/library/include/hiprand/hiprand.h
HIPRAND_RNG_PSEUDO_XORWOW = 401
HIPRAND_RNG_PSEUDO_MRG32K3A = 402
HIPRAND_RNG_PSEUDO_MTGP32 = 403
HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405
HIPRAND_TYPES = {
    "philox": HIPRAND_RNG_PSEUDO_PHILOX4_32_10,
    "xorwow": HIPRAND_RNG_PSEUDO_XORWOW,
    "mrg32k3a": HIPRAND_RNG_PSEUDO_MRG32K3A,
    "mtgp32": HIPRAND_RNG_PSEUDO_MTGP32,
}
FUNCTIONS = {
    "raw32": ("hiprandGenerate", None),
    "uniform_f32": ("hiprandGenerateUniform", None),
    "uniform_f64": ("hiprandGenerateUniformDouble", None),
    "normal_f32": ("hiprandGenerateNormal", C.c_float),
    "normal_f64": ("hiprandGenerateNormalDouble", C.c_double),
    "lognormal_f32": ("hiprandGenerateLogNormal", C.c_float),
    "lognormal_f64": ("hiprandGenerateLogNormalDouble", C.c_double),
    "poisson": ("hiprandGeneratePoisson", C.c_double),
}


def hiprand_library_path():
    return os.environ.get(
        "FLAGRAND_HIPRAND_PATH", str(Path(os.environ.get("DTK_HOME", "/opt/dtk")) / "lib/libhiprand.so")
    )


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} returned hipRAND status {status}")


@lru_cache(maxsize=None)
def load_library(path):
    lib = C.CDLL(path)
    signatures = {
        "hiprandGetVersion": [C.POINTER(C.c_int)],
        "hiprandCreateGenerator": [C.POINTER(C.c_void_p), C.c_int],
        "hiprandDestroyGenerator": [C.c_void_p],
        "hiprandSetPseudoRandomGeneratorSeed": [C.c_void_p, C.c_ulonglong],
        "hiprandSetGeneratorOffset": [C.c_void_p, C.c_ulonglong],
        "hiprandSetStream": [C.c_void_p, C.c_void_p],
    }
    for name, scalar in FUNCTIONS.values():
        parameters = [C.c_void_p, C.c_void_p, C.c_size_t]
        if name == "hiprandGeneratePoisson":
            parameters += [C.c_double]
        elif scalar:
            parameters += [scalar, scalar]
        signatures[name] = parameters
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = C.c_int
    return lib


def hiprand_version():
    value = C.c_int()
    check(load_library(hiprand_library_path()).hiprandGetVersion(C.byref(value)), "hiprandGetVersion")
    return value.value


class HiprandGenerator:
    def __init__(self, *, seed, offset=0, rng_type=HIPRAND_RNG_PSEUDO_PHILOX4_32_10):
        self.lib = load_library(hiprand_library_path())
        self.handle = C.c_void_p()
        self.stream = None
        check(
            self.lib.hiprandCreateGenerator(C.byref(self.handle), rng_type),
            "hiprandCreateGenerator",
        )
        check(
            self.lib.hiprandSetPseudoRandomGeneratorSeed(self.handle, C.c_ulonglong(seed)),
            "hiprandSetPseudoRandomGeneratorSeed",
        )
        # MTGP32 has no absolute-offset API, including setting offset zero.
        # A fresh seeded generator already starts at its initial position.
        if offset or rng_type != HIPRAND_RNG_PSEUDO_MTGP32:
            check(self.lib.hiprandSetGeneratorOffset(self.handle, C.c_ulonglong(offset)), "hiprandSetGeneratorOffset")

    def generate(self, distribution, output, *, mean=0.0, stddev=1.0, lam=10.0):
        stream = int(torch.cuda.current_stream().cuda_stream)
        if stream != self.stream:
            check(self.lib.hiprandSetStream(self.handle, C.c_void_p(stream)), "hiprandSetStream")
            self.stream = stream
        name, scalar = FUNCTIONS[distribution]
        args = [self.handle, C.c_void_p(output.data_ptr()), C.c_size_t(output.numel())]
        if distribution == "poisson":
            args += [C.c_double(lam)]
        elif scalar:
            args += [scalar(mean), scalar(stddev)]
        check(getattr(self.lib, name)(*args), name)

    def close(self):
        check(self.lib.hiprandDestroyGenerator(self.handle), "hiprandDestroyGenerator")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
