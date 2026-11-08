"""PPU ACRAND host API, using persistent generators and the current Torch CUDA stream."""

from __future__ import annotations

import ctypes as C
import os
from functools import lru_cache

import torch

from benchmark.ppu_matrix.acrand import ppu_acrand_library_path

# Installed PPU acrand.h, archived in
# reports/xorwow-optimization-20260930/ppu-native-readonly.json (2026-09-30).
ACRAND_RNG_PSEUDO_XORWOW = 101
ACRAND_RNG_PSEUDO_MRG32K3A = 121
ACRAND_RNG_PSEUDO_MTGP32 = 141
ACRAND_RNG_PSEUDO_PHILOX4_32_10 = 161
ACRAND_TYPES = {
    "philox": ACRAND_RNG_PSEUDO_PHILOX4_32_10,
    "xorwow": ACRAND_RNG_PSEUDO_XORWOW,
    "mrg32k3a": ACRAND_RNG_PSEUDO_MRG32K3A,
    "mtgp32": ACRAND_RNG_PSEUDO_MTGP32,
}
FUNCTIONS = {
    "raw32": ("acrandGenerate", None),
    "uniform_f32": ("acrandGenerateUniform", None),
    "uniform_f64": ("acrandGenerateUniformDouble", None),
    "normal_f32": ("acrandGenerateNormal", C.c_float),
    "normal_f64": ("acrandGenerateNormalDouble", C.c_double),
    "lognormal_f32": ("acrandGenerateLogNormal", C.c_float),
    "lognormal_f64": ("acrandGenerateLogNormalDouble", C.c_double),
    "poisson": ("acrandGeneratePoisson", C.c_double),
}


def acrand_library_path():
    return os.environ.get("FLAGRAND_ACRAND_PATH") or ppu_acrand_library_path()


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} returned ACRAND status {status}")


@lru_cache(maxsize=None)
def load_library(path):
    lib = C.CDLL(path)
    signatures = {
        "acrandGetVersion": [C.POINTER(C.c_int)],
        "acrandCreateGenerator": [C.POINTER(C.c_void_p), C.c_int],
        "acrandDestroyGenerator": [C.c_void_p],
        "acrandSetPseudoRandomGeneratorSeed": [C.c_void_p, C.c_ulonglong],
        "acrandSetGeneratorOffset": [C.c_void_p, C.c_ulonglong],
        "acrandSetStream": [C.c_void_p, C.c_void_p],
    }
    for name, scalar in FUNCTIONS.values():
        parameters = [C.c_void_p, C.c_void_p, C.c_size_t]
        if name == "acrandGeneratePoisson":
            parameters += [C.c_double]
        elif scalar:
            parameters += [scalar, scalar]
        signatures[name] = parameters
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = C.c_int
    return lib


def acrand_version():
    value = C.c_int()
    check(load_library(acrand_library_path()).acrandGetVersion(C.byref(value)), "acrandGetVersion")
    return value.value


class AcrandGenerator:
    def __init__(self, *, seed, offset=0, rng_type=ACRAND_RNG_PSEUDO_PHILOX4_32_10):
        self.lib = load_library(acrand_library_path())
        self.handle = C.c_void_p()
        self.stream = None
        check(
            self.lib.acrandCreateGenerator(C.byref(self.handle), rng_type),
            "acrandCreateGenerator",
        )
        check(
            self.lib.acrandSetPseudoRandomGeneratorSeed(self.handle, C.c_ulonglong(seed)),
            "acrandSetPseudoRandomGeneratorSeed",
        )
        # MTGP32 has no absolute-offset API, including setting offset zero.
        # A fresh seeded generator already starts at its initial position.
        if offset or rng_type != ACRAND_RNG_PSEUDO_MTGP32:
            check(self.lib.acrandSetGeneratorOffset(self.handle, C.c_ulonglong(offset)), "acrandSetGeneratorOffset")

    def generate(self, distribution, output, *, mean=0.0, stddev=1.0, lam=10.0):
        stream = int(torch.cuda.current_stream().cuda_stream)
        if stream != self.stream:
            check(self.lib.acrandSetStream(self.handle, C.c_void_p(stream)), "acrandSetStream")
            self.stream = stream
        name, scalar = FUNCTIONS[distribution]
        args = [self.handle, C.c_void_p(output.data_ptr()), C.c_size_t(output.numel())]
        if distribution == "poisson":
            args += [C.c_double(lam)]
        elif scalar:
            args += [scalar(mean), scalar(stddev)]
        check(getattr(self.lib, name)(*args), name)

    def close(self):
        check(self.lib.acrandDestroyGenerator(self.handle), "acrandDestroyGenerator")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
