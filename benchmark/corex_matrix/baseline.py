from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

from benchmark.ppu_matrix.curand import PpuCurandGenerator

COREX_FUNCTIONS = {
    "raw32": ("curandGenerate", None),
    "uniform_f32": ("curandGenerateUniform", None),
    "uniform_f64": ("curandGenerateUniformDouble", None),
    "normal_f32": ("curandGenerateNormal", ctypes.c_float),
    "normal_f64": ("curandGenerateNormalDouble", ctypes.c_double),
    "lognormal_f32": ("curandGenerateLogNormal", ctypes.c_float),
    "lognormal_f64": ("curandGenerateLogNormalDouble", ctypes.c_double),
    "poisson": ("curandGeneratePoisson", ctypes.c_double),
}


class CorexCurandError(RuntimeError):
    def __init__(self, status: int, operation: str) -> None:
        self.status = int(status)
        self.operation = operation
        super().__init__(f"{operation} returned CoreX cuRAND status {self.status}")


def _check_corex(status: int, operation: str) -> None:
    if status:
        raise CorexCurandError(status, operation)


def corex_curand_library_path() -> str:
    candidates = [
        os.environ.get("FLAGRAND_COREX_CURAND_PATH"),
        os.environ.get("CURAND_LIBRARY_PATH"),
        "/usr/local/corex/lib64/libcurand.so",
        "/usr/local/corex-4.4.0/lib64/libcurand.so",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("CoreX libcurand.so was not found.")


@lru_cache(maxsize=None)
def _load_corex_library(path: str):
    library = ctypes.CDLL(path)
    signatures = {
        "curandGetVersion": [ctypes.POINTER(ctypes.c_int)],
        "curandCreateGenerator": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
        "curandDestroyGenerator": [ctypes.c_void_p],
        "curandSetPseudoRandomGeneratorSeed": [ctypes.c_void_p, ctypes.c_ulonglong],
        "curandSetGeneratorOffset": [ctypes.c_void_p, ctypes.c_ulonglong],
        "curandSetStream": [ctypes.c_void_p, ctypes.c_void_p],
    }
    for api, (name, scalar) in COREX_FUNCTIONS.items():
        args = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        if api == "poisson":
            args.append(ctypes.c_double)
        elif scalar is not None:
            args.extend((scalar, scalar))
        signatures[name] = args
    for name, args in signatures.items():
        function = getattr(library, name)
        function.argtypes = args
        function.restype = ctypes.c_int
    return library


def corex_curand_version() -> int:
    version = ctypes.c_int()
    _check_corex(
        _load_corex_library(corex_curand_library_path()).curandGetVersion(ctypes.byref(version)), "curandGetVersion"
    )
    return version.value


class CorexRngGenerator:
    """CoreX host generator selected by its installed cuRAND enum value."""

    def __init__(self, *, seed: int, offset: int = 0, rng_type: int = 161) -> None:
        self._lib = _load_corex_library(corex_curand_library_path())
        self._handle = ctypes.c_void_p()
        self._stream = None
        _check_corex(self._lib.curandCreateGenerator(ctypes.byref(self._handle), rng_type), "curandCreateGenerator")
        try:
            _check_corex(
                self._lib.curandSetPseudoRandomGeneratorSeed(self._handle, ctypes.c_ulonglong(seed)),
                "curandSetPseudoRandomGeneratorSeed",
            )
            # A seeded MTGP32 starts at zero; its host API has no absolute offset.
            if offset or rng_type != 141:
                _check_corex(
                    self._lib.curandSetGeneratorOffset(self._handle, ctypes.c_ulonglong(offset)),
                    "curandSetGeneratorOffset",
                )
        except Exception:
            self.close()
            raise

    def generate(self, distribution, output, *, mean=0.0, stddev=1.0, lam=10.0):
        stream = int(torch.cuda.current_stream().cuda_stream)
        if stream != self._stream:
            _check_corex(self._lib.curandSetStream(self._handle, ctypes.c_void_p(stream)), "curandSetStream")
            self._stream = stream
        name, scalar = COREX_FUNCTIONS[distribution]
        args = [self._handle, ctypes.c_void_p(output.data_ptr()), ctypes.c_size_t(output.numel())]
        if distribution == "poisson":
            args.append(ctypes.c_double(lam))
        elif scalar is not None:
            args.extend((scalar(mean), scalar(stddev)))
        _check_corex(getattr(self._lib, name)(*args), name)

    def close(self):
        _check_corex(self._lib.curandDestroyGenerator(self._handle), "curandDestroyGenerator")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class CorexCurandGenerator:
    def __init__(self, *, seed: int) -> None:
        self._generator = PpuCurandGenerator(
            seed=seed,
            library_path=corex_curand_library_path(),
        )

    def generate(self, distribution: str, output: torch.Tensor) -> None:
        self._generator.generate(distribution, output)

    def close(self) -> None:
        self._generator.close()

    def __enter__(self) -> CorexCurandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def probe_native_fp64(seed):
    """Record actual native return codes instead of assuming every CoreX has the same FP64 support."""
    result = {}
    with CorexCurandGenerator(seed=seed) as wrapper:
        generator = wrapper._generator
        version = ctypes.c_int()
        generator._lib.curandGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version_status = int(generator._lib.curandGetVersion(ctypes.byref(version)))
        result["version"] = {"status": version_status, "value": version.value}
        generator._set_current_stream()
        output = torch.empty(2, device="cuda", dtype=torch.float64)
        common = (generator._handle, ctypes.c_void_p(output.data_ptr()), ctypes.c_size_t(2))
        for api, function, args in (
            ("uniform_f64", "curandGenerateUniformDouble", common),
            ("normal_f64", "curandGenerateNormalDouble", common + (ctypes.c_double(0), ctypes.c_double(1))),
            ("lognormal_f64", "curandGenerateLogNormalDouble", common + (ctypes.c_double(0), ctypes.c_double(1))),
        ):
            status = int(getattr(generator._lib, function)(*args))
            result[api] = {
                "function": function,
                "return_code": status,
                "status": "pass" if status == 0 else "unsupported" if status == 106 else "error",
            }
        torch.cuda.synchronize()
    return result
