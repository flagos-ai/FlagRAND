"""ctypes wrapper for the MetaX native mcRAND baseline."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import torch

MCRAND_RNG_PSEUDO_PHILOX4_32_10 = 161


class McrandError(RuntimeError):
    pass


class McrandGenerator:
    def __init__(
        self,
        *,
        seed: int,
        offset: int = 0,
        library_path: str | None = None,
        rng_type: int = MCRAND_RNG_PSEUDO_PHILOX4_32_10,
    ) -> None:
        seed_value = int(seed)
        offset_value = int(offset)
        if not 0 <= seed_value <= (1 << 64) - 1:
            raise ValueError(f"mcRAND seed must be in [0, 2^64-1], got {seed!r}.")
        if offset_value < 0:
            raise ValueError(f"mcRAND offset must be non-negative, got {offset!r}.")
        self._lib = _load_mcrand(library_path)
        self._handle = ctypes.c_void_p()
        _check(
            self._lib.mcrandCreateGenerator(
                ctypes.byref(self._handle),
                rng_type,
            ),
            "mcrandCreateGenerator",
        )
        self._closed = False
        try:
            _check(
                self._lib.mcrandSetPseudoRandomGeneratorSeed(
                    self._handle,
                    ctypes.c_ulonglong(seed_value),
                ),
                "mcrandSetPseudoRandomGeneratorSeed",
            )
            if offset_value or rng_type != 141:
                _check(
                    self._lib.mcrandSetGeneratorOffset(self._handle, ctypes.c_ulonglong(offset_value)),
                    "mcrandSetGeneratorOffset",
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _check(self._lib.mcrandDestroyGenerator(self._handle), "mcrandDestroyGenerator")

    def generate(self, distribution: str, output: torch.Tensor, *, lam: float = 10.0) -> None:
        if output.device.type != "cuda" or not output.is_contiguous():
            raise ValueError("mcRAND: output must be a contiguous CUDA-compatible tensor.")
        if output.numel() == 0:
            return
        self._set_current_stream()
        ptr = ctypes.c_void_p(output.data_ptr())
        n = ctypes.c_size_t(output.numel())
        if distribution == "poisson":
            status = self._lib.mcrandGeneratePoisson(self._handle, ptr, n, ctypes.c_double(lam))
            _check(status, "mcrandGeneratePoisson")
        elif distribution == "raw32":
            _require_dtype(output, torch.int32, distribution)
            status = self._lib.mcrandGenerate(self._handle, ptr, n)
            _check(status, "mcrandGenerate")
        elif distribution == "uniform_f32":
            _require_dtype(output, torch.float32, distribution)
            status = self._lib.mcrandGenerateUniform(self._handle, ptr, n)
            _check(status, "mcrandGenerateUniform")
        elif distribution == "uniform_f64":
            _require_dtype(output, torch.float64, distribution)
            status = self._lib.mcrandGenerateUniformDouble(self._handle, ptr, n)
            _check(status, "mcrandGenerateUniformDouble")
        elif distribution == "normal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.mcrandGenerateNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "mcrandGenerateNormal")
        elif distribution == "normal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.mcrandGenerateNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "mcrandGenerateNormalDouble")
        elif distribution == "lognormal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.mcrandGenerateLogNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "mcrandGenerateLogNormal")
        elif distribution == "lognormal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.mcrandGenerateLogNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "mcrandGenerateLogNormalDouble")
        else:
            raise ValueError(f"mcRAND: unsupported distribution {distribution!r}.")

    def _set_current_stream(self) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self._lib.mcrandSetStream(self._handle, stream), "mcrandSetStream")

    def __enter__(self) -> McrandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def mcrand_library_path() -> str:
    candidates: list[str] = []
    for name in ("FLAGRAND_MCRAND_PATH", "MCRAND_LIBRARY_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(value)
    for name in ("MACA_PATH", "MACA_HOME", "MACA_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append(str(Path(value) / "lib" / "libmcrand.so"))
    candidates.extend(
        (
            "/opt/maca/lib/libmcrand.so",
            "/opt/maca-3.7.1/lib/libmcrand.so",
        )
    )
    candidates.extend(str(path) for path in sorted(Path("/opt").glob("maca-*/lib/libmcrand.so"), reverse=True))
    found = ctypes.util.find_library("mcrand")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and (candidate == found or Path(candidate).exists()):
            return candidate
    raise FileNotFoundError("Could not find libmcrand.so. Set FLAGRAND_MCRAND_PATH or MACA_PATH.")


def mcrand_available() -> bool:
    try:
        _load_mcrand()
    except Exception:
        return False
    return True


def mcrand_version() -> int:
    version = ctypes.c_int()
    _check(_load_mcrand().mcrandGetVersion(ctypes.byref(version)), "mcrandGetVersion")
    return int(version.value)


_MCRAND_LIBRARY: ctypes.CDLL | None = None


def _load_mcrand(library_path: str | None = None) -> ctypes.CDLL:
    global _MCRAND_LIBRARY
    if _MCRAND_LIBRARY is not None and library_path is None:
        return _MCRAND_LIBRARY

    path = library_path or mcrand_library_path()
    lib = ctypes.CDLL(path)
    _bind_signatures(lib)
    if library_path is None:
        _MCRAND_LIBRARY = lib
    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.mcrandCreateGenerator.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.mcrandCreateGenerator.restype = ctypes.c_int
    lib.mcrandDestroyGenerator.argtypes = [ctypes.c_void_p]
    lib.mcrandDestroyGenerator.restype = ctypes.c_int
    lib.mcrandGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.mcrandGetVersion.restype = ctypes.c_int
    lib.mcrandSetPseudoRandomGeneratorSeed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    lib.mcrandSetPseudoRandomGeneratorSeed.restype = ctypes.c_int
    lib.mcrandSetGeneratorOffset.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
    lib.mcrandSetGeneratorOffset.restype = ctypes.c_int
    lib.mcrandSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mcrandSetStream.restype = ctypes.c_int
    lib.mcrandGenerate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.mcrandGenerate.restype = ctypes.c_int
    lib.mcrandGeneratePoisson.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_double]
    lib.mcrandGeneratePoisson.restype = ctypes.c_int
    lib.mcrandGenerateUniform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.mcrandGenerateUniform.restype = ctypes.c_int
    lib.mcrandGenerateUniformDouble.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.mcrandGenerateUniformDouble.restype = ctypes.c_int
    lib.mcrandGenerateNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.mcrandGenerateNormal.restype = ctypes.c_int
    lib.mcrandGenerateNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.mcrandGenerateNormalDouble.restype = ctypes.c_int
    lib.mcrandGenerateLogNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.mcrandGenerateLogNormal.restype = ctypes.c_int
    lib.mcrandGenerateLogNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.mcrandGenerateLogNormalDouble.restype = ctypes.c_int


def _require_dtype(output: torch.Tensor, dtype: torch.dtype, distribution: str) -> None:
    if output.dtype != dtype:
        raise TypeError(f"mcRAND {distribution}: expected {dtype}, got {output.dtype}.")


def _require_even(output: torch.Tensor, distribution: str) -> None:
    if output.numel() % 2 != 0:
        raise ValueError(f"mcRAND {distribution}: output length must be even.")


_STATUS_NAMES = {
    0: "MCRAND_STATUS_SUCCESS",
    100: "MCRAND_STATUS_VERSION_MISMATCH",
    101: "MCRAND_STATUS_NOT_INITIALIZED",
    102: "MCRAND_STATUS_ALLOCATION_FAILED",
    103: "MCRAND_STATUS_TYPE_ERROR",
    104: "MCRAND_STATUS_OUT_OF_RANGE",
    105: "MCRAND_STATUS_LENGTH_NOT_MULTIPLE",
    106: "MCRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
    201: "MCRAND_STATUS_LAUNCH_FAILURE",
    202: "MCRAND_STATUS_PREEXISTING_FAILURE",
    203: "MCRAND_STATUS_INITIALIZATION_FAILED",
    204: "MCRAND_STATUS_ARCH_MISMATCH",
    999: "MCRAND_STATUS_INTERNAL_ERROR",
}


def _check(status: int, function: str) -> None:
    if int(status) == 0:
        return
    name = _STATUS_NAMES.get(int(status), f"unknown status {int(status)}")
    raise McrandError(f"{function} failed with {name}.")
