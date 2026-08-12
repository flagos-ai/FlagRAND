"""ctypes wrapper for the PPU cuRAND-compatible baseline."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import torch

CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161


class PpuCurandError(RuntimeError):
    pass


class PpuCurandGenerator:
    def __init__(self, *, seed: int, offset: int = 0, library_path: str | None = None) -> None:
        seed_value = int(seed)
        offset_value = int(offset)
        if not 0 <= seed_value <= (1 << 64) - 1:
            raise ValueError(f"PPU cuRAND seed must be in [0, 2^64-1], got {seed!r}.")
        if offset_value < 0:
            raise ValueError(f"PPU cuRAND offset must be non-negative, got {offset!r}.")
        self._lib = _load_curand(library_path)
        self._handle = ctypes.c_void_p()
        _check(
            self._lib.curandCreateGenerator(
                ctypes.byref(self._handle),
                CURAND_RNG_PSEUDO_PHILOX4_32_10,
            ),
            "curandCreateGenerator",
        )
        self._closed = False
        try:
            _check(
                self._lib.curandSetPseudoRandomGeneratorSeed(
                    self._handle,
                    ctypes.c_ulonglong(seed_value),
                ),
                "curandSetPseudoRandomGeneratorSeed",
            )
            _check(
                self._lib.curandSetGeneratorOffset(
                    self._handle,
                    ctypes.c_ulonglong(offset_value),
                ),
                "curandSetGeneratorOffset",
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _check(self._lib.curandDestroyGenerator(self._handle), "curandDestroyGenerator")

    def generate(self, distribution: str, output: torch.Tensor) -> None:
        if output.device.type != "cuda" or not output.is_contiguous():
            raise ValueError("PPU cuRAND: output must be contiguous CUDA tensor.")
        if output.numel() == 0:
            return
        self._set_current_stream()
        ptr = ctypes.c_void_p(output.data_ptr())
        n = ctypes.c_size_t(output.numel())
        if distribution == "raw32":
            _require_dtype(output, torch.int32, distribution)
            status = self._lib.curandGenerate(self._handle, ptr, n)
            _check(status, "curandGenerate")
        elif distribution == "uniform_f32":
            _require_dtype(output, torch.float32, distribution)
            status = self._lib.curandGenerateUniform(self._handle, ptr, n)
            _check(status, "curandGenerateUniform")
        elif distribution == "uniform_f64":
            _require_dtype(output, torch.float64, distribution)
            status = self._lib.curandGenerateUniformDouble(self._handle, ptr, n)
            _check(status, "curandGenerateUniformDouble")
        elif distribution == "normal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.curandGenerateNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "curandGenerateNormal")
        elif distribution == "normal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.curandGenerateNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "curandGenerateNormalDouble")
        elif distribution == "lognormal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.curandGenerateLogNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "curandGenerateLogNormal")
        elif distribution == "lognormal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.curandGenerateLogNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "curandGenerateLogNormalDouble")
        else:
            raise ValueError(f"PPU cuRAND: unsupported distribution {distribution!r}.")

    def _set_current_stream(self) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self._lib.curandSetStream(self._handle, stream), "curandSetStream")

    def __enter__(self) -> PpuCurandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def ppu_curand_library_path() -> str:
    candidates: list[str] = []
    for name in ("FLAGRAND_PPU_CURAND_PATH", "CURAND_LIBRARY_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(value)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "lib64" / "libcurand.so"))
    candidates.extend(
        (
            "/usr/local/PPU_SDK/CUDA_SDK/lib64/libcurand.so",
            "/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcurand.so",
        )
    )
    found = ctypes.util.find_library("curand")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and (candidate == found or Path(candidate).exists()):
            return candidate
    raise FileNotFoundError("Could not find libcurand.so. Set FLAGRAND_PPU_CURAND_PATH or CUDA_HOME.")


def ppu_curand_available() -> bool:
    try:
        _load_curand()
    except Exception:
        return False
    return True


_CURAND_LIBRARY: ctypes.CDLL | None = None


def _load_curand(library_path: str | None = None) -> ctypes.CDLL:
    global _CURAND_LIBRARY
    if _CURAND_LIBRARY is not None and library_path is None:
        return _CURAND_LIBRARY

    path = library_path or ppu_curand_library_path()
    lib = ctypes.CDLL(path)
    _bind_signatures(lib)
    if library_path is None:
        _CURAND_LIBRARY = lib
    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.curandCreateGenerator.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.curandCreateGenerator.restype = ctypes.c_int
    lib.curandDestroyGenerator.argtypes = [ctypes.c_void_p]
    lib.curandDestroyGenerator.restype = ctypes.c_int
    lib.curandSetPseudoRandomGeneratorSeed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    lib.curandSetPseudoRandomGeneratorSeed.restype = ctypes.c_int
    lib.curandSetGeneratorOffset.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
    lib.curandSetGeneratorOffset.restype = ctypes.c_int
    lib.curandSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.curandSetStream.restype = ctypes.c_int
    lib.curandGenerate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.curandGenerate.restype = ctypes.c_int
    lib.curandGenerateUniform.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.curandGenerateUniform.restype = ctypes.c_int
    lib.curandGenerateUniformDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.curandGenerateUniformDouble.restype = ctypes.c_int
    lib.curandGenerateNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.curandGenerateNormal.restype = ctypes.c_int
    lib.curandGenerateNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.curandGenerateNormalDouble.restype = ctypes.c_int
    lib.curandGenerateLogNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.curandGenerateLogNormal.restype = ctypes.c_int
    lib.curandGenerateLogNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.curandGenerateLogNormalDouble.restype = ctypes.c_int


def _require_dtype(output: torch.Tensor, dtype: torch.dtype, distribution: str) -> None:
    if output.dtype != dtype:
        raise TypeError(f"PPU cuRAND {distribution}: expected {dtype}, got {output.dtype}.")


def _require_even(output: torch.Tensor, distribution: str) -> None:
    if output.numel() % 2 != 0:
        raise ValueError(f"PPU cuRAND {distribution}: output length must be even.")


_STATUS_NAMES = {
    0: "CURAND_STATUS_SUCCESS",
    100: "CURAND_STATUS_VERSION_MISMATCH",
    101: "CURAND_STATUS_NOT_INITIALIZED",
    102: "CURAND_STATUS_ALLOCATION_FAILED",
    103: "CURAND_STATUS_TYPE_ERROR",
    104: "CURAND_STATUS_OUT_OF_RANGE",
    105: "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
    106: "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
    201: "CURAND_STATUS_LAUNCH_FAILURE",
    202: "CURAND_STATUS_PREEXISTING_FAILURE",
    203: "CURAND_STATUS_INITIALIZATION_FAILED",
    204: "CURAND_STATUS_ARCH_MISMATCH",
    999: "CURAND_STATUS_INTERNAL_ERROR",
}


def _check(status: int, function: str) -> None:
    if int(status) == 0:
        return
    name = _STATUS_NAMES.get(int(status), f"unknown status {int(status)}")
    raise PpuCurandError(f"{function} failed with {name}.")
