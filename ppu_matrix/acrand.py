from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import torch


ACRAND_RNG_PSEUDO_PHILOX4_32_10 = 161


class PpuAcrandError(RuntimeError):
    pass


class PpuAcrandGenerator:
    def __init__(self, *, seed: int, offset: int = 0, library_path: str | None = None) -> None:
        seed_value = int(seed)
        offset_value = int(offset)
        if not 0 <= seed_value <= (1 << 64) - 1:
            raise ValueError(f"PPU acrand seed must be in [0, 2^64-1], got {seed!r}.")
        if offset_value < 0:
            raise ValueError(f"PPU acrand offset must be non-negative, got {offset!r}.")
        self._lib = _load_acrand(library_path)
        self._handle = ctypes.c_void_p()
        _check(
            self._lib.acrandCreateGenerator(
                ctypes.byref(self._handle),
                ACRAND_RNG_PSEUDO_PHILOX4_32_10,
            ),
            "acrandCreateGenerator",
        )
        self._closed = False
        try:
            _check(
                self._lib.acrandSetPseudoRandomGeneratorSeed(
                    self._handle,
                    ctypes.c_ulonglong(seed_value),
                ),
                "acrandSetPseudoRandomGeneratorSeed",
            )
            _check(
                self._lib.acrandSetGeneratorOffset(
                    self._handle,
                    ctypes.c_ulonglong(offset_value),
                ),
                "acrandSetGeneratorOffset",
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _check(self._lib.acrandDestroyGenerator(self._handle), "acrandDestroyGenerator")

    def generate(self, distribution: str, output: torch.Tensor) -> None:
        if output.device.type != "cuda" or not output.is_contiguous():
            raise ValueError("PPU acrand: output must be contiguous CUDA tensor.")
        if output.numel() == 0:
            return
        self._set_current_stream()
        ptr = ctypes.c_void_p(output.data_ptr())
        n = ctypes.c_size_t(output.numel())
        if distribution == "raw32":
            _require_dtype(output, torch.int32, distribution)
            status = self._lib.acrandGenerate(self._handle, ptr, n)
            _check(status, "acrandGenerate")
        elif distribution == "uniform_f32":
            _require_dtype(output, torch.float32, distribution)
            status = self._lib.acrandGenerateUniform(self._handle, ptr, n)
            _check(status, "acrandGenerateUniform")
        elif distribution == "uniform_f64":
            _require_dtype(output, torch.float64, distribution)
            status = self._lib.acrandGenerateUniformDouble(self._handle, ptr, n)
            _check(status, "acrandGenerateUniformDouble")
        elif distribution == "normal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.acrandGenerateNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "acrandGenerateNormal")
        elif distribution == "normal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.acrandGenerateNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "acrandGenerateNormalDouble")
        elif distribution == "lognormal_f32":
            _require_dtype(output, torch.float32, distribution)
            _require_even(output, distribution)
            status = self._lib.acrandGenerateLogNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(0.0),
                ctypes.c_float(1.0),
            )
            _check(status, "acrandGenerateLogNormal")
        elif distribution == "lognormal_f64":
            _require_dtype(output, torch.float64, distribution)
            _require_even(output, distribution)
            status = self._lib.acrandGenerateLogNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(0.0),
                ctypes.c_double(1.0),
            )
            _check(status, "acrandGenerateLogNormalDouble")
        else:
            raise ValueError(f"PPU acrand: unsupported distribution {distribution!r}.")

    def _set_current_stream(self) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check(self._lib.acrandSetStream(self._handle, stream), "acrandSetStream")

    def __enter__(self) -> PpuAcrandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def ppu_acrand_library_path() -> str:
    candidates: list[str] = []
    for name in ("FLAGRAND_PPU_ACRAND_PATH", "ACRAND_LIBRARY_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(value)
    candidates.append("/usr/local/PPU_SDK/targets/x86_64-linux/lib/libacrand.so")
    found = ctypes.util.find_library("acrand")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and (candidate == found or Path(candidate).exists()):
            return candidate
    raise FileNotFoundError(
        "Could not find libacrand.so. Set FLAGRAND_PPU_ACRAND_PATH."
    )


def ppu_acrand_available() -> bool:
    try:
        _load_acrand()
    except Exception:
        return False
    return True


_ACRAND_LIBRARY: ctypes.CDLL | None = None


def _load_acrand(library_path: str | None = None) -> ctypes.CDLL:
    global _ACRAND_LIBRARY
    if _ACRAND_LIBRARY is not None and library_path is None:
        return _ACRAND_LIBRARY

    path = library_path or ppu_acrand_library_path()
    lib = ctypes.CDLL(path)
    _bind_signatures(lib)
    if library_path is None:
        _ACRAND_LIBRARY = lib
    return lib


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.acrandCreateGenerator.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.acrandCreateGenerator.restype = ctypes.c_int
    lib.acrandDestroyGenerator.argtypes = [ctypes.c_void_p]
    lib.acrandDestroyGenerator.restype = ctypes.c_int
    lib.acrandSetPseudoRandomGeneratorSeed.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    lib.acrandSetPseudoRandomGeneratorSeed.restype = ctypes.c_int
    lib.acrandSetGeneratorOffset.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
    lib.acrandSetGeneratorOffset.restype = ctypes.c_int
    lib.acrandSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.acrandSetStream.restype = ctypes.c_int
    lib.acrandGenerate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.acrandGenerate.restype = ctypes.c_int
    lib.acrandGenerateUniform.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.acrandGenerateUniform.restype = ctypes.c_int
    lib.acrandGenerateUniformDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.acrandGenerateUniformDouble.restype = ctypes.c_int
    lib.acrandGenerateNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.acrandGenerateNormal.restype = ctypes.c_int
    lib.acrandGenerateNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.acrandGenerateNormalDouble.restype = ctypes.c_int
    lib.acrandGenerateLogNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.acrandGenerateLogNormal.restype = ctypes.c_int
    lib.acrandGenerateLogNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.acrandGenerateLogNormalDouble.restype = ctypes.c_int


def _require_dtype(output: torch.Tensor, dtype: torch.dtype, distribution: str) -> None:
    if output.dtype != dtype:
        raise TypeError(f"PPU acrand {distribution}: expected {dtype}, got {output.dtype}.")


def _require_even(output: torch.Tensor, distribution: str) -> None:
    if output.numel() % 2 != 0:
        raise ValueError(f"PPU acrand {distribution}: output length must be even.")


_STATUS_NAMES = {
    0: "ACRAND_STATUS_SUCCESS",
    100: "ACRAND_STATUS_VERSION_MISMATCH",
    101: "ACRAND_STATUS_NOT_INITIALIZED",
    102: "ACRAND_STATUS_ALLOCATION_FAILED",
    103: "ACRAND_STATUS_TYPE_ERROR",
    104: "ACRAND_STATUS_OUT_OF_RANGE",
    105: "ACRAND_STATUS_LENGTH_NOT_MULTIPLE",
    106: "ACRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
    201: "ACRAND_STATUS_LAUNCH_FAILURE",
    202: "ACRAND_STATUS_PREEXISTING_FAILURE",
    203: "ACRAND_STATUS_INITIALIZATION_FAILED",
    204: "ACRAND_STATUS_ARCH_MISMATCH",
    999: "ACRAND_STATUS_INTERNAL_ERROR",
}


def _check(status: int, function: str) -> None:
    if int(status) == 0:
        return
    name = _STATUS_NAMES.get(int(status), f"unknown status {int(status)}")
    raise PpuAcrandError(f"{function} failed with {name}.")
