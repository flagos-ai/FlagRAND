from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import torch

CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161


class NvidiaCurandError(RuntimeError):
    pass


class NvidiaCurandGenerator:
    def __init__(
        self,
        *,
        seed: int,
        offset: int = 0,
        library_path: str | None = None,
    ) -> None:
        seed_value = int(seed)
        offset_value = int(offset)
        if not 0 <= seed_value <= (1 << 64) - 1:
            raise ValueError(f"NVIDIA cuRAND seed must be in [0, 2^64-1], got {seed!r}.")
        if offset_value < 0:
            raise ValueError(f"NVIDIA cuRAND offset must be non-negative, got {offset!r}.")

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
        self._stream: int | None = None
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

    def generate(
        self,
        distribution: str,
        output: torch.Tensor,
        *,
        mean: float = 0.0,
        stddev: float = 1.0,
    ) -> None:
        if output.device.type != "cuda" or not output.is_contiguous():
            raise ValueError("NVIDIA cuRAND: output must be a contiguous CUDA tensor.")
        if output.numel() == 0:
            return
        self._bind_current_stream()
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
            _require_even_dtype(output, torch.float32, distribution)
            status = self._lib.curandGenerateNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(mean),
                ctypes.c_float(stddev),
            )
            _check(status, "curandGenerateNormal")
        elif distribution == "normal_f64":
            _require_even_dtype(output, torch.float64, distribution)
            status = self._lib.curandGenerateNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(mean),
                ctypes.c_double(stddev),
            )
            _check(status, "curandGenerateNormalDouble")
        elif distribution == "lognormal_f32":
            _require_even_dtype(output, torch.float32, distribution)
            status = self._lib.curandGenerateLogNormal(
                self._handle,
                ptr,
                n,
                ctypes.c_float(mean),
                ctypes.c_float(stddev),
            )
            _check(status, "curandGenerateLogNormal")
        elif distribution == "lognormal_f64":
            _require_even_dtype(output, torch.float64, distribution)
            status = self._lib.curandGenerateLogNormalDouble(
                self._handle,
                ptr,
                n,
                ctypes.c_double(mean),
                ctypes.c_double(stddev),
            )
            _check(status, "curandGenerateLogNormalDouble")
        else:
            raise ValueError(f"NVIDIA cuRAND: unsupported distribution {distribution!r}.")

    def _bind_current_stream(self) -> None:
        stream = int(torch.cuda.current_stream().cuda_stream)
        if stream == self._stream:
            return
        _check(
            self._lib.curandSetStream(self._handle, ctypes.c_void_p(stream)),
            "curandSetStream",
        )
        self._stream = stream

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _check(self._lib.curandDestroyGenerator(self._handle), "curandDestroyGenerator")

    def __enter__(self) -> NvidiaCurandGenerator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def nvidia_curand_library_path() -> str:
    candidates = [
        os.environ.get("FLAGRAND_NVIDIA_CURAND_PATH"),
        os.environ.get("CURAND_LIBRARY_PATH"),
    ]
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidates.extend(
        (
            str(cuda_home / "lib64" / "libcurand.so"),
            str(cuda_home / "targets" / "x86_64-linux" / "lib" / "libcurand.so"),
            "/usr/local/cuda/lib64/libcurand.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libcurand.so",
        )
    )
    found = ctypes.util.find_library("curand")
    if found:
        candidates.append(found)
    for candidate in candidates:
        if candidate and (candidate == found or Path(candidate).exists()):
            return candidate
    raise FileNotFoundError("NVIDIA libcurand.so was not found. Set FLAGRAND_NVIDIA_CURAND_PATH or CUDA_HOME.")


def nvidia_curand_version(library_path: str | None = None) -> int:
    library = _load_curand(library_path)
    version = ctypes.c_int()
    _check(library.curandGetVersion(ctypes.byref(version)), "curandGetVersion")
    return int(version.value)


_CURAND_LIBRARY: ctypes.CDLL | None = None


def _load_curand(library_path: str | None = None) -> ctypes.CDLL:
    global _CURAND_LIBRARY
    if _CURAND_LIBRARY is not None and library_path is None:
        return _CURAND_LIBRARY
    path = library_path or nvidia_curand_library_path()
    library = ctypes.CDLL(path)
    _bind_signatures(library)
    if library_path is None:
        _CURAND_LIBRARY = library
    return library


def _bind_signatures(library: ctypes.CDLL) -> None:
    library.curandGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.curandGetVersion.restype = ctypes.c_int
    library.curandCreateGenerator.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    library.curandCreateGenerator.restype = ctypes.c_int
    library.curandDestroyGenerator.argtypes = [ctypes.c_void_p]
    library.curandDestroyGenerator.restype = ctypes.c_int
    library.curandSetPseudoRandomGeneratorSeed.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
    library.curandSetPseudoRandomGeneratorSeed.restype = ctypes.c_int
    library.curandSetGeneratorOffset.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
    library.curandSetGeneratorOffset.restype = ctypes.c_int
    library.curandSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.curandSetStream.restype = ctypes.c_int
    library.curandGenerate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.curandGenerate.restype = ctypes.c_int
    library.curandGenerateUniform.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.curandGenerateUniform.restype = ctypes.c_int
    library.curandGenerateUniformDouble.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    library.curandGenerateUniformDouble.restype = ctypes.c_int
    library.curandGenerateNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    library.curandGenerateNormal.restype = ctypes.c_int
    library.curandGenerateNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    library.curandGenerateNormalDouble.restype = ctypes.c_int
    library.curandGenerateLogNormal.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
    ]
    library.curandGenerateLogNormal.restype = ctypes.c_int
    library.curandGenerateLogNormalDouble.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    library.curandGenerateLogNormalDouble.restype = ctypes.c_int


def _require_dtype(output: torch.Tensor, dtype: torch.dtype, distribution: str) -> None:
    if output.dtype != dtype:
        raise TypeError(f"NVIDIA cuRAND {distribution}: expected {dtype}, got {output.dtype}.")


def _require_even_dtype(output: torch.Tensor, dtype: torch.dtype, distribution: str) -> None:
    _require_dtype(output, dtype, distribution)
    if output.numel() % 2 != 0:
        raise ValueError(f"NVIDIA cuRAND {distribution}: output length must be even.")


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
    raise NvidiaCurandError(f"{function} failed with {name}.")
