from __future__ import annotations

import os
import ctypes
from pathlib import Path

import torch

from benchmark.ppu_matrix.curand import PpuCurandGenerator


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
        generator=wrapper._generator
        version=ctypes.c_int()
        generator._lib.curandGetVersion.argtypes=[ctypes.POINTER(ctypes.c_int)]
        version_status=int(generator._lib.curandGetVersion(ctypes.byref(version)))
        result['version']={'status':version_status,'value':version.value}
        generator._set_current_stream()
        output=torch.empty(2,device='cuda',dtype=torch.float64)
        common=(generator._handle,ctypes.c_void_p(output.data_ptr()),ctypes.c_size_t(2))
        for api,function,args in (
            ('uniform_f64','curandGenerateUniformDouble',common),
            ('normal_f64','curandGenerateNormalDouble',common+(ctypes.c_double(0),ctypes.c_double(1))),
            ('lognormal_f64','curandGenerateLogNormalDouble',common+(ctypes.c_double(0),ctypes.c_double(1)))):
            status=int(getattr(generator._lib,function)(*args))
            result[api]={'function':function,'return_code':status,
                'status':'pass' if status==0 else 'unsupported' if status==106 else 'error'}
        torch.cuda.synchronize()
    return result
