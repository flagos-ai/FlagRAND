"""Copy cached MTGP outputs through the installed HIP runtime."""

import ctypes
import os
import struct
from functools import lru_cache
from pathlib import Path

from triton.runtime.driver import driver


@lru_cache(maxsize=1)
def _copy_function():
    library = ctypes.CDLL(str(Path(os.environ.get("DTK_HOME", "/opt/dtk")) / "lib/libamdhip64.so"))
    function = library.hipMemcpyAsync
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return library, function


class CachedCopy:
    """One raw window and one transformation; resolve each actual output."""

    def __init__(self, engine, raw, source, api, stride, residue, units, parameters, lam, stream, output_dtype):
        self.library, self.copy = _copy_function()
        self.raw = raw
        self.source = source
        self.seed = engine.seed
        self.api = api
        self.dtype = output_dtype
        self.device = source.device
        self.device_index = source.device.index
        self.start = engine._chunk_cache_start
        self.end = self.start + raw.numel()
        self.stride, self.residue, self.units = stride, residue, units
        self.parameters, self.lam = parameters, float(lam)
        self.paired = api in ("normal", "lognormal")
        self.format = "<d" if source.element_size() == 8 else "<f"
        self.pointer, self.itemsize = source.data_ptr(), source.element_size()
        self.stream, self.handle = stream, stream.cuda_stream
        self.read_stream = driver.active.get_current_stream

    def try_copy(self, engine, out, n, api, mean, stddev, lam):
        if (
            engine._chunk_cache is not self.raw
            or engine.seed != self.seed
            or api != self.api
            or out.dtype != self.dtype
            or out.device != self.device
        ):
            return 0
        words = self.stride * ((n + 1) // 2 if self.paired else n)
        offset = engine.offset - self.start
        if offset < 0 or engine.offset + words > self.end or offset % self.stride != self.residue:
            return 0
        if self.paired:
            if (struct.pack(self.format, mean), struct.pack(self.format, stddev)) != self.parameters:
                return 0
        elif api == "poisson" and float(lam) != self.lam:
            return 0
        if self.read_stream(self.device_index) != self.handle:
            return 0
        first = self.units * ((offset - self.residue) // self.stride)
        status = self.copy(out.data_ptr(), self.pointer + first * self.itemsize, n * self.itemsize, 3, self.handle)
        if status:
            raise RuntimeError(f"hipMemcpyAsync returned {status}")
        out.record_stream(self.stream)
        return words
