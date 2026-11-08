"""Copy cached MTGP32 output on its current Ascend stream.

Contiguous views use their actual data pointers, as in Torch NPU's native
device copy path. The source contains values from the MTGP32 Triton kernels.
"""

import ctypes
import os
import struct
from functools import lru_cache
from pathlib import Path

import torch
import torch_npu


@lru_cache(maxsize=1)
def copy_function():
    root = Path(
        os.environ.get("FLAGRAND_ASCEND_CANN_ROOT")
        or os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    library = ctypes.CDLL(str(root / "lib64/libascendcl.so"))
    function = library.aclrtMemcpyAsync
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return library, function


class CachedCopy:
    def __init__(self, engine, raw, source, api, out_dtype, stride, units, residue, parameters, stream):
        self.library, self.copy = copy_function()
        self.raw, self.source = raw, source
        self.seed, self.api, self.dtype = engine.seed, api, out_dtype
        self.device, self.device_index = source.device, source.device.index
        self.start = engine._chunk_cache_start
        self.end = self.start + raw.numel()
        self.stride, self.units, self.residue = stride, units, residue
        self.paired, self.parameters = api in ("normal", "lognormal"), parameters
        self.pointer, self.itemsize = source.data_ptr(), source.element_size()
        self.stream = stream
        # The non-NoWait reader drains Torch NPU's host task queue before direct
        # ACL submission. This is not a device synchronization and is timed.
        self.read_stream = torch_npu._C._npu_getCurrentRawStream
        self.handle = self.read_stream(self.device_index)
        raw.record_stream(stream)
        source.record_stream(stream)

    def try_copy(self, engine, out, n, api, mean, stddev):
        if (
            engine._chunk_cache is not self.raw
            or engine.seed != self.seed
            or api != self.api
            or out.dtype != self.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            return 0
        words = self.stride * ((n + 1) // 2 if self.paired else n)
        offset = engine.offset - self.start
        if offset < 0 or engine.offset + words > self.end or offset % self.stride != self.residue:
            return 0
        if self.paired and struct.pack("<dd", float(mean), float(stddev)) != self.parameters:
            return 0
        handle = self.read_stream(self.device_index)
        if handle != self.handle:
            return 0
        first = self.units * ((offset - self.residue) // self.stride)
        source_ptr = self.pointer + first * self.itemsize
        destination = out.data_ptr()
        count = n * self.itemsize
        status = self.copy(destination, count, source_ptr, count, 3, handle)
        if status:
            raise RuntimeError(f"aclrtMemcpyAsync returned {status}")
        out.record_stream(self.stream)
        return words
