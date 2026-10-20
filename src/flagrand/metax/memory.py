"""Stream-ordered device copies through the installed MACA runtime."""

import ctypes
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def device_memcpy_async():
    """Return mcMemcpyAsync with the ABI declared by mc_runtime_api.h."""
    library = ctypes.CDLL(os.path.join(os.environ.get("MACA_PATH", "/opt/maca"), "lib/libmcruntime.so"))
    function = library.mcMemcpyAsync
    function.restype = ctypes.c_int
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    return function
