"""PPU SDK MTGP32 device functions with persistent host-owned states.

Each API/parameter combination maintains its own position. Equal seeds start
those states from the same SDK initial values; cross-API statistical independence
is not claimed. Tail values are retained, so splitting a fixed-parameter call
preserves the sequence. The caller orders accesses when changing CUDA streams.
"""

from __future__ import annotations

import ctypes as C
import hashlib
from functools import lru_cache
from pathlib import Path

import torch

from benchmark.ppu_matrix.build import prepare_mtgp32_baseline

APIS = {
    "raw32": 0,
    "uniform_f32": 1,
    "uniform_f64": 2,
    "normal_f32": 3,
    "normal_f64": 4,
    "lognormal_f32": 5,
    "lognormal_f64": 6,
    "poisson": 7,
}
BASELINE_KIND = "ppu_sdk_mtgp32_device"
STATE_COUNT = 200
THREAD_COUNT = 256


def mtgp32_device_library_path():
    return prepare_mtgp32_baseline()


def mtgp32_device_version():
    """Installed acRAND component version; not a version of the device headers."""
    from benchmark.ppu_matrix.baseline import acrand_version

    return acrand_version()


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation} returned SDK status {status}")


@lru_cache(maxsize=None)
def load_library(path):
    library = C.CDLL(path)
    signatures = {
        "mtgp_device_create": [C.POINTER(C.c_void_p), C.c_int, C.c_uint64, C.c_double, C.c_double, C.c_double],
        "mtgp_device_generate": [C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p],
        "mtgp_device_reset": [C.c_void_p, C.c_uint64],
        "mtgp_device_destroy": [C.c_void_p],
        "mtgp_device_export_state": [C.c_void_p, C.POINTER(C.c_uint32)],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = C.c_int
    return library


class Mtgp32DeviceGenerator:
    """Same-parameter continuation using SDK distribution functions unchanged."""

    baseline_kind = BASELINE_KIND

    def __init__(self, *, seed, offset=0, rng_type=141):
        if rng_type != 141 or offset != 0:
            raise ValueError("This device baseline implements MTGP32 with seed reset and no absolute offset API")
        self.seed = int(seed)
        self.lib = load_library(mtgp32_device_library_path())
        self._handles = {}

    @staticmethod
    def _key(distribution, mean, stddev, lam):
        if distribution.startswith(("normal", "lognormal")):
            return distribution, float(mean), float(stddev), 0.0
        if distribution == "poisson":
            return distribution, 0.0, 1.0, float(lam)
        return distribution, 0.0, 1.0, 0.0

    def generate(self, distribution, output, *, mean=0.0, stddev=1.0, lam=10.0):
        key = self._key(distribution, mean, stddev, lam)
        handle = self._handles.get(key)
        if handle is None:
            handle = C.c_void_p()
            check(
                self.lib.mtgp_device_create(C.byref(handle), APIS[distribution], self.seed, *key[1:]),
                "mtgp_device_create",
            )
            self._handles[key] = handle
        stream = int(torch.cuda.current_stream(output.device).cuda_stream)
        check(
            self.lib.mtgp_device_generate(handle, output.data_ptr(), output.numel(), stream),
            "mtgp_device_generate",
        )

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        for handle in self._handles.values():
            check(self.lib.mtgp_device_reset(handle, self.seed), "mtgp_device_reset")

    def state_snapshot(self, distribution, *, mean=0.0, stddev=1.0, lam=10.0):
        """Copy logical state for untimed correctness comparisons."""
        key = self._key(distribution, mean, stddev, lam)
        storage = (C.c_uint32 * (STATE_COUNT * 1026 + 1))()
        check(self.lib.mtgp_device_export_state(self._handles[key], storage), "mtgp_device_export_state")
        return bytes(storage)

    def metadata(self):
        return {
            "baseline_kind": BASELINE_KIND,
            "host_generate_api_called": False,
            "states": STATE_COUNT,
            "threads_per_block": THREAD_COUNT,
            "sequence_order": "200 states x 256 lanes, repeated in fixed order",
            "parameter_state_scope": "API and distribution parameters maintain separate positions from the same seed",
            "tail_policy": "retain unused values from each complete 256-lane SDK call",
            "fp64_random_input_bits": 32,
            "poisson_large_lambda": "unchanged SDK rounded-normal approximation for lambda >= 1000",
            "library_path": mtgp32_device_library_path(),
            "library_sha256": hashlib.sha256(Path(mtgp32_device_library_path()).read_bytes()).hexdigest(),
            "version_scope": "library_version records installed acRAND; device headers are identified by build.json SHA256",
        }

    def close(self):
        for handle in self._handles.values():
            check(self.lib.mtgp_device_destroy(handle), "mtgp_device_destroy")
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
