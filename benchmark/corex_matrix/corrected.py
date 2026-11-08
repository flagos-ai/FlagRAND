"""Persistent CoreX SDK corrected references, distinct from stock host cuRAND.

Each API/parameter combination has an independent position from the same seed.
MTGP tails therefore cannot cross lambda values. As with the native provider,
the caller orders operations when moving a context between CUDA streams.
"""

import ctypes as C
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch

from benchmark.corex_matrix.corrected_build import RECORD_ENV, prepare_corrected_baseline

MRG_BASELINE = "corex_sdk_corrected_mrg"
POISSON_BASELINE = "corex_sdk_raw_alias_poisson"
APIS = {"raw32": 0, "uniform_f32": 1, "normal_f32": 2, "lognormal_f32": 3, "poisson": 4}


def library_path():
    return prepare_corrected_baseline()


def check(status, operation):
    if status:
        raise RuntimeError(f"CoreX corrected reference {operation} returned {status}")


@lru_cache(maxsize=None)
def load_library(path):
    library = C.CDLL(path)
    signatures = {
        "create": ([C.POINTER(C.c_void_p), C.c_int, C.c_int, C.c_uint64, C.c_float, C.c_float, C.c_double], C.c_int),
        "reset": ([C.c_void_p, C.c_uint64], C.c_int),
        "generate": ([C.c_void_p, C.c_void_p, C.c_uint64, C.c_void_p], C.c_int),
        "destroy": ([C.c_void_p], C.c_int),
        "state_words": ([C.c_void_p], C.c_uint64),
        "export_state": ([C.c_void_p, C.POINTER(C.c_uint32)], C.c_int),
        "header_version": ([], C.c_int),
        "host_mantissa_bits": ([], C.c_int),
        "table_metadata": ([C.c_void_p, C.POINTER(C.c_double)], C.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(library, f"corex_corrected_{name}")
        function.argtypes, function.restype = args, result
    return library


def library_version():
    return load_library(library_path()).corex_corrected_header_version()


class CorexCorrectedGenerator:
    def __init__(self, *, seed, offset=0, rng_type=121):
        if offset:
            raise ValueError("The corrected benchmark reference supports seed reset and same-parameter continuation")
        self.seed, self.rng_type = int(seed), int(rng_type)
        self.lib = load_library(library_path())
        self._handles = {}

    @staticmethod
    def _key(distribution, mean, stddev, lam):
        if distribution in {"normal_f32", "lognormal_f32"}:
            return distribution, float(mean), float(stddev), 0.0
        return distribution, 0.0, 1.0, float(lam) if distribution == "poisson" else 0.0

    def generate(self, distribution, output, *, mean=0.0, stddev=1.0, lam=10.0):
        if distribution != "poisson" and self.rng_type != 121:
            raise ValueError("This corrected reference provides MRG FP32 APIs and four-generator Poisson")
        key = self._key(distribution, mean, stddev, lam)
        handle = self._handles.get(key)
        if handle is None:
            handle = C.c_void_p()
            check(
                self.lib.corex_corrected_create(
                    C.byref(handle), self.rng_type, APIS[distribution], self.seed, *key[1:]
                ),
                "create",
            )
            self._handles[key] = handle
        stream = int(torch.cuda.current_stream(output.device).cuda_stream)
        check(self.lib.corex_corrected_generate(handle, output.data_ptr(), output.numel(), stream), "generate")

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
        for handle in self._handles.values():
            check(self.lib.corex_corrected_reset(handle, self.seed), "reset")

    def state_snapshot(self, distribution, *, mean=0.0, stddev=1.0, lam=10.0):
        handle = self._handles[self._key(distribution, mean, stddev, lam)]
        count = self.lib.corex_corrected_state_words(handle)
        storage = (C.c_uint32 * count)()
        check(self.lib.corex_corrected_export_state(handle, storage), "export_state")
        return bytes(storage)

    def metadata(self):
        path = library_path()
        record = os.environ.get(RECORD_ENV)
        tables = []
        for key, handle in self._handles.items():
            if key[0] == "poisson":
                values = (C.c_double * 5)()
                check(self.lib.corex_corrected_table_metadata(handle, values), "table_metadata")
                tables.append(dict(zip(("first", "last", "size", "retained_mass", "chernoff_tail_bound"), values)))
                tables[-1]["lambda"] = key[3]
        return {
            "baseline_kind": "CoreX SDK corrected reference",
            "stock_host_generate_called": False,
            "original_sdk_poisson_called": False,
            "rng_type": self.rng_type,
            "state_count": 200 if self.rng_type == 141 else 32768,
            "threads_per_block": 256,
            "parameter_state_scope": "Independent API/mean/stddev/lambda positions from the same seed; no cross-parameter independence claim",
            "mtgp_tail_policy": "Retain remaining 256-lane Poisson outputs only inside the same lambda context",
            "mrg": "Exact SDK host initialization, portable SDK integer-fold recurrence; explicit FP32 mapping",
            "poisson": "CPU long-double PMF + uint32 Walker alias; two raw calls per output",
            "host_long_double_mantissa_bits": self.lib.corex_corrected_host_mantissa_bits(),
            "poisson_tables": tables,
            "library": path,
            "library_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "build_record": record,
            "build_record_sha256": hashlib.sha256(Path(record).read_bytes()).hexdigest() if record else None,
            "version_scope": "Installed CURAND_VERSION header, not stock host-library implementation version",
        }

    def close(self):
        for handle in self._handles.values():
            check(self.lib.corex_corrected_destroy(handle), "destroy")
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
