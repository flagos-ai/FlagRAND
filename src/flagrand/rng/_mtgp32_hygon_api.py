"""Hygon MTGP32 selection for complete blocks and partial stream calls."""

import torch

from ._mtgp32_hygon import generate_aligned
from ._mtgp32_hygon_buffered import generate_buffered
from ._mtgp32_hygon_shared import generate_shared
from ._mtgp32_sequence import clear_chunk_cache


def generate(engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0):
    n = out.numel()
    buffer = getattr(engine, "_hygon_mtgp_buffer", None)
    cached = None if buffer is None else buffer.cached_copy
    if cached is not None:
        words = cached.try_copy(engine, out, n, api, mean, stddev, lam)
        if words:
            engine.offset += words
            if engine.offset == cached.end:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.cached_copy = None
            return out
    f64 = out.dtype == torch.float64
    paired = api in ("normal", "lognormal")
    if api == "raw32":
        words = n
    elif api == "poisson":
        words = 2 * n
    elif paired:
        words = ((n + 1) // 2) * (4 if f64 else 2)
    else:
        words = n * (2 if f64 else 1)
    if n < 65536 or engine.offset % 256 or words % 256 or getattr(engine, "_chunk_cache", None) is not None:
        return generate_buffered(engine, out, api, mean=mean, stddev=stddev, lam=lam)
    if n >= 1048576:
        return generate_shared(engine, out, api, mean=mean, stddev=stddev, lam=lam, fused=not paired, num_warps=4)
    return generate_aligned(engine, out, api, mean=mean, stddev=stddev, lam=lam, num_warps=4, linear_table=True)
