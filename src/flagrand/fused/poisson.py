"""Cached alias-table Poisson sampling; Philox generation is fused.

The two-pass path consumes two raw values per output and therefore also works
with collective generators such as MTGP32. Table construction is outside warm
steady-state timing, as in a reusable native host generator.
"""

import torch
import triton
import triton.language as tl

from flagrand.rng._poisson_table import poisson_alias_table
from flagrand.rng.mrg32k3a import Mrg32k3aGenerator
from flagrand.rng.mtgp32 import Mtgp32Generator
from flagrand.rng.philox import PhiloxGenerator, _philox_window4
from flagrand.rng.xorwow import XorwowGenerator
from flagrand.runtime.backend import get_vendor_info

_DEVICE_TABLES = {}


@triton.jit
def _sample_alias(r0, r1, threshold, aliases, FIRST: tl.constexpr, SIZE: tl.constexpr, RAW_RANGE: tl.constexpr):
    if RAW_RANGE == 4294967296:
        column = r0.to(tl.uint32) & (SIZE - 1)
        coin = r1.to(tl.uint32)
    else:
        # MRG32K3A returns [1, m1], rather than the full uint32 range.
        column = (((r0.to(tl.uint64) - 1) * SIZE) // RAW_RANGE).to(tl.int32)
        coin = r1.to(tl.uint32) - 1
    cutoff = tl.load(threshold + column).to(tl.uint32)
    other = tl.load(aliases + column)
    return FIRST + tl.where(coin < cutoff, column, other)


@triton.jit
def _philox_poisson(
    out,
    threshold,
    aliases,
    seed,
    counter,
    n,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a, b, c, d = _philox_window4(seed, tl.cast(counter, tl.uint64) + i.to(tl.uint64), SHIFT)
    x = _sample_alias(a, b, threshold, aliases, FIRST, SIZE, 4294967296)
    y = _sample_alias(c, d, threshold, aliases, FIRST, SIZE, 4294967296)
    tl.store(out + 2 * i, x, 2 * i < n)
    tl.store(out + 2 * i + 1, y, 2 * i + 1 < n)


@triton.jit
def _raw_poisson(
    raw,
    out,
    threshold,
    aliases,
    n,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    RAW_RANGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(raw + 2 * i, i < n, other=1).to(tl.uint32)
    b = tl.load(raw + 2 * i + 1, i < n, other=1).to(tl.uint32)
    value = _sample_alias(a, b, threshold, aliases, FIRST, SIZE, RAW_RANGE)
    tl.store(out + i, value, i < n)


def generate_poisson(engine, out, lam: float, *, block_size=None, num_warps=None, fused_mtgp32=True):
    requested_warps = num_warps
    requested_block_size = block_size
    num_warps = 4 if num_warps is None else num_warps
    n = out.numel()
    if not n:
        return out
    if block_size is None:
        # At 16 Mi outputs on PPU, one element per thread removes the shared
        # state-writeback layout conversion seen with BLOCK=256 and 4 warps.
        block_size = (
            128
            if isinstance(engine, XorwowGenerator)
            and n >= 16777216
            and num_warps == 4
            and get_vendor_info().vendor_name == "thead"
            else 256
        )
    raw_range = getattr(engine, "raw_range", 1 << 32)
    key = (float(lam), raw_range, str(out.device))
    if key not in _DEVICE_TABLES:
        cutoff, alias, first = poisson_alias_table(float(lam), raw_range)
        # int32 storage is portable to backends without torch.uint32 kernels.
        signed = [x if x < (1 << 31) else x - (1 << 32) for x in cutoff]
        _DEVICE_TABLES[key] = (
            torch.tensor(signed, dtype=torch.int32, device=out.device),
            torch.tensor(alias, dtype=torch.int32, device=out.device),
            first,
        )
    threshold, aliases, first = _DEVICE_TABLES[key]
    size = threshold.numel()
    if isinstance(engine, PhiloxGenerator):
        _philox_poisson[(triton.cdiv(n, 2 * block_size),)](
            out,
            threshold,
            aliases,
            engine.seed,
            engine.offset // 4,
            n,
            engine.offset % 4,
            first,
            size,
            block_size,
            num_warps=num_warps,
        )
        engine.offset += 2 * n
    elif isinstance(engine, XorwowGenerator) and engine.nstreams % 2 == 0:
        from flagrand.fused._internal.xorwow_poisson import launch_xorwow_poisson

        launch_xorwow_poisson(engine, out, threshold, aliases, first, size, block_size=block_size, num_warps=num_warps)
        engine.offset += 2 * n
        engine._state_offset = engine.offset
    elif isinstance(engine, Mrg32k3aGenerator) and engine.nstreams % 2 == 0 and n <= 262144:
        # Fusion removes a launch for short/medium calls. The C550 comparison
        # found the separate raw kernel faster at 1M/4M outputs, so retain it
        # above the measured 262144-output crossover region (32768 streams).
        from flagrand.fused._internal.mrg32k3a_poisson import launch_mrg32k3a_poisson

        launch_mrg32k3a_poisson(
            engine, out, threshold, aliases, first, size, block_size=block_size // 2, num_warps=num_warps
        )
        engine.offset += 2 * n
        engine._state_offset = engine.offset
        engine._cursor = (engine._cursor + 2 * n) % engine.nstreams
    elif (
        isinstance(engine, Mtgp32Generator)
        and fused_mtgp32
        and requested_warps is None
        and requested_block_size is None
        and out.dtype == torch.int32
        and engine._uses_thead_buffer(n)
    ):
        from flagrand.rng._mtgp32_thead_buffered import generate_buffered

        generate_buffered(engine, out, "poisson", threshold=threshold, aliases=aliases, first=first, size=size)
    elif (
        isinstance(engine, Mtgp32Generator)
        and fused_mtgp32
        and requested_warps is None
        and requested_block_size is None
        and out.dtype == torch.int32
        and n >= 128
        and n % 128 == 0
        and engine.offset % 256 == 0
        and getattr(engine, "_chunk_cache", None) is None
        and get_vendor_info().vendor_name == "thead"
    ):
        from flagrand.rng._mtgp32_thead_poisson import generate_poisson_aligned

        engine._prepare_thead_stream(out.device)
        # The PPU 256/512-entry table comparison selects the load predicates.
        generate_poisson_aligned(engine, out, threshold, aliases, first, size, mask_alias=size >= 512)
    elif (
        isinstance(engine, Mtgp32Generator)
        and fused_mtgp32
        and requested_warps is None
        and block_size == 256
        and (
            engine._uses_raw_buffer(2 * n)
            or (n <= 65536 and getattr(engine, "_use_buffered", True) and getattr(engine, "_buffered_backend", False))
        )
    ):
        from flagrand.rng._mtgp32_buffered import generate_buffered

        generate_buffered(engine, out, "poisson", threshold=threshold, aliases=aliases, first=first, size=size)
    elif (
        isinstance(engine, Mtgp32Generator)
        and fused_mtgp32
        and n % 128 == 0
        and engine.offset % 256 == 0
        and getattr(engine, "_chunk_cache", None) is None
        and get_vendor_info().vendor_name == "metax"
    ):
        from flagrand.rng._mtgp32_direct_pairs import generate_direct_pairs

        generate_direct_pairs(
            engine,
            out,
            "poisson",
            threshold=threshold,
            aliases=aliases,
            first=first,
            size=size,
            num_warps=2 if requested_warps is None else num_warps,
        )
    else:
        raw = torch.empty(2 * n, dtype=torch.int32, device=out.device)
        engine.generate(raw)
        _raw_poisson[(triton.cdiv(n, block_size),)](
            raw,
            out,
            threshold,
            aliases,
            n,
            first,
            size,
            raw_range,
            block_size,
            num_warps=num_warps,
        )
    return out
