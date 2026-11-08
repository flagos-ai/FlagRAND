"""Fused Poisson sampling with signed 32-bit arithmetic on Ascend."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flagrand.rng._poisson_table import poisson_alias_table
from flagrand.rng._xorwow.ascend_kernels import _next_i32
from flagrand.rng.mtgp32 import Mtgp32Generator
from flagrand.rng.philox import PhiloxGenerator
from flagrand.rng.xorwow import XorwowGenerator
from flagrand.runtime.compiled_launcher import CachedKernelLauncher


@triton.jit
def _unsigned_less(a, b):
    return (a ^ -2147483648) < (b ^ -2147483648)


@triton.jit
def _multiply_words(a, B: tl.constexpr):
    # Four 16-bit partial products retain the exact 64-bit product without
    # the unsupported uint32-to-uint64 vector cast in this Ascend compiler.
    a0 = a & 65535
    a1 = (a >> 16) & 65535
    w0 = a0 * (B & 65535)
    t = a1 * (B & 65535) + ((w0 >> 16) & 65535)
    w1 = a0 * (B >> 16) + (t & 65535)
    high = a1 * (B >> 16) + ((t >> 16) & 65535) + ((w1 >> 16) & 65535)
    low = (w1 << 16) | (w0 & 65535)
    return high.to(tl.int32), low.to(tl.int32)


@triton.jit
def _philox_words(seed_lo, seed_hi, counter_lo, counter_hi):
    a = tl.full(counter_lo.shape, 0, tl.int32)
    b = tl.full(counter_lo.shape, 0, tl.int32)
    c, d = counter_lo, counter_hi
    k0, k1 = seed_lo, seed_hi
    for _ in tl.static_range(10):
        h0, l0 = _multiply_words(a, 0xD2511F53)
        h1, l1 = _multiply_words(c, 0xCD9E8D57)
        a, b, c, d = h1 ^ b ^ k0, l1, h0 ^ d ^ k1, l0
        k0 = (k0 + -1640531527).to(tl.int32)
        k1 = (k1 + -1150833019).to(tl.int32)
    return a, b, c, d


@triton.jit
def _alias(a, b, threshold, aliases, FIRST: tl.constexpr, SIZE: tl.constexpr):
    column = a & (SIZE - 1)
    cutoff = tl.load(threshold + column)
    other = tl.load(aliases + column)
    return FIRST + tl.where(_unsigned_less(b, cutoff), column, other)


@triton.jit(do_not_specialize=["seed_lo", "seed_hi", "counter_lo", "counter_hi"])
def _philox_poisson(
    out,
    threshold,
    aliases,
    seed_lo,
    seed_hi,
    counter_lo,
    counter_hi,
    n,
    SHIFT: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = (counter_lo + i).to(tl.int32)
    high = counter_hi + _unsigned_less(low, counter_lo).to(tl.int32)
    a, b, c, d = _philox_words(seed_lo, seed_hi, low, high)
    if SHIFT != 0:
        low2 = (low + 1).to(tl.int32)
        high2 = high + (low2 == 0).to(tl.int32)
        e, f, g, h = _philox_words(seed_lo, seed_hi, low2, high2)
        if SHIFT == 1:
            a, b, c, d = b, c, d, e
        elif SHIFT == 2:
            a, b, c, d = c, d, e, f
        else:
            a, b, c, d = d, e, f, g
    x = _alias(a, b, threshold, aliases, FIRST, SIZE)
    y = _alias(c, d, threshold, aliases, FIRST, SIZE)
    tl.store(out + 2 * i, x, 2 * i < n)
    tl.store(out + 2 * i + 1, y, 2 * i + 1 < n)


@triton.jit
def _xorwow_poisson(
    state, out, threshold, aliases, n, S: tl.constexpr, FIRST: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    present = (p < S // 2) & (p < n)
    lane = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    lane_mask = (lane < S) & (lane < 2 * n)
    x0, x1 = tl.split(tl.reshape(tl.load(state + 0 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    y0, y1 = tl.split(tl.reshape(tl.load(state + 1 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    z0, z1 = tl.split(tl.reshape(tl.load(state + 2 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    w0, w1 = tl.split(tl.reshape(tl.load(state + 3 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    v0, v1 = tl.split(tl.reshape(tl.load(state + 4 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    d0, d1 = tl.split(tl.reshape(tl.load(state + 5 * S + lane, lane_mask, other=0), (BLOCK, 2)))
    for batch in range(tl.cdiv(n, S // 2)):
        i = batch * (S // 2) + p
        active = present & (i < n)
        a, nx0, ny0, nz0, nw0, nv0, nd0 = _next_i32(x0, y0, z0, w0, v0, d0)
        b, nx1, ny1, nz1, nw1, nv1, nd1 = _next_i32(x1, y1, z1, w1, v1, d1)
        value = _alias(a, b, threshold, aliases, FIRST, SIZE)
        tl.store(out + i, value, active)
        x0, y0, z0 = tl.where(active, nx0, x0), tl.where(active, ny0, y0), tl.where(active, nz0, z0)
        w0, v0, d0 = tl.where(active, nw0, w0), tl.where(active, nv0, v0), tl.where(active, nd0, d0)
        x1, y1, z1 = tl.where(active, nx1, x1), tl.where(active, ny1, y1), tl.where(active, nz1, z1)
        w1, v1, d1 = tl.where(active, nw1, w1), tl.where(active, nv1, v1), tl.where(active, nd1, d1)
    tl.store(state + 0 * S + lane, tl.interleave(x0, x1), lane_mask)
    tl.store(state + 1 * S + lane, tl.interleave(y0, y1), lane_mask)
    tl.store(state + 2 * S + lane, tl.interleave(z0, z1), lane_mask)
    tl.store(state + 3 * S + lane, tl.interleave(w0, w1), lane_mask)
    tl.store(state + 4 * S + lane, tl.interleave(v0, v1), lane_mask)
    tl.store(state + 5 * S + lane, tl.interleave(d0, d1), lane_mask)


def _signed(value):
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


@lru_cache(maxsize=32)
def _tables(lam, device, raw_range):
    cutoff, alias, first = poisson_alias_table(lam, raw_range)
    return (
        torch.tensor([_signed(x) for x in cutoff], dtype=torch.int32, device=device),
        torch.tensor(alias, dtype=torch.int32, device=device),
        first,
    )


@triton.jit
def _raw_poisson(RAW, OUT, THRESHOLD, ALIASES, N, FIRST: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(RAW + 2 * i, i < N, other=0)
    b = tl.load(RAW + 2 * i + 1, i < N, other=0)
    tl.store(OUT + i, _alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE), i < N)


_philox_launch = CachedKernelLauncher(
    _philox_poisson,
    constexpr_names=("SHIFT", "FIRST", "SIZE", "BLOCK"),
    option_names=("num_warps", "sync_solver"),
)
_xorwow_launch = CachedKernelLauncher(
    _xorwow_poisson,
    constexpr_names=("S", "FIRST", "SIZE", "BLOCK"),
    option_names=("num_warps", "sync_solver"),
)


def generate_poisson(engine, out, lam, block_size=None, num_warps=None):
    if (
        isinstance(engine, Mtgp32Generator)
        and block_size is None
        and num_warps is None
        and getattr(engine, "_use_buffered", True)
    ):
        from flagrand.rng._mtgp32_ascend_poisson import generate

        return generate(engine, out, lam)
    n = out.numel()
    if not n:
        return out
    threshold, aliases, first = _tables(float(lam), out.device, getattr(engine, "raw_range", 1 << 32))
    size = threshold.numel()
    block = 256 if block_size is None else block_size
    warps = 4 if num_warps is None else num_warps
    if isinstance(engine, PhiloxGenerator):
        counter = engine.offset // 4
        constants = (engine.offset % 4, first, size, block)
        _philox_launch.launch(
            (triton.cdiv(n, 2 * block),),
            (
                out,
                threshold,
                aliases,
                _signed(engine.seed),
                _signed(engine.seed >> 32),
                _signed(counter),
                _signed(counter >> 32),
                n,
            ),
            constants,
            (warps, False),
            specialization_key=(out.device, out.dtype, out.data_ptr() % 16, n, constants, warps),
        )
        engine.offset += 2 * n
    elif isinstance(engine, XorwowGenerator) and engine.nstreams % 2 == 0:
        state = engine.ensure_state(out.device)
        constants = (engine.nstreams, first, size, block)
        _xorwow_launch.launch(
            (triton.cdiv(min(n, engine.nstreams // 2), block),),
            (state, out, threshold, aliases, n),
            constants,
            (warps, False),
            specialization_key=(out.device, out.dtype, out.data_ptr() % 16, n, constants, warps),
        )
        engine.offset += 2 * n
        engine._state_offset = engine.offset
    else:
        raw = torch.empty(2 * n, dtype=torch.int32, device=out.device)
        engine.generate(raw)
        _raw_poisson[(triton.cdiv(n, block),)](raw, out, threshold, aliases, n, first, size, block)
    return out
