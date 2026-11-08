"""MTGP32 recurrence and distribution transforms for Ascend signed-i32 SIMD."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._xorwow._binding import bind_kernel
from ._xorwow.api import _float_bits, _soft_parameters
from ._xorwow.native_math import sincos_turns
from ._xorwow.packed64 import uniform64_words


@triton.jit
def _ring_load(STATE, s_base, lane, base):
    boundary = 1024 - base
    first = tl.load(STATE + s_base + base + lane, lane < boundary, other=0)
    second = tl.load(STATE + s_base + base - 1024 + lane, lane >= boundary, other=0)
    return tl.where(lane < boundary, first, second)


@triton.jit
def _ring_store(STATE, s_base, lane, base, value):
    boundary = 1024 - base
    tl.store(STATE + s_base + base + lane, value, lane < boundary)
    tl.store(STATE + s_base + base - 1024 + lane, value, lane >= boundary)


@triton.jit
def _linear_table(TABLE, base, index):
    p0 = tl.load(TABLE + base + 1)
    p1 = tl.load(TABLE + base + 2)
    p2 = tl.load(TABLE + base + 4)
    p3 = tl.load(TABLE + base + 8)
    return (
        tl.where((index & 1) != 0, p0, 0)
        ^ tl.where((index & 2) != 0, p1, 0)
        ^ tl.where((index & 4) != 0, p2, 0)
        ^ tl.where((index & 8) != 0, p3, 0)
    )


@triton.jit
def _step(STATE, PARAM, TEMPER, s_base, p_base, lane, offset, pos, sh1, sh2):
    # offset is 0, 256, 512 or 768, so this first 256-word range cannot wrap.
    a = tl.load(STATE + s_base + offset + lane)
    b = _ring_load(STATE, s_base, lane, (offset + 1) & 1023)
    y = _ring_load(STATE, s_base, lane, (offset + pos) & 1023)
    x = (a & -524288) ^ b
    x = x ^ (x << sh1)
    y = x ^ ((y >> sh2) & (2147483647 >> (sh2 - 1)))
    value = y ^ _linear_table(PARAM, p_base, y)
    t = _ring_load(STATE, s_base, lane, (offset + pos - 1) & 1023)
    t = t ^ ((t >> 16) & 65535)
    t = t ^ ((t >> 8) & 16777215)
    output = value ^ _linear_table(TEMPER, p_base, t)
    _ring_store(STATE, s_base, lane, (offset + 351) & 1023, value)
    return output


@triton.jit
def contiguous_kernel(OUT, STATE, POS, SH1, SH2, PARAM, TEMPER, START, COUNT, STATES: tl.constexpr):
    p = tl.program_id(0)
    first = START % STATES
    state_id = (first + p) % STATES
    first_round = (START // STATES) % 4 + (first + p) // STATES
    pos, sh1, sh2 = tl.load(POS + state_id), tl.load(SH1 + state_id), tl.load(SH2 + state_id)
    lane = tl.arange(0, 256)
    for k in range(tl.cdiv(COUNT - p, STATES)):
        offset = ((first_round + k) * 256) & 1023
        output = _step(STATE, PARAM, TEMPER, state_id * 1024, state_id * 16, lane, offset, pos, sh1, sh2)
        tl.store(OUT + (k * STATES + p) * 256 + lane, output)
        tl.debug_barrier()


@triton.jit
def blocks_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    N,
    CHUNKS: tl.constexpr,
    START: tl.constexpr,
    STATES: tl.constexpr,
    MODE: tl.constexpr,
):
    p = tl.program_id(0)
    lane = tl.arange(0, 256)
    pos, sh1, sh2 = tl.load(POS + p), tl.load(SH1 + p), tl.load(SH2 + p)
    for k in range(CHUNKS):
        offset = ((START + k) * 256) & 1023
        output = _step(STATE, PARAM, TEMPER, p * 1024, p * 16, lane, offset, pos, sh1, sh2)
        index = (k * STATES + p) * 256 + lane
        if MODE == 1:
            value = (((output >> 8) & 16777215).to(tl.float32) + 1.0) * 5.960464477539063e-8
            tl.store(OUT + index, value, index < N)
        else:
            tl.store(OUT + index, output, index < N)
        tl.debug_barrier()


@triton.jit
def _uniform(RAW, OUT, N, F64: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if F64:
        a = tl.load(RAW + 2 * i, i < N, other=0)
        b = tl.load(RAW + 2 * i + 1, i < N, other=0)
        lo, hi = uniform64_words(a, b)
        tl.store(OUT + 2 * i, lo, i < N)
        tl.store(OUT + 2 * i + 1, hi, i < N)
    else:
        a = tl.load(RAW + i, i < N, other=0)
        u = (((a >> 8) & 16777215).to(tl.float32) + 1.0) * 5.960464477539063e-8
        tl.store(OUT + i, u, i < N)


@triton.jit
def _normal32(RAW, OUT, N, MU: tl.constexpr, SIGMA: tl.constexpr, LOG: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(RAW + 2 * i, 2 * i < N, other=0)
    b = tl.load(RAW + 2 * i + 1, 2 * i < N, other=0)
    u = (((a >> 8) & 16777215).to(tl.float32) + 1.0) * 5.960464477539063e-8
    v = ((b >> 8) & 16777215).to(tl.float32) * 5.960464477539063e-8
    radius = tl.sqrt(-2.0 * tl.log(u))
    sine, cosine = sincos_turns(v, False)
    x, y = MU + SIGMA * radius * cosine, MU + SIGMA * radius * sine
    if LOG:
        x, y = tl.exp(x), tl.exp(y)
    tl.store(OUT + 2 * i, x, 2 * i < N)
    tl.store(OUT + 2 * i + 1, y, 2 * i + 1 < N)


@lru_cache(maxsize=1)
def _normal64_kernel():
    from ._xorwow.ascend_soft64_math import bind_ascend_soft64

    functions = bind_ascend_soft64(runtime_loops=True, direct_product=True)
    standard_pair = functions.standard_pair_words
    affine = functions.affine_pair_words

    def normal64(RAW, OUT, PARAMS, N, LOG: tl.constexpr, STANDARD: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(RAW + 4 * i, 2 * i < N, other=0)
        b = tl.load(RAW + 4 * i + 1, 2 * i < N, other=0)
        c = tl.load(RAW + 4 * i + 2, 2 * i < N, other=0)
        d = tl.load(RAW + 4 * i + 3, 2 * i < N, other=0)
        if STANDARD:
            l0, h0, l1, h1 = standard_pair(a, b, c, d, 3 if LOG else 2, INTEGER_QUADRANT_SIGN=True)
        else:
            mu0 = tl.load(PARAMS).to(tl.float32, bitcast=True)
            mu1 = tl.load(PARAMS + 1).to(tl.float32, bitcast=True)
            mu2 = tl.load(PARAMS + 2).to(tl.float32, bitcast=True)
            mue = tl.load(PARAMS + 3)
            s0 = tl.load(PARAMS + 4).to(tl.float32, bitcast=True)
            s1 = tl.load(PARAMS + 5).to(tl.float32, bitcast=True)
            s2 = tl.load(PARAMS + 6).to(tl.float32, bitcast=True)
            se = tl.load(PARAMS + 7)
            l0, h0, l1, h1 = affine(a, b, c, d, mu0, mu1, mu2, mue, s0, s1, s2, se, 3 if LOG else 2, True)
        tl.store(OUT + 4 * i, l0, 2 * i < N)
        tl.store(OUT + 4 * i + 1, h0, 2 * i < N)
        tl.store(OUT + 4 * i + 2, l1, 2 * i + 1 < N)
        tl.store(OUT + 4 * i + 3, h1, 2 * i + 1 < N)

    return bind_kernel(normal64)


def generate_distribution(generator, out, distribution, mean=0.0, stddev=1.0, **kwargs):
    n = out.numel()
    f64 = out.dtype == torch.float64
    words = n * (2 if f64 else 1) if distribution == "uniform" else ((n + 1) // 2) * (4 if f64 else 2)
    raw = torch.empty(words, dtype=torch.int32, device=out.device)
    generator.generate(raw, **kwargs)
    if distribution == "uniform":
        _uniform[(triton.cdiv(n, 256),)](raw, out.view(torch.int32) if f64 else out, n, f64, 256)
    elif f64:
        standard = mean == 0.0 and stddev == 1.0
        parameters = (
            None if standard else _soft_parameters(_float_bits(mean, True), _float_bits(stddev, True), str(out.device))
        )
        _normal64_kernel()[(triton.cdiv(n, 128),)](
            raw,
            out.view(torch.int32),
            parameters,
            n,
            distribution == "lognormal",
            standard,
            64,
            enable_fp_fusion=False,
            sync_solver=False,
        )
    else:
        _normal32[(triton.cdiv(n, 512),)](
            raw, out, n, mean, stddev, distribution == "lognormal", 256, enable_fp_fusion=False
        )
    return out
