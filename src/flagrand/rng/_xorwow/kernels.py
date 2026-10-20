"""Triton XORWOW recurrence and in-register distribution fusion."""

import triton
import triton.language as tl

from ._binding import bind_math_kernel
from .metax_math import sqrt_radial64
from .mthreads_math import sincos_turns32 as musa_sincos_turns32
from .native_math import hygon_sincos_turns32, sincos_turns
from .range_math64 import exp64, log_unit64


@triton.jit
def _next(x, y, z, w, v, d):
    t = x ^ (x >> 2)
    nv = ((v ^ (v << 4)) ^ (t ^ (t << 1))).to(tl.uint32)
    nd = (d + tl.full((), 362437, tl.uint32)).to(tl.uint32)
    result = (nv + nd).to(tl.uint32)
    return result, y, z, w, v, nv, nd


@triton.jit
def _uniform64(a, b, OPEN: tl.constexpr):
    # Construct 52 mantissa bits directly, then add the 53rd bit. This avoids
    # uint64-to-f64 numeric conversion, which differs across vendor compilers.
    mantissa = ((a >> 5).to(tl.uint64) << 25) | (b >> 7).to(tl.uint64)
    bits = mantissa | tl.full((), 0x3FF0000000000000, tl.uint64)
    high = bits.to(tl.float64, bitcast=True) - tl.full((), 1.0, tl.float64)
    low = (b & 64) != 0
    if OPEN:
        delta = tl.where(
            low, tl.full((), 2.220446049250313e-16, tl.float64), tl.full((), 1.1102230246251565e-16, tl.float64)
        )
    else:
        delta = tl.where(low, tl.full((), 1.1102230246251565e-16, tl.float64), tl.full((), 0.0, tl.float64))
    return high + delta


@triton.jit
def _xpu_store_contiguous(OUT, base, s, value, active, full):
    if full:
        tl.store((OUT + base) + s, value, active, offset_state_policy="continuous")
    else:
        # XPU Continuous DMA reads only the first mask lane. Partial blocks
        # need the installed backend's elementwise Unknown lowering.
        tl.store((OUT + base) + s, value, active, offset_state_policy="unknown")


def build_generate_kernel(device_math, parameter_bits_constexpr=False):
    """Bind vendor math without callable constexpr launch arguments.

    mcTriton 3.1 explicitly rejects callable constexprs in JITFunction.run.
    A captured module is resolved in Triton's normal capture scope instead.
    """

    def generate_kernel(
        STATE,
        OUT,
        N,
        MU_BITS,
        SIGMA_BITS,
        NSTREAMS: tl.constexpr,
        BLOCK: tl.constexpr,
        ROUNDS: tl.constexpr,
        DISTRIBUTION: tl.constexpr,
        F64: tl.constexpr,
        MATH_TAG: tl.constexpr,
        REDUCED_TRIG: tl.constexpr,
        ALIGNED: tl.constexpr = False,
        INDEX32: tl.constexpr = False,
        FAST_F32: tl.constexpr = False,
        HCU_SFU: tl.constexpr = False,
        INTRINSIC_TRIG: tl.constexpr = False,
        RANGE_LOG64: tl.constexpr = False,
        RANGE_EXP64: tl.constexpr = False,
        FAST_EXP_F32: tl.constexpr = False,
        TL_SQRT_F64: tl.constexpr = False,
        RSQRT_REFINED_F64: tl.constexpr = False,
        PACKED_INTERLEAVE: tl.constexpr = False,
        STANDARD_PARAMS: tl.constexpr = False,
        MUSA_PI32: tl.constexpr = False,
        XPU_SCALAR_BASE: tl.constexpr = False,
    ):
        INDEX: tl.constexpr = tl.int32 if INDEX32 else tl.int64
        s = tl.cast(tl.program_id(0), INDEX) * BLOCK + tl.cast(tl.arange(0, BLOCK), INDEX)
        n = tl.cast(N, INDEX)
        if ALIGNED:
            present = True
        else:
            present = (s < NSTREAMS) & (s < n)
        x = tl.load(STATE + s, present, other=1).to(tl.uint32, bitcast=True)
        y = tl.load(STATE + NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
        z = tl.load(STATE + 2 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
        w = tl.load(STATE + 3 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
        v = tl.load(STATE + 4 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
        d = tl.load(STATE + 5 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
        PAIR: tl.constexpr = 2 if DISTRIBUTION >= 2 else 1
        if DISTRIBUTION >= 2 and not STANDARD_PARAMS:
            if F64:
                mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
            else:
                mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)

        if ALIGNED:
            batches = n // (NSTREAMS * PAIR * ROUNDS)
        else:
            batches = tl.cdiv(n, NSTREAMS * PAIR * ROUNDS)
        for batch in range(0, batches):
            for unroll in tl.static_range(ROUNDS):
                k = tl.cast(batch, INDEX) * ROUNDS + unroll
                index = k * (NSTREAMS * PAIR) + s
                if ALIGNED:
                    active = True
                else:
                    active = present & (index < n)
                a, nx, ny, nz, nw, nv, nd = _next(x, y, z, w, v, d)
                if DISTRIBUTION == 0:
                    value0 = a.to(tl.int32, bitcast=True)
                elif DISTRIBUTION == 1:
                    if F64:
                        b, nx, ny, nz, nw, nv, nd = _next(nx, ny, nz, nw, nv, nd)
                        value0 = _uniform64(a, b, True)
                    else:
                        value0 = ((a >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
                else:
                    b, nx, ny, nz, nw, nv, nd = _next(nx, ny, nz, nw, nv, nd)
                    if F64:
                        c, nx, ny, nz, nw, nv, nd = _next(nx, ny, nz, nw, nv, nd)
                        e, nx, ny, nz, nw, nv, nd = _next(nx, ny, nz, nw, nv, nd)
                        u1 = _uniform64(a, b, True)
                        u2 = _uniform64(c, e, False)
                        theta = u2 * tl.full((), 6.283185307179586, tl.float64)
                    else:
                        # All 32 radial input bits participate before fp32 rounding.
                        # Minimum u1 is 2**-32; there is no arbitrary lower clamp.
                        u1 = (a.to(tl.float32) + 1.0) * 2.3283064365386963e-10
                        u2 = (b >> 8).to(tl.float32) * 5.960464477539063e-8
                        theta = u2 * 6.283185307179586
                    if RSQRT_REFINED_F64 and F64:
                        radius = sqrt_radial64(-2.0 * device_math.log(u1))
                    elif RANGE_LOG64 and F64:
                        radius = device_math.sqrt(-2.0 * log_unit64(u1))
                    elif TL_SQRT_F64 and F64:
                        radius = tl.sqrt(-2.0 * device_math.log(u1))
                    elif FAST_F32 and not F64:
                        radius = tl.sqrt(-2.0 * tl.log(u1))
                    else:
                        radius = device_math.sqrt(-2.0 * device_math.log(u1))
                    if MUSA_PI32 and not F64:
                        sine, cosine = musa_sincos_turns32(u2)
                    elif INTRINSIC_TRIG and not F64:
                        sine = tl.sin(theta)
                        cosine = tl.cos(theta)
                    elif HCU_SFU and not F64:
                        sine, cosine = hygon_sincos_turns32(u2)
                    elif REDUCED_TRIG:
                        sine, cosine = sincos_turns(u2, F64)
                    else:
                        sine = device_math.sin(theta)
                        cosine = device_math.cos(theta)
                    if STANDARD_PARAMS:
                        value0 = radius * cosine
                        value1 = radius * sine
                    else:
                        value0 = mu + sigma * (radius * cosine)
                        value1 = mu + sigma * (radius * sine)
                    if DISTRIBUTION == 3:
                        if RANGE_EXP64 and F64:
                            value0 = exp64(value0)
                            value1 = exp64(value1)
                        elif (FAST_F32 or FAST_EXP_F32) and not F64:
                            value0 = tl.exp(value0)
                            value1 = tl.exp(value1)
                        else:
                            value0 = device_math.exp(value0)
                            value1 = device_math.exp(value1)
                    if XPU_SCALAR_BASE:
                        base1 = k * (NSTREAMS * PAIR) + NSTREAMS
                        block_end = (tl.program_id(0) + 1) * BLOCK
                        full1 = (block_end <= NSTREAMS) & (block_end <= n - base1)
                        _xpu_store_contiguous(OUT, base1, s, value1, active & (index + NSTREAMS < n), full1)
                    elif ALIGNED:
                        tl.store(OUT + index + NSTREAMS, value1)
                    else:
                        tl.store(OUT + index + NSTREAMS, value1, active & (index + NSTREAMS < n))
                if XPU_SCALAR_BASE:
                    base0 = k * (NSTREAMS * PAIR)
                    block_end = (tl.program_id(0) + 1) * BLOCK
                    full0 = (block_end <= NSTREAMS) & (block_end <= n - base0)
                    _xpu_store_contiguous(OUT, base0, s, value0, active, full0)
                else:
                    tl.store(OUT + index, value0, active)
                if ALIGNED:
                    x, y, z, w, v, d = nx, ny, nz, nw, nv, nd
                else:
                    x = tl.where(active, nx, x)
                    y = tl.where(active, ny, y)
                    z = tl.where(active, nz, z)
                    w = tl.where(active, nw, w)
                    v = tl.where(active, nv, v)
                    d = tl.where(active, nd, d)

        tl.store(STATE + s, x.to(tl.int32, bitcast=True), present)
        tl.store(STATE + NSTREAMS + s, y.to(tl.int32, bitcast=True), present)
        tl.store(STATE + 2 * NSTREAMS + s, z.to(tl.int32, bitcast=True), present)
        tl.store(STATE + 3 * NSTREAMS + s, w.to(tl.int32, bitcast=True), present)
        tl.store(STATE + 4 * NSTREAMS + s, v.to(tl.int32, bitcast=True), present)
        tl.store(STATE + 5 * NSTREAMS + s, d.to(tl.int32, bitcast=True), present)

    return bind_math_kernel(generate_kernel, device_math, parameter_bits_constexpr=parameter_bits_constexpr)
