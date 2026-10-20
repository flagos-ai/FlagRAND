"""Device-evaluation prototypes: normal/lognormal f64 via three f32s.

The standard entry denotes mu=0, sigma=1. The affine entry accepts an eight-
i32 parameter array for finite binary64 mu/sigma. OUT is an int32 view of a
little-endian binary64 allocation. No native f64/u64 math enters the device;
the recurrence uses Ascend signed-i32 operations. CPU comparisons do not
establish device support, precision or performance.
"""

import triton
import triton.language as tl

from .ascend_kernels import _next_i32
from .packed64 import store_interleaved_words
from .soft64_math import (
    affine_scaled,
    exp_from_scaled,
    exp_scaled,
    exp_scaled_outlined,
    from_words53,
    log,
    mul,
    mul_scaled,
    pack_binary64,
    sincos_turns,
    sqrt,
)


@triton.jit
def standard_pair_words(
    a,
    b,
    c,
    d,
    DISTRIBUTION: tl.constexpr,
    OUTLINED_EXP: tl.constexpr = False,
    JOINED_PAIR: tl.constexpr = False,
    INTEGER_QUADRANT_SIGN: tl.constexpr = False,
):
    """Four signed raw words -> (lo0,hi0,lo1,hi1); kinds 2=normal, 3=lognormal."""
    u0, u1, u2 = from_words53(a, b, True)
    t0, t1, t2 = from_words53(c, d, False)
    l0, l1, l2 = log(u0, u1, u2)
    r0, r1, r2 = sqrt(-2.0 * l0, -2.0 * l1, -2.0 * l2)
    s0, s1, s2, c0, c1, c2 = sincos_turns(t0, t1, t2, INTEGER_QUADRANT_SIGN)
    if DISTRIBUTION == 2:
        m0, m1, m2, e0 = mul_scaled(r0, r1, r2, c0, c1, c2)
        n0, n1, n2, e1 = mul_scaled(r0, r1, r2, s0, s1, s2)
    else:
        z0, z1, z2 = mul(r0, r1, r2, c0, c1, c2)
        w0, w1, w2 = mul(r0, r1, r2, s0, s1, s2)
        if JOINED_PAIR:
            m0, m1, m2, e0 = exp_scaled(tl.join(z0, w0), tl.join(z1, w1), tl.join(z2, w2))
        elif OUTLINED_EXP:
            m0, m1, m2, e0 = exp_scaled_outlined(z0, z1, z2)
            n0, n1, n2, e1 = exp_scaled_outlined(w0, w1, w2)
        else:
            m0, m1, m2, e0 = exp_scaled(z0, z1, z2)
            n0, n1, n2, e1 = exp_scaled(w0, w1, w2)
    if JOINED_PAIR and DISTRIBUTION == 3:
        low, high = pack_binary64(m0, m1, m2, e0, NORMAL_ONLY=True)
        low0, low1 = tl.split(low)
        high0, high1 = tl.split(high)
    else:
        low0, high0 = pack_binary64(m0, m1, m2, e0, NORMAL_ONLY=DISTRIBUTION == 3)
        low1, high1 = pack_binary64(n0, n1, n2, e1, NORMAL_ONLY=DISTRIBUTION == 3)
    return low0, high0, low1, high1


@triton.jit
def standard_lognormal_flat_words(a, b, c, d, BLOCK: tl.constexpr, INTEGER_QUADRANT_SIGN: tl.constexpr = False):
    """Return flat low/high words in cosine-B then sine-B order."""
    u0, u1, u2 = from_words53(a, b, True)
    t0, t1, t2 = from_words53(c, d, False)
    l0, l1, l2 = log(u0, u1, u2)
    r0, r1, r2 = sqrt(-2.0 * l0, -2.0 * l1, -2.0 * l2)
    s0, s1, s2, c0, c1, c2 = sincos_turns(t0, t1, t2, INTEGER_QUADRANT_SIGN)
    z0, z1, z2 = mul(r0, r1, r2, c0, c1, c2)
    w0, w1, w2 = mul(r0, r1, r2, s0, s1, s2)
    f0 = tl.reshape(tl.trans(tl.join(z0, w0), 1, 0), (2 * BLOCK,), can_reorder=False)
    f1 = tl.reshape(tl.trans(tl.join(z1, w1), 1, 0), (2 * BLOCK,), can_reorder=False)
    f2 = tl.reshape(tl.trans(tl.join(z2, w2), 1, 0), (2 * BLOCK,), can_reorder=False)
    m0, m1, m2, exponent = exp_scaled(f0, f1, f2)
    low, high = pack_binary64(m0, m1, m2, exponent, NORMAL_ONLY=True)
    return low, high


@triton.jit(do_not_specialize=["N"])
def generate_standard_soft64_kernel(
    STATE,
    OUT,
    N,
    NSTREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    ALIGNED: tl.constexpr = False,
    INDEX32: tl.constexpr = False,
    OUTLINED_EXP: tl.constexpr = False,
    JOINED_PAIR: tl.constexpr = False,
    FLAT_PAIR: tl.constexpr = False,
    PACKED_INTERLEAVE: tl.constexpr = False,
    INTEGER_QUADRANT_SIGN: tl.constexpr = False,
    XPU_SCALAR_BASE: tl.constexpr = False,
):
    """SoA state + packed standard output, with common kernel consumption.

    Stream s owns output 2*k*S+s and (2*k+1)*S+s, consuming four raw words
    when the first output exists. The absent second output is discarded.
    ALIGNED has the same full-batch meaning as ascend_kernels; INDEX32 must
    cover the doubled packed-word address range. Automatic FP contraction
    and reassociation must be disabled while retaining explicit tl.fma.
    No MATH_TAG/REDUCED_TRIG/FAST_F32 flags apply to this software-f64 path.
    OUTLINED_EXP experiments with a shared exp device-call body for standard
    lognormal; it changes no arithmetic and is disabled by default.
    JOINED_PAIR instead computes standard lognormal exp/pack once on a
    [BLOCK,2] tensor and splits its columns back to the original output order.
    FLAT_PAIR keeps the standard lognormal pair as ordered [2*BLOCK] through
    exp, packing and stores. State still belongs to the original BLOCK streams.
    """
    INDEX: tl.constexpr = tl.int32 if INDEX32 else tl.int64
    s = tl.cast(tl.program_id(0), INDEX) * BLOCK + tl.cast(tl.arange(0, BLOCK), INDEX)
    n = tl.cast(N, INDEX)
    if FLAT_PAIR and DISTRIBUTION == 3:
        flat_lane = tl.cast(tl.arange(0, 2 * BLOCK), INDEX)
        flat_stream = tl.cast(tl.program_id(0), INDEX) * BLOCK + flat_lane % BLOCK
        flat_member = flat_lane // BLOCK
    if ALIGNED:
        present = True
    else:
        present = (s < NSTREAMS) & (s < n)
    x = tl.load(STATE + s, present, other=1)
    y = tl.load(STATE + NSTREAMS + s, present, other=0)
    z = tl.load(STATE + 2 * NSTREAMS + s, present, other=0)
    w = tl.load(STATE + 3 * NSTREAMS + s, present, other=0)
    v = tl.load(STATE + 4 * NSTREAMS + s, present, other=0)
    d = tl.load(STATE + 5 * NSTREAMS + s, present, other=0)
    if ALIGNED:
        batches = n // (NSTREAMS * 2 * ROUNDS)
    else:
        batches = tl.cdiv(n, NSTREAMS * 2 * ROUNDS)
    for batch in range(0, batches):
        for unroll in tl.static_range(ROUNDS):
            k = tl.cast(batch, INDEX) * ROUNDS + unroll
            index = k * (NSTREAMS * 2) + s
            if ALIGNED:
                active = True
            else:
                active = present & (index < n)
            a, nx, ny, nz, nw, nv, nd = _next_i32(x, y, z, w, v, d)
            b, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            c, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            e, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            if FLAT_PAIR and DISTRIBUTION == 3:
                low, high = standard_lognormal_flat_words(a, b, c, e, BLOCK, INTEGER_QUADRANT_SIGN)
                flat_index = k * (NSTREAMS * 2) + flat_stream + flat_member * NSTREAMS
                if ALIGNED:
                    flat_active = True
                else:
                    flat_active = (flat_stream < NSTREAMS) & (flat_index < n)
                tl.store(OUT + flat_index * 2, low, flat_active)
                tl.store(OUT + flat_index * 2 + 1, high, flat_active)
            else:
                low0, high0, low1, high1 = standard_pair_words(
                    a, b, c, e, DISTRIBUTION, OUTLINED_EXP, JOINED_PAIR, INTEGER_QUADRANT_SIGN
                )
                if PACKED_INTERLEAVE:
                    base = k * (NSTREAMS * 2) + tl.cast(tl.program_id(0), INDEX) * BLOCK
                    store_interleaved_words(
                        OUT,
                        base,
                        low0,
                        high0,
                        tl.minimum(NSTREAMS - tl.program_id(0) * BLOCK, n - base),
                        BLOCK,
                        ALIGNED,
                        XPU_SCALAR_BASE,
                    )
                    store_interleaved_words(
                        OUT,
                        base + NSTREAMS,
                        low1,
                        high1,
                        tl.minimum(NSTREAMS - tl.program_id(0) * BLOCK, n - base - NSTREAMS),
                        BLOCK,
                        ALIGNED,
                        XPU_SCALAR_BASE,
                    )
                else:
                    tl.store(OUT + index * 2, low0, active)
                    tl.store(OUT + index * 2 + 1, high0, active)
                    if ALIGNED:
                        tl.store(OUT + (index + NSTREAMS) * 2, low1)
                        tl.store(OUT + (index + NSTREAMS) * 2 + 1, high1)
                    else:
                        second = active & (index + NSTREAMS < n)
                        tl.store(OUT + (index + NSTREAMS) * 2, low1, second)
                        tl.store(OUT + (index + NSTREAMS) * 2 + 1, high1, second)
            if ALIGNED:
                x, y, z, w, v, d = nx, ny, nz, nw, nv, nd
            else:
                x = tl.where(active, nx, x)
                y = tl.where(active, ny, y)
                z = tl.where(active, nz, z)
                w = tl.where(active, nw, w)
                v = tl.where(active, nv, v)
                d = tl.where(active, nd, d)
    tl.store(STATE + s, x, present)
    tl.store(STATE + NSTREAMS + s, y, present)
    tl.store(STATE + 2 * NSTREAMS + s, z, present)
    tl.store(STATE + 3 * NSTREAMS + s, w, present)
    tl.store(STATE + 4 * NSTREAMS + s, v, present)
    tl.store(STATE + 5 * NSTREAMS + s, d, present)


@triton.jit
def affine_pair_words(
    a,
    b,
    c,
    d,
    mu0,
    mu1,
    mu2,
    mue,
    sigma0,
    sigma1,
    sigma2,
    sigmae,
    DISTRIBUTION: tl.constexpr,
    INTEGER_QUADRANT_SIGN: tl.constexpr = False,
):
    """F64 pair with finite mu/sigma supplied as normalized scaled triples."""
    u0, u1, u2 = from_words53(a, b, True)
    t0, t1, t2 = from_words53(c, d, False)
    l0, l1, l2 = log(u0, u1, u2)
    r0, r1, r2 = sqrt(-2.0 * l0, -2.0 * l1, -2.0 * l2)
    s0, s1, s2, c0, c1, c2 = sincos_turns(t0, t1, t2, INTEGER_QUADRANT_SIGN)
    z0, z1, z2, ze = mul_scaled(r0, r1, r2, c0, c1, c2)
    w0, w1, w2, we = mul_scaled(r0, r1, r2, s0, s1, s2)
    z0, z1, z2, ze = affine_scaled(z0, z1, z2, ze, mu0, mu1, mu2, mue, sigma0, sigma1, sigma2, sigmae)
    w0, w1, w2, we = affine_scaled(w0, w1, w2, we, mu0, mu1, mu2, mue, sigma0, sigma1, sigma2, sigmae)
    if DISTRIBUTION == 3:
        z0, z1, z2, ze = exp_from_scaled(z0, z1, z2, ze)
        w0, w1, w2, we = exp_from_scaled(w0, w1, w2, we)
    low0, high0 = pack_binary64(z0, z1, z2, ze)
    low1, high1 = pack_binary64(w0, w1, w2, we)
    return low0, high0, low1, high1


@triton.jit(do_not_specialize=["N"])
def generate_affine_soft64_kernel(
    STATE,
    OUT,
    N,
    PARAMETERS,
    NSTREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUNDS: tl.constexpr,
    DISTRIBUTION: tl.constexpr,
    ALIGNED: tl.constexpr = False,
    INDEX32: tl.constexpr = False,
    PACKED_INTERLEAVE: tl.constexpr = False,
    INTEGER_QUADRANT_SIGN: tl.constexpr = False,
    XPU_SCALAR_BASE: tl.constexpr = False,
):
    """General finite-parameter soft-f64 normal/lognormal; OUT is an i32 view.

    PARAMETERS is an eight-i32 device array: mu's three binary32 bit patterns
    and signed exponent, then sigma's three patterns and exponent. Host frexp
    decomposition of each finite binary64 parameter is exact. NaN/Inf
    parameters are outside this prototype's contract. No native f64 values
    or unsigned conversions enter the device computation.
    """
    INDEX: tl.constexpr = tl.int32 if INDEX32 else tl.int64
    s = tl.cast(tl.program_id(0), INDEX) * BLOCK + tl.cast(tl.arange(0, BLOCK), INDEX)
    n = tl.cast(N, INDEX)
    if ALIGNED:
        present = True
    else:
        present = (s < NSTREAMS) & (s < n)
    x = tl.load(STATE + s, present, other=1)
    y = tl.load(STATE + NSTREAMS + s, present, other=0)
    z = tl.load(STATE + 2 * NSTREAMS + s, present, other=0)
    w = tl.load(STATE + 3 * NSTREAMS + s, present, other=0)
    v = tl.load(STATE + 4 * NSTREAMS + s, present, other=0)
    d = tl.load(STATE + 5 * NSTREAMS + s, present, other=0)
    mu0 = tl.load(PARAMETERS + 0).to(tl.float32, bitcast=True)
    mu1 = tl.load(PARAMETERS + 1).to(tl.float32, bitcast=True)
    mu2 = tl.load(PARAMETERS + 2).to(tl.float32, bitcast=True)
    mue = tl.load(PARAMETERS + 3)
    sigma0 = tl.load(PARAMETERS + 4).to(tl.float32, bitcast=True)
    sigma1 = tl.load(PARAMETERS + 5).to(tl.float32, bitcast=True)
    sigma2 = tl.load(PARAMETERS + 6).to(tl.float32, bitcast=True)
    sigmae = tl.load(PARAMETERS + 7)
    if ALIGNED:
        batches = n // (NSTREAMS * 2 * ROUNDS)
    else:
        batches = tl.cdiv(n, NSTREAMS * 2 * ROUNDS)
    for batch in range(0, batches):
        for unroll in tl.static_range(ROUNDS):
            k = tl.cast(batch, INDEX) * ROUNDS + unroll
            index = k * (NSTREAMS * 2) + s
            if ALIGNED:
                active = True
            else:
                active = present & (index < n)
            a, nx, ny, nz, nw, nv, nd = _next_i32(x, y, z, w, v, d)
            b, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            c, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            e, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
            low0, high0, low1, high1 = affine_pair_words(
                a, b, c, e, mu0, mu1, mu2, mue, sigma0, sigma1, sigma2, sigmae, DISTRIBUTION, INTEGER_QUADRANT_SIGN
            )
            if PACKED_INTERLEAVE:
                base = k * (NSTREAMS * 2) + tl.cast(tl.program_id(0), INDEX) * BLOCK
                store_interleaved_words(
                    OUT,
                    base,
                    low0,
                    high0,
                    tl.minimum(NSTREAMS - tl.program_id(0) * BLOCK, n - base),
                    BLOCK,
                    ALIGNED,
                    XPU_SCALAR_BASE,
                )
                store_interleaved_words(
                    OUT,
                    base + NSTREAMS,
                    low1,
                    high1,
                    tl.minimum(NSTREAMS - tl.program_id(0) * BLOCK, n - base - NSTREAMS),
                    BLOCK,
                    ALIGNED,
                    XPU_SCALAR_BASE,
                )
            else:
                tl.store(OUT + index * 2, low0, active)
                tl.store(OUT + index * 2 + 1, high0, active)
                if ALIGNED:
                    tl.store(OUT + (index + NSTREAMS) * 2, low1)
                    tl.store(OUT + (index + NSTREAMS) * 2 + 1, high1)
                else:
                    second = active & (index + NSTREAMS < n)
                    tl.store(OUT + (index + NSTREAMS) * 2, low1, second)
                    tl.store(OUT + (index + NSTREAMS) * 2 + 1, high1, second)
            if ALIGNED:
                x, y, z, w, v, d = nx, ny, nz, nw, nv, nd
            else:
                x = tl.where(active, nx, x)
                y = tl.where(active, ny, y)
                z = tl.where(active, nz, z)
                w = tl.where(active, nw, w)
                v = tl.where(active, nv, v)
                d = tl.where(active, nd, d)
    tl.store(STATE + s, x, present)
    tl.store(STATE + NSTREAMS + s, y, present)
    tl.store(STATE + 2 * NSTREAMS + s, z, present)
    tl.store(STATE + 3 * NSTREAMS + s, w, present)
    tl.store(STATE + 4 * NSTREAMS + s, v, present)
    tl.store(STATE + 5 * NSTREAMS + s, d, present)
