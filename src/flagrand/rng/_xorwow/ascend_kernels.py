"""Ascend-oriented signed-i32 XORWOW kernels, prepared for device evaluation.

Cases: raw32, uniform_f32, normal_f32, lognormal_f32, uniform_f64.
F64=True selects packed-word storage for uniform only. For that case OUT
must be a float64 allocation's int32 view; no float64 value enters the kernel.
The f64 normal/lognormal entries are outside this factory's contract.
"""

import triton
import triton.language as tl

from ._binding import bind_math_kernel
from .native_math import sincos_turns
from .packed64 import store_interleaved_words, uniform64_words


@triton.jit
def _next_i32(x, y, z, w, v, d):
    # The mask makes the logical shift independent of signed-shift lowering.
    t = x ^ ((x >> 2) & 0x3FFFFFFF)
    nv = tl.cast((v ^ (v << 4)) ^ (t ^ (t << 1)), tl.int32)
    nd = tl.cast(d + 362437, tl.int32)
    result = tl.cast(nv + nd, tl.int32)
    return result, y, z, w, v, nv, nd


@triton.jit
def _radial_u1_i32(word):
    # Both halves convert through nonnegative signed i32. Their conversion
    # and the multiplication by 2**16 are exact; the addition rounds the
    # original logical uint32 once, matching the common f32 radial mapping.
    high = tl.cast((word >> 16) & 0xFFFF, tl.float32)
    low = tl.cast(word & 0xFFFF, tl.float32)
    unsigned_float = high * 65536.0 + low
    return (unsigned_float + 1.0) * 2.3283064365386963e-10


def build_generate_kernel(device_math, parameter_bits_constexpr=False):
    """Capture the installed cann.libdevice module using the common ABI.

    Callers pass the same arguments as kernels.build_generate_kernel.
    Select this entry only for the five cases listed in the module docstring;
    normal/lognormal parameters contain binary32 bits. Kernel options,
    including enable_fp_fusion=False, remain the caller's responsibility.
    This module has not established compilation or performance on a device.
    """

    def generate_ascend_kernel(
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
        XPU_SCALAR_BASE: tl.constexpr = False,
    ):
        # The host selects i32 only for shapes whose full index range fits.
        # Otherwise widen before the first address multiply/add.
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
        PAIR: tl.constexpr = 2 if DISTRIBUTION >= 2 else 1
        if DISTRIBUTION >= 2:
            mu = tl.cast(tl.cast(MU_BITS, tl.int32), tl.float32, bitcast=True)
            sigma = tl.cast(tl.cast(SIGMA_BITS, tl.int32), tl.float32, bitcast=True)

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
                a, nx, ny, nz, nw, nv, nd = _next_i32(x, y, z, w, v, d)
                if DISTRIBUTION == 0:
                    tl.store(OUT + index, a, active)
                elif DISTRIBUTION == 1:
                    if F64:
                        b, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
                        low_word, high_word = uniform64_words(a, b)
                        # Packed output doubles the index in the selected
                        # width; the host includes that range in INDEX32.
                        if XPU_SCALAR_BASE:
                            stream_base = tl.cast(tl.program_id(0), INDEX) * BLOCK
                            base = k * NSTREAMS + stream_base
                            valid_count = tl.minimum(NSTREAMS - stream_base, n - base)
                            store_interleaved_words(
                                OUT, base, low_word, high_word, valid_count, BLOCK, ALIGNED, XPU_SCALAR_BASE
                            )
                        elif PACKED_INTERLEAVE:
                            words = tl.interleave(low_word, high_word)
                            word_index = (k * NSTREAMS + tl.cast(tl.program_id(0), INDEX) * BLOCK) * 2 + tl.cast(
                                tl.arange(0, BLOCK * 2), INDEX
                            )
                            if ALIGNED:
                                tl.store(OUT + word_index, words)
                            else:
                                stream_base = tl.cast(tl.program_id(0), INDEX) * BLOCK
                                valid_count = tl.minimum(NSTREAMS - stream_base, n - k * NSTREAMS - stream_base)
                                tl.store(OUT + word_index, words, tl.arange(0, BLOCK * 2) < valid_count * 2)
                        else:
                            tl.store(OUT + index * 2, low_word, active)
                            tl.store(OUT + index * 2 + 1, high_word, active)
                    else:
                        mantissa = ((a >> 8) & 0x00FFFFFF) + 1
                        value = tl.cast(mantissa, tl.float32) * 5.960464477539063e-8
                        tl.store(OUT + index, value, active)
                else:
                    # A pair consumes both words even when only its first
                    # output lies in range, as in the common CPU reference.
                    b, nx, ny, nz, nw, nv, nd = _next_i32(nx, ny, nz, nw, nv, nd)
                    u1 = _radial_u1_i32(a)
                    u2 = tl.cast((b >> 8) & 0x00FFFFFF, tl.float32) * 5.960464477539063e-8
                    if FAST_F32:
                        radius = tl.sqrt(-2.0 * tl.log(u1))
                    else:
                        radius = device_math.sqrt(-2.0 * device_math.log(u1))
                    if INTRINSIC_TRIG:
                        theta = u2 * 6.283185307179586
                        sine = tl.sin(theta)
                        cosine = tl.cos(theta)
                    elif REDUCED_TRIG:
                        sine, cosine = sincos_turns(u2, False)
                    else:
                        theta = u2 * 6.283185307179586
                        sine = device_math.sin(theta)
                        cosine = device_math.cos(theta)
                    value0 = mu + sigma * (radius * cosine)
                    value1 = mu + sigma * (radius * sine)
                    if DISTRIBUTION == 3:
                        if FAST_F32 or FAST_EXP_F32:
                            value0 = tl.exp(value0)
                            value1 = tl.exp(value1)
                        else:
                            value0 = device_math.exp(value0)
                            value1 = device_math.exp(value1)
                    tl.store(OUT + index, value0, active)
                    if ALIGNED:
                        tl.store(OUT + index + NSTREAMS, value1)
                    else:
                        tl.store(OUT + index + NSTREAMS, value1, active & (index + NSTREAMS < n))
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

    return bind_math_kernel(generate_ascend_kernel, device_math, parameter_bits_constexpr=parameter_bits_constexpr)
