from __future__ import annotations

import triton
import triton.language as tl

BOX_MULLER_MIN_U_F32 = 2.3283064365386963e-10
BOX_MULLER_MIN_U_F64 = 1.1102230246251565e-16
_BOX_MULLER_MIN_U_F32_TL = tl.constexpr(BOX_MULLER_MIN_U_F32)
_BOX_MULLER_MIN_U_F64_TL = tl.constexpr(BOX_MULLER_MIN_U_F64)


@triton.jit
def uint32_to_uniform(x):
    return uint32_to_uniform_curand_compat(x)


@triton.jit
def uint32_to_uniform_curand_compat(x):
    unsigned = x.to(tl.uint32, bitcast=True)
    return unsigned.to(tl.float32) * 2.3283064365386963e-10 + 1.1641532182693481e-10


@triton.jit
def uint32_pair_to_uniform64_curand_compat(x0, x1):
    low = x0.to(tl.uint32, bitcast=True).to(tl.uint64)
    high = x1.to(tl.uint32, bitcast=True).to(tl.uint64)
    bits = low ^ (high << 21)
    return bits.to(tl.float64) * 1.1102230246251565e-16 + 5.5511151231257827e-17


@triton.jit
def uniform_to_normal_fast_f64(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F64_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    sine, cosine = _sincos_poly_f64(u2)
    n1 = r * cosine
    n2 = r * sine
    return n1, n2


@triton.jit
def uniform_to_normal_fast_f32(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F32_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.283185307179586 * u2
    n1 = r * _cos_approx_f32(theta)
    n2 = r * _sin_approx_f32(theta)
    return n1, n2


@triton.jit
def uniform_to_normal_portable_f32(u1, u2):
    u1 = tl.maximum(u1, _BOX_MULLER_MIN_U_F32_TL)
    r = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.283185307179586 * u2
    return r * tl.cos(theta), r * tl.sin(theta)


@triton.jit
def _sin_approx_f32(x):
    return tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cos_approx_f32(x):
    return tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sincos_poly_f64(unit_angle):
    quadrant = tl.floor(4.0 * unit_angle + 0.5).to(tl.int32)
    reduced = 6.283185307179586476925286766559 * unit_angle - 1.5707963267948966192313216916398 * quadrant.to(
        tl.float64
    )
    squared = reduced * reduced

    sine_coeff = 1.0 / 355687428096000.0
    sine_coeff = -1.0 / 1307674368000.0 + squared * sine_coeff
    sine_coeff = 1.0 / 6227020800.0 + squared * sine_coeff
    sine_coeff = -1.0 / 39916800.0 + squared * sine_coeff
    sine_coeff = 1.0 / 362880.0 + squared * sine_coeff
    sine_coeff = -1.0 / 5040.0 + squared * sine_coeff
    sine_coeff = 1.0 / 120.0 + squared * sine_coeff
    sine_coeff = -1.0 / 6.0 + squared * sine_coeff
    sine = reduced + reduced * squared * sine_coeff

    cosine_coeff = 1.0 / 20922789888000.0
    cosine_coeff = -1.0 / 87178291200.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 479001600.0 + squared * cosine_coeff
    cosine_coeff = -1.0 / 3628800.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 40320.0 + squared * cosine_coeff
    cosine_coeff = -1.0 / 720.0 + squared * cosine_coeff
    cosine_coeff = 1.0 / 24.0 + squared * cosine_coeff
    cosine_coeff = -0.5 + squared * cosine_coeff
    cosine = 1.0 + squared * cosine_coeff

    quadrant = quadrant & 3
    restored_sine = tl.where(
        quadrant == 0,
        sine,
        tl.where(quadrant == 1, cosine, tl.where(quadrant == 2, -sine, -cosine)),
    )
    restored_cosine = tl.where(
        quadrant == 0,
        cosine,
        tl.where(quadrant == 1, -sine, tl.where(quadrant == 2, -cosine, sine)),
    )
    return restored_sine, restored_cosine
