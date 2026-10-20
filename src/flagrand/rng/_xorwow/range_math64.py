"""Native binary64 range-math candidates for device performance experiments.

log_unit64 accepts the actual uniform grid k/2**53, 1 <= k <= 2**53.
exp64 accepts binary64 inputs, retaining subnormal, zero and infinity outputs.
These use native f64 FMA/division and signed i64 bit reinterpretation. The
range_log64/range_exp64/range64 API modes select them for recorded experiments.
Ordinary arithmetic must remain uncontracted/unreassociated except tl.fma;
callers should use enable_fp_fusion=False, matching the CPU comparison.

Polynomial coefficients are generated from reciprocal odd integers and
factorials below; no external implementation or coefficient table is copied.
Neither helper claims all-input correctly rounded results.
"""

import math

import triton
import triton.language as tl

# Generated with Decimal(80): ln(2), its leading 21 significand bits, and the
# rounded residual. An exponent of magnitude <= 1076 times the high part is exact.
LN2_HI = tl.constexpr(0.6931467056274414)
LN2_LO = tl.constexpr(4.7493250390316726e-07)
INV_LN2 = tl.constexpr(1.4426950408889634)
# CPU references retain ordinary tuples. GPU helpers spell out the same
# Horner steps, avoiding incompatible global/local tuple indexing in vendor JITs.
LOG_COEFFICIENTS: tl.constexpr = tuple(1.0 / (2 * k + 1) for k in range(1, 13))
EXP_COEFFICIENTS: tl.constexpr = tuple(1.0 / math.factorial(k) for k in range(2, 17))


@triton.jit
def log_unit64(u):
    """Approximate log on the positive 53-bit unit grid using native f64.

    No zero/negative, subnormal input or NaN handling is part of this restricted
    interface. The smallest permitted input is the normal value 2**-53.
    """
    bits = tl.cast(u, tl.int64, bitcast=True)
    exponent = tl.cast((bits >> 52) & 2047, tl.int32) - 1023
    mantissa = tl.cast((bits & 4503599627370495) | 4607182418800017408, tl.float64, bitcast=True)
    upper = mantissa > tl.full((), 1.4142135623730951, tl.float64)
    m = tl.where(upper, mantissa * 0.5, mantissa)
    exponent = exponent + tl.cast(upper, tl.int32)
    numerator = m - 1.0
    denominator = m + 1.0
    # TwoSum preserves the low part lost in m+1, then one quotient correction
    # keeps the finite f64 division's error out of the leading log term.
    v = denominator - m
    denominator_low = (m - (denominator - v)) + (1.0 - v)
    r = numerator / denominator
    r_low = (tl.fma(-r, denominator, numerator) - r * denominator_low) / denominator
    z = r * r
    # Explicit Horner steps retain the original static-range order.
    polynomial = tl.full((), 0.04, tl.float64)
    polynomial = tl.fma(polynomial, z, tl.full((), 0.043478260869565216, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.047619047619047616, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.05263157894736842, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.058823529411764705, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.06666666666666667, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.07692307692307693, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.09090909090909091, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.1111111111111111, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.14285714285714285, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.2, tl.float64))
    polynomial = tl.fma(polynomial, z, tl.full((), 0.3333333333333333, tl.float64))
    correction = (2.0 * r_low) / (1.0 - z)
    tail = tl.fma(2.0 * r * z, polynomial, correction)
    e = tl.cast(exponent, tl.float64)
    high = tl.fma(e, tl.full((), LN2_HI, tl.float64), 2.0 * r)
    low = tl.fma(e, tl.full((), LN2_LO, tl.float64), tail)
    return high + low


@triton.jit
def exp64(x, DEGREE13: tl.constexpr = False, FINITE_RANGE: tl.constexpr = False):
    """Native f64 exp candidate, including IEEE output range and NaN/Inf.

    Only [-746, 710] needs evaluation: finite inputs outside it always round
    to +0/+Inf in binary64. Power-of-two scaling stays normal until its final
    multiplication, so underflow is not replaced with f32 or prematurely zeroed.
    This requires the target to preserve binary64 subnormals on that multiply.
    """
    if FINITE_RANGE:
        safe_x = x
    else:
        inside = (x >= -746.0) & (x <= 710.0)
        safe_x = tl.where(inside, x, tl.full((), 0.0, tl.float64))
    n = tl.cast(tl.floor(tl.fma(safe_x, tl.full((), INV_LN2, tl.float64), 0.5)), tl.int32)
    nf = tl.cast(n, tl.float64)
    r_hi = tl.fma(-nf, tl.full((), LN2_HI, tl.float64), safe_x)
    r = tl.fma(-nf, tl.full((), LN2_LO, tl.float64), r_hi)
    # MetaX's measured degree13 specialization removes three leading terms.
    # The default retains degree16 for existing callers on every backend.
    if DEGREE13:
        polynomial = tl.full((), 1.6059043836821613e-10, tl.float64)
    else:
        polynomial = tl.full((), 4.779477332387385e-14, tl.float64)
        polynomial = tl.fma(polynomial, r, tl.full((), 7.647163731819816e-13, tl.float64))
        polynomial = tl.fma(polynomial, r, tl.full((), 1.1470745597729725e-11, tl.float64))
        polynomial = tl.fma(polynomial, r, tl.full((), 1.6059043836821613e-10, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 2.08767569878681e-09, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 2.505210838544172e-08, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 2.755731922398589e-07, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 2.7557319223985893e-06, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 2.48015873015873e-05, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.0001984126984126984, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.001388888888888889, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.008333333333333333, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.041666666666666664, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.16666666666666666, tl.float64))
    polynomial = tl.fma(polynomial, r, tl.full((), 0.5, tl.float64))
    m = tl.fma(r, tl.fma(r, polynomial, tl.full((), 1.0, tl.float64)), tl.full((), 1.0, tl.float64))
    if FINITE_RANGE:
        scale = tl.cast(tl.cast(n + 1023, tl.int64) << 52, tl.float64, bitcast=True)
        return m * scale
    else:
        e = tl.minimum(tl.maximum(n, -1022), 1023)
        scale_bits = tl.cast(e + 1023, tl.int64) << 52
        remainder_bits = tl.cast(n - e + 1023, tl.int64) << 52
        scale = tl.cast(scale_bits, tl.float64, bitcast=True)
        remainder = tl.cast(remainder_bits, tl.float64, bitcast=True)
        result = (m * remainder) * scale
        result = tl.where(x < -746.0, tl.full((), 0.0, tl.float64), result)
        result = tl.where(x > 710.0, tl.full((), float("inf"), tl.float64), result)
        return tl.where(x != x, x, result)


@triton.jit
def exp64_metax(x):
    """Measured C550 degree13 path with the same binary64 range handling."""
    return exp64(x, True)


@triton.jit
def exp64_metax_finite(x):
    """Same degree13 arithmetic for inputs known to lie in [-700, 700]."""
    return exp64(x, True, True)
