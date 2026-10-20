"""Experimental three-binary32 arithmetic for Ascend's missing native f64 math.

Each value is a normalized (hi, middle, low) tuple, about 72 significant bits.
Callers must disable automatic FP contraction/reassociation. Explicit tl.fma
is required to be a correctly rounded fused operation for two_prod. This
property, and any FTZ effect, still require a device comparison on 910C.

These helpers do not cast an f32 approximation into f64. They compute with
three f32 components; exp_scaled keeps its power-of-two exponent separate.
CPU scalar comparison: the actual helper bodies are interpreted by
soft64_check.py, with integer-exact FMA and Decimal(100) references. This is
not a proof of correctly rounded binary64 for every input. The intended
combination uses log([2**-53,1]), sqrt([2**-53,80] or zero), sincos_turns of a
53-bit uniform, and mul_scaled for the final normal value. Arbitrary triple
exponent ranges outside the scaled interfaces are unsupported. Finite binary64
mu/sigma are handled by affine_scaled; NaN/Inf parameters are outside scope.
No device performance or accuracy claim is made by this source file.
"""

import triton
import triton.language as tl

# CoreX3.1 recognizes annotated globals but cannot subscript a constexpr wrapper.
LN2: tl.constexpr = (0.6931471824645996, -1.9046542121259336e-09, -8.78318373858934e-17)

TAU: tl.constexpr = (6.2831854820251465, -1.7484555314695172e-07, -6.8604980400235275e-15)

LOG_COEFF: tl.constexpr = (
    (1.0, 0.0, 0.0),
    (0.3333333432674408, -9.934107758624577e-09, 2.9605948205663494e-16),
    (0.20000000298023224, -2.9802322831784522e-09, 4.440892164675075e-17),
    (0.1428571492433548, -6.38621200366174e-09, 6.34413185289139e-17),
    (0.1111111119389534, -8.278422947149977e-10, 6.167905738316459e-18),
    (0.09090909361839294, -2.709302115988521e-09, 8.074349510635498e-17),
    (0.07692307978868484, -2.8656079731348427e-09, 5.1241062929524336e-17),
    (0.06666667014360428, -3.47693762670076e-09, 1.4802974433703992e-17),
    (0.05882352963089943, -2.1913472425527658e-10, 8.163404623243085e-19),
    (0.05263157933950424, -3.9213582381236733e-10, 2.921639560255165e-18),
    (0.0476190485060215, -8.869738832295582e-10, 2.6433882031063616e-18),
    (0.043478261679410934, -8.098456905081264e-10, -2.6548811746142652e-17),
    (0.03999999910593033, 8.940696516468449e-10, 1.9984015237346206e-17),
    (0.03703703731298447, -2.759474315716659e-10, 2.0559685794388198e-18),
    (0.03448275849223137, 1.284582856753147e-10, 4.785444053832989e-19),
    (0.032258063554763794, 9.613652096618353e-10, 2.865091591065631e-17),
    (0.03030303120613098, -9.03100705329507e-10, 2.6914498368784995e-17),
    (0.02857142873108387, -1.5965530286710106e-10, 4.3615906075038e-18),
)

SIN_COEFF: tl.constexpr = (
    (1.0, 0.0, 0.0),
    (-0.1666666716337204, 4.967053879312289e-09, -1.4802974102831747e-16),
    (0.008333333767950535, -4.34617203337595e-10, 1.850371804212999e-18),
    (-0.00019841270113829523, 2.725596874933456e-12, -5.4382207428126e-20),
    (2.7557318844628753e-06, 3.793571224297229e-14, 1.508226522084065e-21),
    (-2.5052107943679403e-08, -4.4176230446483665e-16, -1.1786075750928585e-23),
    (1.6059044372074283e-10, -5.352526511562726e-18, -1.6836046932930509e-25),
    (-7.647163609812713e-13, -1.2200710471178288e-20, 9.39354696078821e-29),
    (2.8114573589663704e-15, -1.0462084739763658e-22, -2.201494220163496e-30),
    (-8.220635078476521e-18, -1.681478096855898e-25, 1.3193249926885217e-33),
    (1.9572941524685808e-20, -4.612945513820017e-28, 4.654180614862984e-36),
    (-3.868170297964731e-23, 1.2733404317146773e-30, 3.614914794655009e-38),
    (6.446950340479164e-26, -5.609469144028533e-34, 9.547046437444979e-42),
)

COS_COEFF: tl.constexpr = (
    (1.0, 0.0, 0.0),
    (-0.5, 0.0, 0.0),
    (0.0416666679084301, -1.2417634698280722e-09, 3.700743525707937e-17),
    (-0.0013888889225199819, 3.3631094437103215e-11, -1.4648776180200379e-18),
    (2.4801587642286904e-05, -3.40699609366682e-13, 6.79777592851575e-21),
    (-2.755731998149713e-07, 7.575112209051195e-15, 1.8799052669331363e-22),
    (2.0876755879584152e-09, 1.1082839147459852e-16, 3.1879878803137138e-24),
    (-1.147074536050896e-11, -2.372207689231238e-19, 4.640206185514343e-27),
    (4.7794772561329454e-14, 7.62544404448643e-22, -5.870966850492631e-30),
    (-1.5619206814541513e-16, -1.5404471465941993e-24, 1.2741222941497115e-32),
    (4.110317590937049e-19, 3.237511732860265e-27, -1.4300344511402812e-34),
    (-8.896790959566848e-22, -4.3288372034341836e-29, -4.851232505321315e-37),
    (1.6117375912827668e-24, -2.0186648406604e-32, -3.6847143119421065e-41),
    (-2.4795962847996784e-27, 2.157488154394507e-35, -5.885453550164232e-43),
)

EXP_COEFF: tl.constexpr = (
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.1666666716337204, -4.967053879312289e-09, 1.4802974102831747e-16),
    (0.0416666679084301, -1.2417634698280722e-09, 3.700743525707937e-17),
    (0.008333333767950535, -4.34617203337595e-10, 1.850371804212999e-18),
    (0.0013888889225199819, -3.3631094437103215e-11, 1.4648776180200379e-18),
    (0.00019841270113829523, -2.725596874933456e-12, 5.4382207428126e-20),
    (2.4801587642286904e-05, -3.40699609366682e-13, 6.79777592851575e-21),
    (2.7557318844628753e-06, 3.793571224297229e-14, 1.508226522084065e-21),
    (2.755731998149713e-07, -7.575112209051195e-15, -1.8799052669331363e-22),
    (2.5052107943679403e-08, 4.4176230446483665e-16, 1.1786075750928585e-23),
    (2.0876755879584152e-09, 1.1082839147459852e-16, 3.1879878803137138e-24),
    (1.6059044372074283e-10, -5.352526511562726e-18, -1.6836046932930509e-25),
    (1.147074536050896e-11, 2.372207689231238e-19, -4.640206185514343e-27),
    (7.647163609812713e-13, 1.2200710471178288e-20, -9.39354696078821e-29),
    (4.7794772561329454e-14, 7.62544404448643e-22, -5.870966850492631e-30),
    (2.8114573589663704e-15, -1.0462084739763658e-22, -2.201494220163496e-30),
    (1.5619206814541513e-16, 1.5404471465941993e-24, -1.2741222941497115e-32),
    (8.220635078476521e-18, 1.681478096855898e-25, -1.3193249926885217e-33),
    (4.110317590937049e-19, 3.237511732860265e-27, -1.4300344511402812e-34),
    (1.9572941524685808e-20, -4.612945513820017e-28, 4.654180614862984e-36),
    (8.896790959566848e-22, 4.3288372034341836e-29, 4.851232505321315e-37),
    (3.868170297964731e-23, -1.2733404317146773e-30, -3.614914794655009e-38),
    (1.6117375912827668e-24, -2.0186648406604e-32, -3.6847143119421065e-41),
)


@triton.jit
def two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


@triton.jit
def two_prod(a, b):
    p = a * b
    e = tl.fma(a, b, -p)
    return p, e


@triton.jit
def renorm(a, b, c):
    b, c = two_sum(b, c)
    a, b = two_sum(a, b)
    b, c = two_sum(b, c)
    a, b = two_sum(a, b)
    b, c = two_sum(b, c)
    return a, b, c


@triton.jit
def add(a0, a1, a2, b0, b1, b2):
    s, e = two_sum(a0, b0)
    t, f = two_sum(a1, b1)
    t, g = two_sum(e, t)
    low = ((a2 + b2) + f) + g
    return renorm(s, t, low)


@triton.jit
def sub(a0, a1, a2, b0, b1, b2):
    return add(a0, a1, a2, -b0, -b1, -b2)


@triton.jit
def mul(a0, a1, a2, b0, b1, b2):
    p, q = two_prod(a0, b0)
    r, s = two_prod(a0, b1)
    t, u = two_prod(a1, b0)
    r, v = two_sum(r, t)
    r, w = two_sum(q, r)
    low = (((((s + u) + v) + w) + a0 * b2) + a1 * b1) + a2 * b0
    return renorm(p, r, low)


@triton.jit
def mul_scaled(a0, a1, a2, b0, b1, b2):
    """Return normalized product components and a separate base-two exponent.

    Use this for the final normal output: a near-zero trigonometric value
    times a small radius can put the third component below f32's normal
    range. Keeping the exponent separate avoids that FTZ precision loss.
    Input leading components must be zero or have magnitudes in
    [2**-100, 2**100], so their reciprocal scale is also normal binary32.
    """
    ae = ((a0.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    be = ((b0.to(tl.int32, bitcast=True) >> 23) & 255) - 127
    ae = tl.where(a0 == 0.0, 0, ae)
    be = tl.where(b0 == 0.0, 0, be)
    af = ((127 - ae) << 23).to(tl.float32, bitcast=True)
    bf = ((127 - be) << 23).to(tl.float32, bitcast=True)
    p0, p1, p2 = mul(a0 * af, a1 * af, a2 * af, b0 * bf, b1 * bf, b2 * bf)
    return p0, p1, p2, ae + be


@triton.jit
def div(a0, a1, a2, b0, b1, b2):
    z = tl.full((), 0.0, tl.float32)
    q0 = a0 / b0
    q1 = z
    q2 = z
    for _ in tl.static_range(3):
        p0, p1, p2 = mul(b0, b1, b2, q0, q1, q2)
        r0, r1, r2 = sub(a0, a1, a2, p0, p1, p2)
        c = r0 / b0
        q0, q1, q2 = add(q0, q1, q2, c, z, z)
    return q0, q1, q2


@triton.jit
def from_words53(a, b, OPEN: tl.constexpr):
    # Exact 27+26-bit grid, represented as three binary32 components.
    a = a.to(tl.int32, bitcast=True)
    b = b.to(tl.int32, bitcast=True)
    high = ((a >> 8) & 0x00FFFFFF).to(tl.float32) * 5.960464477539063e-8
    middle_bits = (((a >> 5) & 7) << 21) | ((b >> 11) & 0x001FFFFF)
    middle = middle_bits.to(tl.float32) * 3.552713678800501e-15
    low_bits = ((b >> 6) & 31) + (1 if OPEN else 0)
    low = low_bits.to(tl.float32) * 1.1102230246251565e-16
    return renorm(high, middle, low)


@triton.jit
def log(a0, a1, a2):
    """Natural log for positive inputs in [2**-53, 1], without clipping."""
    z = tl.full((), 0.0, tl.float32)
    e = (a0.to(tl.int32, bitcast=True) >> 23) - 127
    scale = ((127 - e) << 23).to(tl.float32, bitcast=True)
    m0, m1, m2 = a0 * scale, a1 * scale, a2 * scale
    upper = m0 > 1.4142135623730951
    f = tl.where(upper, 0.5, 1.0)
    m0, m1, m2 = m0 * f, m1 * f, m2 * f
    e = e + upper.to(tl.int32)
    n0, n1, n2 = add(m0, m1, m2, -1.0, z, z)
    d0, d1, d2 = add(m0, m1, m2, 1.0, z, z)
    y0, y1, y2 = div(n0, n1, n2, d0, d1, d2)
    v0, v1, v2 = mul(y0, y1, y2, y0, y1, y2)
    p0 = tl.full((), LOG_COEFF[17][0], tl.float32)
    p1 = tl.full((), LOG_COEFF[17][1], tl.float32)
    p2 = tl.full((), LOG_COEFF[17][2], tl.float32)
    for k in tl.static_range(16, -1, -1):
        p0, p1, p2 = mul(p0, p1, p2, v0, v1, v2)
        p0, p1, p2 = add(
            p0,
            p1,
            p2,
            tl.full((), LOG_COEFF[k][0], tl.float32),
            tl.full((), LOG_COEFF[k][1], tl.float32),
            tl.full((), LOG_COEFF[k][2], tl.float32),
        )
    p0, p1, p2 = mul(p0, p1, p2, y0 * 2.0, y1 * 2.0, y2 * 2.0)
    q0, q1, q2 = mul(
        e.to(tl.float32),
        z,
        z,
        tl.full((), LN2[0], tl.float32),
        tl.full((), LN2[1], tl.float32),
        tl.full((), LN2[2], tl.float32),
    )
    return add(p0, p1, p2, q0, q1, q2)


@triton.jit
def sqrt(a0, a1, a2):
    """Square root for inputs in [2**-53,80], plus the exact zero case."""
    z = tl.full((), 0.0, tl.float32)
    nonzero = a0 != 0.0
    x0 = tl.where(nonzero, a0, 1.0)
    x1 = tl.where(nonzero, a1, 0.0)
    x2 = tl.where(nonzero, a2, 0.0)
    y0, y1, y2 = tl.sqrt(x0), z, z
    for _ in tl.static_range(2):
        p0, p1, p2 = mul(y0, y1, y2, y0, y1, y2)
        r0, r1, r2 = sub(x0, x1, x2, p0, p1, p2)
        d0, d1, d2 = div(r0, r1, r2, y0 * 2.0, y1 * 2.0, y2 * 2.0)
        y0, y1, y2 = add(y0, y1, y2, d0, d1, d2)
    return tl.where(nonzero, y0, 0.0), tl.where(nonzero, y1, 0.0), tl.where(nonzero, y2, 0.0)


@triton.jit
def sincos_reduced(a0, a1, a2):
    """Return (sin_hi, sin_mid, sin_low, cos_hi, cos_mid, cos_low), |a|<=pi/4."""
    v0, v1, v2 = mul(a0, a1, a2, a0, a1, a2)
    s0 = tl.full((), SIN_COEFF[12][0], tl.float32)
    s1 = tl.full((), SIN_COEFF[12][1], tl.float32)
    s2 = tl.full((), SIN_COEFF[12][2], tl.float32)
    for k in tl.static_range(11, -1, -1):
        s0, s1, s2 = mul(s0, s1, s2, v0, v1, v2)
        s0, s1, s2 = add(
            s0,
            s1,
            s2,
            tl.full((), SIN_COEFF[k][0], tl.float32),
            tl.full((), SIN_COEFF[k][1], tl.float32),
            tl.full((), SIN_COEFF[k][2], tl.float32),
        )
    s0, s1, s2 = mul(s0, s1, s2, a0, a1, a2)
    c0 = tl.full((), COS_COEFF[13][0], tl.float32)
    c1 = tl.full((), COS_COEFF[13][1], tl.float32)
    c2 = tl.full((), COS_COEFF[13][2], tl.float32)
    for k in tl.static_range(12, -1, -1):
        c0, c1, c2 = mul(c0, c1, c2, v0, v1, v2)
        c0, c1, c2 = add(
            c0,
            c1,
            c2,
            tl.full((), COS_COEFF[k][0], tl.float32),
            tl.full((), COS_COEFF[k][1], tl.float32),
            tl.full((), COS_COEFF[k][2], tl.float32),
        )
    return s0, s1, s2, c0, c1, c2


@triton.jit
def sincos_turns(u0, u1, u2, INTEGER_QUADRANT_SIGN: tl.constexpr = False):
    """sincos(2*pi*u) for a 53-bit uniform triple in [0,1)."""
    z = tl.full((), 0.0, tl.float32)
    q = tl.floor(u0 * 4.0 + 0.5).to(tl.int32)
    r0, r1, r2 = add(u0, u1, u2, q.to(tl.float32) * -0.25, z, z)
    a0, a1, a2 = mul(
        r0, r1, r2, tl.full((), TAU[0], tl.float32), tl.full((), TAU[1], tl.float32), tl.full((), TAU[2], tl.float32)
    )
    s0, s1, s2, c0, c1, c2 = sincos_reduced(a0, a1, a2)
    quadrant = q & 3
    odd = (quadrant & 1) != 0
    rs0, rs1, rs2 = tl.where(odd, c0, s0), tl.where(odd, c1, s1), tl.where(odd, c2, s2)
    rc0, rc1, rc2 = tl.where(odd, s0, c0), tl.where(odd, s1, c1), tl.where(odd, s2, c2)
    ss = tl.where(quadrant >= 2, -1.0, 1.0)
    if INTEGER_QUADRANT_SIGN:
        # Avoid the old XPU SelectOp fusion's assumption that ori inputs are cmpf.
        cs = tl.where(((quadrant + 1) & 2) != 0, -1.0, 1.0)
    else:
        cs = tl.where((quadrant == 1) | (quadrant == 2), -1.0, 1.0)
    return rs0 * ss, rs1 * ss, rs2 * ss, rc0 * cs, rc1 * cs, rc2 * cs


@triton.jit
def exp_scaled(a0, a1, a2):
    """Return (m_hi,m_mid,m_low,k), exp(a)=m*2**k, for a in [-746,710].

    A separate exponent avoids f32 exponent-range overflow/underflow. The
    caller must round/pack this representation into binary64, including its
    subnormal and overflow cases; casting the components to f32 is incorrect.
    """
    z = tl.full((), 0.0, tl.float32)
    k = tl.floor(a0 * 1.4426950408889634 + 0.5).to(tl.int32)
    q0, q1, q2 = mul(
        k.to(tl.float32),
        z,
        z,
        tl.full((), LN2[0], tl.float32),
        tl.full((), LN2[1], tl.float32),
        tl.full((), LN2[2], tl.float32),
    )
    r0, r1, r2 = sub(a0, a1, a2, q0, q1, q2)
    p0 = tl.full((), EXP_COEFF[24][0], tl.float32)
    p1 = tl.full((), EXP_COEFF[24][1], tl.float32)
    p2 = tl.full((), EXP_COEFF[24][2], tl.float32)
    for j in tl.static_range(23, -1, -1):
        p0, p1, p2 = mul(p0, p1, p2, r0, r1, r2)
        p0, p1, p2 = add(
            p0,
            p1,
            p2,
            tl.full((), EXP_COEFF[j][0], tl.float32),
            tl.full((), EXP_COEFF[j][1], tl.float32),
            tl.full((), EXP_COEFF[j][2], tl.float32),
        )
    return p0, p1, p2, k


@triton.jit(noinline=True)
def exp_scaled_outlined(x0, x1, x2):
    """Optional device-call boundary around the unchanged exp_scaled body."""
    return exp_scaled(x0, x1, x2)


@triton.jit
def _unsigned_less_i32(a, b):
    sign_bit = tl.full((), -2147483648, tl.int32)
    return (a ^ sign_bit) < (b ^ sign_bit)


@triton.jit
def _logical_right_i32(value, shift):
    # A2/A3's documented vector shift counts are restricted. Every hardware
    # shift below has a compile-time constant count; masks select the stages.
    for bit in tl.static_range(5):
        amount = 1 << bit
        shifted = (value >> amount) & ((1 << (32 - amount)) - 1)
        value = tl.where((shift & amount) != 0, shifted, value)
    return value


@triton.jit
def _left_i32(value, shift):
    for bit in tl.static_range(5):
        amount = 1 << bit
        value = tl.where((shift & amount) != 0, value << amount, value)
    return value


@triton.jit
def pack_binary64(m0, m1, m2, exponent, NORMAL_ONLY: tl.constexpr = False):
    """Round (m0+m1+m2)*2**exponent to (low_int32, high_int32) IEEE words.

    Uses only binary32 and 32-bit integers. Input is a non-overlapping scaled
    triple with |m0| in [0.5,4], or signed zero. Supports normal/subnormal
    binary64 output, round-to-nearest ties-even, signed zero and overflow to
    signed infinity. NaN input and non-normalized triples are outside scope.
    Rounding is relative to the supplied triple, not the unknown exact value
    of a preceding transcendental function.

    NORMAL_ONLY specializes finite nonzero normal-range outputs. Standard
    mu=0/sigma=1 lognormal uses it: exp(+-sqrt(106*ln(2))) stays far inside
    binary64's normal range. The generic affine path retains the default.
    """
    one = tl.full((), 1, tl.int32)
    bits = m0.to(tl.int32, bitcast=True)
    sign = bits & tl.full((), -2147483648, tl.int32)
    negative = sign != 0
    zero = (bits & 0x7FFFFFFF) == 0
    a0 = tl.where(negative, -m0, m0)
    a1 = tl.where(negative, -m1, m1)
    a2 = tl.where(negative, -m2, m2)
    base = ((a0.to(tl.int32, bitcast=True) >> 23) & 255).to(tl.int32) - 127
    base = tl.where(zero, 0, base)
    scale = ((127 - base) << 23).to(tl.float32, bitcast=True)
    a0, a1, a2 = a0 * scale, a1 * scale, a2 * scale
    below_one = (a0 == 1.0) & ((a1 < 0.0) | ((a1 == 0.0) & (a2 < 0.0)))
    factor = tl.where(below_one, 2.0, 1.0)
    a0, a1, a2 = a0 * factor, a1 * factor, a2 * factor
    e = exponent + base - below_one.to(tl.int32)

    # a0 contributes an exact 24-bit significand shifted left by 29. The
    # remaining two components supply a signed integer delta and a fraction.
    significand = (a0.to(tl.int32, bitcast=True) & 0x7FFFFF) | 0x800000
    significand = tl.where(a0 == 2.0, 0x1000000, significand).to(tl.int32)
    d0, d1 = a1 * 4503599627370496.0, a2 * 4503599627370496.0
    i0, i1 = tl.floor(d0).to(tl.int32), tl.floor(d1).to(tl.int32)
    z = tl.full((), 0.0, tl.float32)
    frac, error, last = add(d0, d1, z, -i0.to(tl.float32), -i1.to(tl.float32), z)
    tail_positive = (error > 0.0) | ((error == 0.0) & (last > 0.0))
    tail_zero = (error == 0.0) & (last == 0.0)
    carry_fraction = (frac > 1.0) | ((frac == 1.0) & (tail_positive | tail_zero))
    integer_delta = i0 + i1 + carry_fraction.to(tl.int32)
    fraction_nonzero = (frac != carry_fraction.to(tl.float32)) | (~tail_zero)
    midpoint = carry_fraction.to(tl.float32) + 0.5
    fraction_greater_half = (frac > midpoint) | ((frac == midpoint) & tail_positive)
    fraction_equal_half = (frac == midpoint) & tail_zero

    old_low = significand << 29
    low = old_low + integer_delta.to(tl.int32)
    carry = (integer_delta >= 0) & _unsigned_less_i32(low, old_low)
    borrow = (integer_delta < 0) & _unsigned_less_i32(old_low, low)
    high = (significand >> 3) + carry.to(tl.int32) - borrow.to(tl.int32)
    if NORMAL_ONLY:
        shift = 0
        qlo, qhi = low, high
        increment = fraction_greater_half | (fraction_equal_half & ((qlo & 1) != 0))
    else:
        shift = tl.maximum(-1022 - e, 0)

        # Discarded bits remain separate from the quotient, preventing the double
        # rounding that would arise from rounding to 53 bits before subnormalizing.
        s = tl.minimum(tl.maximum(shift, 1), 31)
        lost_low = low & (_left_i32(one, s) - one)
        half_low = _left_i32(one, s - 1)
        qlo_small = _logical_right_i32(low, s) | _left_i32(high, 32 - s)
        qhi_small = _logical_right_i32(high, s)
        greater_small = lost_low > half_low
        equal_small = lost_low == half_low

        h = tl.minimum(tl.maximum(shift - 32, 1), 21)
        lost_high = high & (_left_i32(one, h) - one)
        half_high = _left_i32(one, h - 1)
        greater_large = (lost_high > half_high) | ((lost_high == half_high) & (low != 0))
        equal_large = (lost_high == half_high) & (low == 0)
        qlo_large = _logical_right_i32(high, h)

        qlo = tl.where(
            shift == 0,
            low,
            tl.where(shift < 32, qlo_small, tl.where(shift == 32, high, tl.where(shift <= 53, qlo_large, 0))),
        )
        qhi = tl.where(shift == 0, high, tl.where(shift < 32, qhi_small, 0))
        greater = tl.where(
            shift < 32, greater_small, tl.where(shift == 32, (low < 0) & (low != -2147483648), greater_large)
        )
        equal = tl.where(shift < 32, equal_small, tl.where(shift == 32, low == -2147483648, equal_large))
        increment_normal = fraction_greater_half | (fraction_equal_half & ((qlo & 1) != 0))
        increment_subnormal = greater | (equal & (fraction_nonzero | ((qlo & 1) != 0)))
        increment = tl.where(shift == 0, increment_normal, (shift <= 53) & increment_subnormal)
    rounded_low = qlo + increment.to(tl.int32)
    rounded_high = qhi + _unsigned_less_i32(rounded_low, qlo).to(tl.int32)

    renormalize = (shift == 0) & (rounded_high == 0x200000)
    rounded_low = tl.where(renormalize, 0, rounded_low)
    rounded_high = tl.where(renormalize, 0x100000, rounded_high)
    e = e + renormalize.to(tl.int32)
    exponent_field = tl.where(shift == 0, e + 1023, 0).to(tl.int32)
    output_high = sign | (exponent_field << 20) | (rounded_high & 0xFFFFF)
    if not NORMAL_ONLY:
        # A subnormal rounding up to 2**52 becomes the smallest normal value.
        output_high = tl.where((shift > 0) & (rounded_high == 0x100000), sign | 0x00100000, output_high)
        overflow = (shift == 0) & (e > 1023)
        output_high = tl.where(overflow, sign | 0x7FF00000, output_high)
        rounded_low = tl.where(overflow, 0, rounded_low)
        output_high = tl.where(zero, sign, output_high)
        rounded_low = tl.where(zero, 0, rounded_low)
    return rounded_low.to(tl.int32, bitcast=True), output_high.to(tl.int32, bitcast=True)


@triton.jit
def normalize_scaled(a0, a1, a2, exponent):
    """Normalize a finite three-component value to |leading| in [0.5,1)."""
    a0, a1, a2 = renorm(a0, a1, a2)
    field = (a0.to(tl.int32, bitcast=True) >> 23) & 255
    adjustment = tl.where(a0 == 0.0, 0, field - 126)
    factor = ((127 - adjustment) << 23).to(tl.float32, bitcast=True)
    return a0 * factor, a1 * factor, a2 * factor, exponent + adjustment


@triton.jit
def add_scaled(a0, a1, a2, ae, b0, b1, b2, be):
    """Add normalized finite scaled triples, retaining approximately 72 bits.

    Components separated by over 100 exponent bits are below this format's
    retained precision. This is arithmetic rounding, not clipping a generated
    sample. Exact cancellation is handled before result normalization.
    """
    ae = tl.where(a0 == 0.0, be, ae)
    be = tl.where(b0 == 0.0, ae, be)
    common = tl.maximum(ae, be)
    da, db = common - ae, common - be
    sa = ((127 - tl.minimum(da, 100)) << 23).to(tl.float32, bitcast=True)
    sb = ((127 - tl.minimum(db, 100)) << 23).to(tl.float32, bitcast=True)
    sa = tl.where(da > 100, 0.0, sa)
    sb = tl.where(db > 100, 0.0, sb)
    c0, c1, c2 = add(a0 * sa, a1 * sa, a2 * sa, b0 * sb, b1 * sb, b2 * sb)
    return normalize_scaled(c0, c1, c2, common)


@triton.jit
def affine_scaled(z0, z1, z2, ze, mu0, mu1, mu2, mue, sigma0, sigma1, sigma2, sigmae):
    """Return mu + sigma*z from normalized finite scaled triples."""
    p0, p1, p2, pe = mul_scaled(z0, z1, z2, sigma0, sigma1, sigma2)
    pe = pe + ze + sigmae
    return add_scaled(p0, p1, p2, pe, mu0, mu1, mu2, mue)


@triton.jit
def exp_from_scaled(a0, a1, a2, exponent):
    """Exponentiate a normalized finite scaled input, with binary64 range.

    The active finite interval [-746,710] includes both binary64 exp rounding
    boundaries. Larger positive/negative inputs map to inf/zero when packed;
    inputs with magnitude below 2**-100 round to one in binary64.
    """
    e = tl.minimum(tl.maximum(exponent, -100), 10)
    scale = ((127 + e) << 23).to(tl.float32, bitcast=True)
    x0, x1, x2 = a0 * scale, a1 * scale, a2 * scale
    positive_overflow = ((exponent > 10) & (a0 > 0.0)) | (x0 > 710.0)
    negative_underflow = ((exponent > 10) & (a0 < 0.0)) | (x0 < -746.0)
    inactive = positive_overflow | negative_underflow | (exponent < -100)
    x0, x1, x2 = tl.where(inactive, 0.0, x0), tl.where(inactive, 0.0, x1), tl.where(inactive, 0.0, x2)
    m0, m1, m2, k = exp_scaled(x0, x1, x2)
    k = tl.where(positive_overflow, 1025, tl.where(negative_underflow, -1200, k))
    return m0, m1, m2, k
