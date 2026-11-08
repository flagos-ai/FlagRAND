"""Joint sine/cosine for the existing FP32 MRG angle in [0, float32(2*pi)].

The polynomial coefficients are reciprocals of factorials, as in the repository
unit-turn helper. The input remains the existing rounded radian angle.
"""

import triton
import triton.language as tl


@triton.jit
def sincos_bounded(angle):
    # Known MRG domain: 0 < angle <= float32(2*pi), hence quadrant is 0..4.
    quadrant = (angle * 0.6366197723675814 + 0.5).to(tl.int32)
    q = quadrant.to(tl.float32)
    x = tl.fma(q, -1.570796251296997, angle)
    x = tl.fma(q, -7.549789415861596e-8, x)
    x = tl.fma(q, -5.390302529957765e-15, x)
    z = x * x
    sp = tl.full((), -2.505210838544172e-8, tl.float32)
    sp = tl.fma(sp, z, 2.7557319223985893e-6)
    sp = tl.fma(sp, z, -0.0001984126984126984)
    sp = tl.fma(sp, z, 0.008333333333333333)
    sp = tl.fma(sp, z, -0.16666666666666666)
    sine = tl.fma(x * z, sp, x)
    cp = tl.full((), -2.755731922398589e-7, tl.float32)
    cp = tl.fma(cp, z, 2.48015873015873e-5)
    cp = tl.fma(cp, z, -0.001388888888888889)
    cp = tl.fma(cp, z, 0.041666666666666664)
    cp = tl.fma(cp, z, -0.5)
    cosine = tl.fma(cp, z, 1.0)
    swap = (quadrant & 1) != 0
    s = tl.where(swap, cosine, sine)
    c = tl.where(swap, sine, cosine)
    s = tl.where((quadrant & 2) != 0, -s, s)
    c = tl.where(((quadrant + 1) & 2) != 0, -c, c)
    return s, c
