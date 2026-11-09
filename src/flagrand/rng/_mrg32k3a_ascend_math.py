"""Unit-turn sin/cos for Ascend FP32 normal transforms.

Degree-seven sine and degree-eight cosine coefficients are from Stephen L.
Moshier's Cephes single/sinf.c, Release 2.2 (June 1992).
Copyright 1985, 1987, 1988, 1992 by Stephen L. Moshier.
Source: https://netlib.org/cephes/single.tgz
The author's free-use permission: https://netlib.org/cephes/readme

Cosine uses the equivalent Horner form. Turn reduction and the low-part
correction retain the existing FP32 transform. The F64 argument preserves
the active kernel's call signature; that kernel always passes False.
"""

import triton
import triton.language as tl


@triton.jit
def sincos_turns(u, F64: tl.constexpr):
    q = (u * 4.0 + 0.5).to(tl.int32)
    r = u - q.to(tl.float32) * 0.25
    hi = tl.full((), 6.283185307179586, tl.float32)
    lo = tl.full((), -1.748455600074497e-7, tl.float32)
    x = r * hi
    tail = tl.fma(r, hi, -x) + r * lo
    z = x * x
    sp = tl.full((), -1.9515295891e-4, tl.float32)
    sp = tl.fma(sp, z, tl.full((), 8.3321608736e-3, tl.float32))
    sp = tl.fma(sp, z, tl.full((), -1.6666654611e-1, tl.float32))
    sine = tl.fma(x * z, sp, x)
    cp = tl.full((), 2.443315711809948e-5, tl.float32)
    cp = tl.fma(cp, z, tl.full((), -1.388731625493765e-3, tl.float32))
    cp = tl.fma(cp, z, tl.full((), 4.166664568298827e-2, tl.float32))
    cp = tl.fma(cp, z, tl.full((), -0.5, tl.float32))
    cosine = tl.fma(cp, z, tl.full((), 1.0, tl.float32))
    st = tl.fma(tail, cosine, sine)
    ct = tl.fma(-tail, sine, cosine)
    swap = (q & 1) != 0
    s = tl.where(swap, ct, st)
    c = tl.where(swap, st, ct)
    s = tl.where((q & 2) != 0, -s, s)
    c = tl.where(((q + 1) & 2) != 0, -c, c)
    return s, c
