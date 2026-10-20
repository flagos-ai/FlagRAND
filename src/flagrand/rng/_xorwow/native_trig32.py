"""Experimental FP32 sin/cos with the existing unit-turn range reduction.

Only the reduced-angle polynomial evaluations become tl.sin/tl.cos. The
integer generator and the number of random draws are outside this helper.
Floating-point output bits may differ from sincos_turns; backend support and
accuracy are measured by benchmark/check_mtgp32_trig.py.
"""

import triton
import triton.language as tl


@triton.jit
def sincos_turns_native32(u):
    q = (u * 4.0 + 0.5).to(tl.int32)
    r = u - q.to(tl.float32) * 0.25
    hi = tl.full((), 6.283185307179586, tl.float32)
    lo = tl.full((), -1.748455600074497e-7, tl.float32)
    x = r * hi
    tail = tl.fma(r, hi, -x) + r * lo
    sine = tl.sin(x)
    cosine = tl.cos(x)
    st = tl.fma(tail, cosine, sine)
    ct = tl.fma(-tail, sine, cosine)
    swap = (q & 1) != 0
    s = tl.where(swap, ct, st)
    c = tl.where(swap, st, ct)
    s = tl.where((q & 2) != 0, -s, s)
    c = tl.where(((q + 1) & 2) != 0, -c, c)
    return s, c
