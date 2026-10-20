"""Joint sin/cos for the known unit-turn input, using native f32/f64 FMA.

The polynomial coefficients below are reciprocals of factorials, not copied
device-library code. Device measurements compare this path with the original
vendor libdevice. It does not alter the underlying integer state or draws.
"""

import triton
import triton.language as tl


@triton.jit
def hygon_sincos_turns32(u):
    # AMDGCN sin/cos instruction input is measured in turns. This entry is
    # selected only for HCU; the input here is already in [0, 1).
    # https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20200601/788834.html
    return tl.inline_asm_elementwise(
        "v_sin_f32 $0, $2\n\tv_cos_f32 $1, $2",
        constraints="=&v,=&v,v",
        args=[u],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def sincos_turns(u, F64: tl.constexpr):
    DT: tl.constexpr = tl.float64 if F64 else tl.float32
    q = (u * 4.0 + 0.5).to(tl.int32)
    r = u - q.to(DT) * 0.25
    hi = tl.full((), 6.283185307179586, DT)
    if F64:
        lo = tl.full((), 2.4492935982947064e-16, DT)
    else:
        lo = tl.full((), -1.748455600074497e-7, DT)
    x = r * hi
    tail = tl.fma(r, hi, -x) + r * lo
    z = x * x
    if F64:
        sp = tl.full((), 2.8114572543455206e-15, DT)  # 1/17!
        sp = tl.fma(sp, z, tl.full((), -7.647163731819816e-13, DT))
        sp = tl.fma(sp, z, tl.full((), 1.6059043836821613e-10, DT))
        sp = tl.fma(sp, z, tl.full((), -2.505210838544172e-8, DT))
        cp = tl.full((), -1.5619206968586225e-16, DT)  # -1/18!
        cp = tl.fma(cp, z, tl.full((), 4.779477332387385e-14, DT))
        cp = tl.fma(cp, z, tl.full((), -1.1470745597729725e-11, DT))
        cp = tl.fma(cp, z, tl.full((), 2.08767569878681e-9, DT))
        cp = tl.fma(cp, z, tl.full((), -2.755731922398589e-7, DT))
    else:
        sp = tl.full((), -2.505210838544172e-8, DT)  # -1/11!
        cp = tl.full((), -2.755731922398589e-7, DT)  # -1/10!
    sp = tl.fma(sp, z, tl.full((), 2.7557319223985893e-6, DT))
    sp = tl.fma(sp, z, tl.full((), -0.0001984126984126984, DT))
    sp = tl.fma(sp, z, tl.full((), 0.008333333333333333, DT))
    sp = tl.fma(sp, z, tl.full((), -0.16666666666666666, DT))
    sine = tl.fma(x * z, sp, x)
    cp = tl.fma(cp, z, tl.full((), 2.48015873015873e-5, DT))
    cp = tl.fma(cp, z, tl.full((), -0.001388888888888889, DT))
    cp = tl.fma(cp, z, tl.full((), 0.041666666666666664, DT))
    cp = tl.fma(cp, z, tl.full((), -0.5, DT))
    cosine = tl.fma(cp, z, tl.full((), 1.0, DT))
    st = tl.fma(tail, cosine, sine)
    ct = tl.fma(-tail, sine, cosine)
    swap = (q & 1) != 0
    s = tl.where(swap, ct, st)
    c = tl.where(swap, st, ct)
    s = tl.where((q & 2) != 0, -s, s)
    c = tl.where(((q + 1) & 2) != 0, -c, c)
    return s, c
