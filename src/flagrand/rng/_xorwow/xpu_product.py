"""XPU product rounding with an explicit binary32 multiply instruction."""

import triton
import triton.language as tl


@triton.jit
def two_prod_inline(a, b):
    p = tl.inline_asm_elementwise(
        "mul.f.rn $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    e = tl.fma(a, b, -p)
    return p, e
