"""Direct binary32 significand splitting for the Ascend soft64 candidate.

This omits the exponent normalization in two_prod_split to reduce compiled
IR.  Partial-product underflow is possible, so applicability is determined
from the actual soft64 operand/output comparisons, including subnormals.
Launch with automatic contraction and reassociation disabled.
"""

import triton
import triton.language as tl


@triton.jit
def two_prod_direct(a, b):
    p = a * b
    ah = (a.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    bh = (b.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    al = a - ah
    bl = b - bh
    t = p - ah * bh
    t = t - al * bh
    t = t - ah * bl
    e = al * bl - t
    return p, e
