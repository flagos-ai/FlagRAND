"""C550 binary64 math candidates for the XORWOW fused kernels.

sqrt_radial64 accepts the radial input -2*log(k/2**53), 1 <= k <= 2**53:
zero or a positive normal value between about 2**-52 and 106*ln(2).
Only the reciprocal-square-root seed uses f32. Both residual corrections use
explicit f64 FMA. Callers keep enable_fp_fusion=False for other arithmetic.

With relative seed error e, the first correction removes its linear term;
the second leaves a leading O(e**3) seed contribution before f64 rounding.
This requires a sufficiently accurate f32 rsqrt seed. The paired device
precision probe records the actual backend's result; no correct-rounding
claim over the complete domain is made.

Local validation executed this function body with exact RN-even f64 FMA and
4,579 Decimal80 reference inputs (sqrt_precision_measure, samples=4096).
Rounded f32 seeds shifted by -2/0/+2 f32 ULP gave maximum errors 0/0/1 f64 ULP;
both signed zero inputs returned +0. These are CPU seed-sensitivity samples,
not observations of the hardware rsqrt instruction.
"""

import triton
import triton.language as tl


@triton.jit
def sqrt_radial64(x):
    seed_x = tl.where(x == 0.0, tl.full((), 1.0, tl.float64), x)
    reciprocal = tl.cast(tl.rsqrt(tl.cast(seed_x, tl.float32)), tl.float64)
    half_reciprocal = reciprocal * 0.5
    root = x * reciprocal
    for _ in tl.static_range(2):
        residual = tl.fma(-root, root, x)
        root = tl.fma(residual, half_reciprocal, root)
    return tl.where(x == 0.0, tl.full((), 0.0, tl.float64), root)
