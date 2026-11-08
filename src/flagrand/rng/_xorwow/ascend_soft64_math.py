"""Ascend binding for software-f64 products without a fused FMA operation.

The observed BiSheng SIMD lowering expands math.fma to vmul + vadd.  This
module substitutes only two_prod; the existing three-component arithmetic
and packed binary64 kernels keep their source and their other backends.
Automatic contraction and reassociation must remain disabled at launch.
"""

from functools import lru_cache
from types import FunctionType, SimpleNamespace

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction


@triton.jit
def _product_operand(a):
    bits = a.to(tl.int32, bitcast=True)
    subnormal = ((bits >> 23) & 255) == 0
    scaled = a * tl.where(subnormal, 16777216.0, 1.0)
    bits = scaled.to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127 - tl.where(subnormal, 24, 0).to(tl.int32)
    mantissa = ((bits & -2139095041) | 1065353216).to(tl.float32, bitcast=True)
    return mantissa, exponent


@triton.jit
def two_prod_split(a, b):
    """Finite f32 product and its RN-even f32 residual, without FMA.

    Normalize before the 12/12-bit split so partial products cannot underflow
    or overflow.  One final power-of-two multiplication rounds the residual.
    A product in the first normal binade or below has a residual smaller
    than/equal to half the minimum subnormal, hence rounds to zero.  Its zero
    residual is canonical +0; the product retains its ordinary IEEE sign.
    Inputs/products are finite, as in the existing scaled soft64 interfaces.
    """
    p = a * b
    an, ae = _product_operand(a)
    bn, be = _product_operand(b)
    ah = (an.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    bh = (bn.to(tl.int32, bitcast=True) & -4096).to(tl.float32, bitcast=True)
    al = an - ah
    bl = bn - bh
    pn = an * bn
    t = pn - ah * bh
    t = t - al * bh
    t = t - ah * bl
    residual = al * bl - t
    normal_residual = ((p.to(tl.int32, bitcast=True) >> 23) & 255) > 1
    exponent = tl.where(normal_residual, ae + be, 0)
    scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    e = tl.where(normal_residual, residual * scale, 0.0)
    return p, e


@lru_cache(maxsize=8)
def bind_ascend_soft64(runtime_loops=False, direct_product=False, gather_coefficients=False):
    """Return JITs sharing an Ascend-only globals dictionary.

    FunctionType retains the existing source/signatures, including packed
    stores; globals are copied, never changed in the original modules.
    """
    from . import soft64_kernels, soft64_math

    modules = (soft64_math, soft64_kernels)
    if runtime_loops:
        from . import ascend_soft64_loop_math

        modules += (ascend_soft64_loop_math,)
    if gather_coefficients:
        from . import ascend_soft64_coefficients

        modules += (ascend_soft64_coefficients,)
    namespace = {}
    for module in modules:
        namespace.update(vars(module))
    # Triton-Ascend resolves annotated global tuples directly. Wrapping a
    # coefficient tuple in tl.constexpr makes nested indexing fail in 3.2.
    namespace["__annotations__"] = soft64_math.__annotations__
    namespace["two_prod"] = two_prod_split
    if direct_product:
        from .ascend_simple_product import two_prod_direct

        namespace["two_prod"] = two_prod_direct
    for module in modules:
        for name, original in vars(module).items():
            if isinstance(original, JITFunction) and original.fn.__module__ == module.__name__ and name != "two_prod":
                function = original.fn
                bound = FunctionType(
                    function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
                )
                bound.__annotations__ = function.__annotations__
                bound.__kwdefaults__ = function.__kwdefaults__
                bound.__qualname__ = function.__qualname__
                bound.__module__ = function.__module__
                namespace[name] = triton.jit(
                    bound, noinline=original.noinline, do_not_specialize=original.do_not_specialize
                )
    return SimpleNamespace(**namespace)
