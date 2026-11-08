"""Three-float arithmetic for bounded standard random distributions on Ascend.

Dekker splitting avoids integer bit masks in the polynomial loops. Its
splitter can overflow for unrestricted inputs, so callers use this binding
only for the bounded standard distributions and their rational uniforms.
Finite user affine parameters retain the existing scaled math binding.
Automatic FP contraction and reassociation must remain disabled at launch.
"""

from functools import lru_cache
from types import FunctionType, SimpleNamespace

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from .soft64_math import COS_COEFF, EXP_COEFF, LN2, LOG_COEFF, SIN_COEFF, add, mul, sub


@triton.jit
def _polynomial_coefficient(k, KIND: tl.constexpr, COMPONENT: tl.constexpr):
    if KIND == 0:
        value = tl.full((), LOG_COEFF[0][COMPONENT], tl.float32)
        for j in tl.static_range(1, 12):
            value = tl.where(k == j, tl.full((), LOG_COEFF[j][COMPONENT], tl.float32), value)
    elif KIND == 1:
        value = tl.full((), SIN_COEFF[0][COMPONENT], tl.float32)
        for j in tl.static_range(1, 10):
            value = tl.where(k == j, tl.full((), SIN_COEFF[j][COMPONENT], tl.float32), value)
    elif KIND == 2:
        value = tl.full((), COS_COEFF[0][COMPONENT], tl.float32)
        for j in tl.static_range(1, 10):
            value = tl.where(k == j, tl.full((), COS_COEFF[j][COMPONENT], tl.float32), value)
    else:
        value = tl.full((), EXP_COEFF[0][COMPONENT], tl.float32)
        for j in tl.static_range(1, 17):
            value = tl.where(k == j, tl.full((), EXP_COEFF[j][COMPONENT], tl.float32), value)
    return value


@triton.jit
def div(a0, a1, a2, b0, b1, b2):
    z = tl.full(a0.shape, 0.0, tl.float32)
    q0 = a0 / b0
    q1 = z
    q2 = z
    for _ in tl.range(0, 3, loop_unroll_factor=1):
        p0, p1, p2 = mul(b0, b1, b2, q0, q1, q2)
        r0, r1, r2 = sub(a0, a1, a2, p0, p1, p2)
        c = r0 / b0
        q0, q1, q2 = add(q0, q1, q2, c, z, z)
    return q0, q1, q2


@triton.jit
def log(a0, a1, a2):
    z = tl.full(a0.shape, 0.0, tl.float32)
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
    p0 = tl.full(a0.shape, LOG_COEFF[11][0], tl.float32)
    p1 = tl.full(a0.shape, LOG_COEFF[11][1], tl.float32)
    p2 = tl.full(a0.shape, LOG_COEFF[11][2], tl.float32)
    for i in tl.range(0, 11, loop_unroll_factor=1):
        k = 10 - i
        p0, p1, p2 = mul(p0, p1, p2, v0, v1, v2)
        p0, p1, p2 = add(
            p0,
            p1,
            p2,
            _polynomial_coefficient(k, 0, 0),
            _polynomial_coefficient(k, 0, 1),
            _polynomial_coefficient(k, 0, 2),
        )
    p0, p1, p2 = mul(p0, p1, p2, y0 * 2.0, y1 * 2.0, y2 * 2.0)
    q0, q1, q2 = mul(
        e.to(tl.float32),
        z,
        z,
        tl.full(a0.shape, LN2[0], tl.float32),
        tl.full(a0.shape, LN2[1], tl.float32),
        tl.full(a0.shape, LN2[2], tl.float32),
    )
    return add(p0, p1, p2, q0, q1, q2)


@triton.jit
def sqrt(a0, a1, a2):
    z = tl.full(a0.shape, 0.0, tl.float32)
    nonzero = a0 != 0.0
    x0 = tl.where(nonzero, a0, 1.0)
    x1 = tl.where(nonzero, a1, 0.0)
    x2 = tl.where(nonzero, a2, 0.0)
    y0, y1, y2 = tl.sqrt(x0), z, z
    for _ in tl.range(0, 2, loop_unroll_factor=1):
        p0, p1, p2 = mul(y0, y1, y2, y0, y1, y2)
        r0, r1, r2 = sub(x0, x1, x2, p0, p1, p2)
        d0, d1, d2 = r0 / (y0 * 2.0), z, z
        y0, y1, y2 = add(y0, y1, y2, d0, d1, d2)
    return tl.where(nonzero, y0, 0.0), tl.where(nonzero, y1, 0.0), tl.where(nonzero, y2, 0.0)


@triton.jit
def sincos_reduced(a0, a1, a2):
    v0, v1, v2 = mul(a0, a1, a2, a0, a1, a2)
    s0 = tl.full(a0.shape, SIN_COEFF[9][0], tl.float32)
    s1 = tl.full(a0.shape, SIN_COEFF[9][1], tl.float32)
    s2 = tl.full(a0.shape, SIN_COEFF[9][2], tl.float32)
    for i in tl.range(0, 9, loop_unroll_factor=1):
        k = 8 - i
        s0, s1, s2 = mul(s0, s1, s2, v0, v1, v2)
        s0, s1, s2 = add(
            s0,
            s1,
            s2,
            _polynomial_coefficient(k, 1, 0),
            _polynomial_coefficient(k, 1, 1),
            _polynomial_coefficient(k, 1, 2),
        )
    s0, s1, s2 = mul(s0, s1, s2, a0, a1, a2)
    c0 = tl.full(a0.shape, COS_COEFF[9][0], tl.float32)
    c1 = tl.full(a0.shape, COS_COEFF[9][1], tl.float32)
    c2 = tl.full(a0.shape, COS_COEFF[9][2], tl.float32)
    for i in tl.range(0, 9, loop_unroll_factor=1):
        k = 8 - i
        c0, c1, c2 = mul(c0, c1, c2, v0, v1, v2)
        c0, c1, c2 = add(
            c0,
            c1,
            c2,
            _polynomial_coefficient(k, 2, 0),
            _polynomial_coefficient(k, 2, 1),
            _polynomial_coefficient(k, 2, 2),
        )
    return s0, s1, s2, c0, c1, c2


@triton.jit
def exp_scaled(a0, a1, a2):
    z = tl.full(a0.shape, 0.0, tl.float32)
    k = tl.floor(a0 * 1.4426950408889634 + 0.5).to(tl.int32)
    q0, q1, q2 = mul(
        k.to(tl.float32),
        z,
        z,
        tl.full(a0.shape, LN2[0], tl.float32),
        tl.full(a0.shape, LN2[1], tl.float32),
        tl.full(a0.shape, LN2[2], tl.float32),
    )
    r0, r1, r2 = sub(a0, a1, a2, q0, q1, q2)
    p0 = tl.full(a0.shape, EXP_COEFF[16][0], tl.float32)
    p1 = tl.full(a0.shape, EXP_COEFF[16][1], tl.float32)
    p2 = tl.full(a0.shape, EXP_COEFF[16][2], tl.float32)
    for i in tl.range(0, 16, loop_unroll_factor=1):
        j = 15 - i
        p0, p1, p2 = mul(p0, p1, p2, r0, r1, r2)
        p0, p1, p2 = add(
            p0,
            p1,
            p2,
            _polynomial_coefficient(j, 3, 0),
            _polynomial_coefficient(j, 3, 1),
            _polynomial_coefficient(j, 3, 2),
        )
    return p0, p1, p2, k


@triton.jit
def two_prod_dekker(a, b):
    p = a * b
    ca = 4097.0 * a
    ah = ca - (ca - a)
    al = a - ah
    cb = 4097.0 * b
    bh = cb - (cb - b)
    bl = b - bh
    t = p - ah * bh
    t = t - al * bh
    t = t - ah * bl
    e = al * bl - t
    return p, e


@lru_cache(maxsize=1)
def bind_bounded_math():
    from . import ascend_soft64_loop_math, soft64_kernels, soft64_math

    namespace, originals = {}, {}
    for module in (soft64_math, soft64_kernels, ascend_soft64_loop_math):
        namespace.update(vars(module))
        for name, value in vars(module).items():
            if isinstance(value, JITFunction) and value.fn.__module__ == module.__name__:
                originals[name] = value
    for name in ("_polynomial_coefficient", "log", "sincos_reduced", "exp_scaled", "sqrt"):
        originals[name] = globals()[name]
    namespace["__annotations__"] = soft64_math.__annotations__
    namespace["two_prod"] = two_prod_dekker
    for name, original in originals.items():
        if name == "two_prod":
            continue
        fn = original.fn
        bound = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
        bound.__annotations__, bound.__kwdefaults__ = fn.__annotations__, fn.__kwdefaults__
        bound.__qualname__, bound.__module__ = fn.__qualname__, fn.__module__
        namespace[name] = triton.jit(bound, noinline=original.noinline, do_not_specialize=original.do_not_specialize)
    return SimpleNamespace(**namespace)
