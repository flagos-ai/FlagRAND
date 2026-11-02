"""Distribution transforms over the existing MTGP32 raw-word sequence.

The binary64 math is shared with the project's XORWOW implementation. This
first integration keeps generation and transforms separate for state clarity.
"""

import math
from functools import lru_cache

import triton
import triton.language as tl

from ._xorwow._binding import bind_kernel
from ._xorwow.metax_math import sqrt_radial64
from ._xorwow.native_math import sincos_turns
from ._xorwow.range_math64 import exp64, exp64_metax, exp64_metax_finite, log_unit64


@lru_cache(maxsize=128)
def exp_finite_range(mean: float, stddev: float) -> bool:
    """Conservative exp domain for this 53-bit Box--Muller implementation.

    sqrt(106*ln(2)) is about 8.572; 9 retains numerical margin. Each host
    arithmetic step is rounded outward. NaN/Inf and larger parameter ranges
    select the existing general exp path.
    """
    spread = math.nextafter(9.0 * abs(stddev), math.inf)
    bound = math.nextafter(abs(mean) + spread, math.inf)
    return bound <= 700.0


@triton.jit
def _uniform64(a, b):
    # 53 independent bits, mapped to (0, 1]. Keep integer-to-f64 conversion
    # out of the path, matching the existing XORWOW bit construction.
    mantissa = ((a >> 5).to(tl.uint64) << 25) | (b >> 7).to(tl.uint64)
    bits = mantissa | tl.full((), 0x3FF0000000000000, tl.uint64)
    high = bits.to(tl.float64, bitcast=True) - tl.full((), 1.0, tl.float64)
    delta = tl.where(
        (b & 64) != 0,
        tl.full((), 2.220446049250313e-16, tl.float64),
        tl.full((), 1.1102230246251565e-16, tl.float64),
    )
    return high + delta


@triton.jit
def _uniform_kernel(RAW, OUT, N, F64: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if F64:
        a = tl.load(RAW + 2 * i, i < N, other=0).to(tl.uint32, bitcast=True)
        b = tl.load(RAW + 2 * i + 1, i < N, other=0).to(tl.uint32, bitcast=True)
        value = _uniform64(a, b)
    else:
        a = tl.load(RAW + i, i < N, other=0).to(tl.uint32, bitcast=True)
        value = ((a >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
    tl.store(OUT + i, value, i < N)


@lru_cache(maxsize=8)
def build_normal_kernel(
    native_log: bool = False,
    finite_exp: bool = False,
    ppu_fast_log: bool = False,
    *,
    ppu_native_log64: bool = False,
    ppu_native_exp64: bool = False,
):
    """Bind the backend math once for public normal dispatch.

    The PPU FP32 log, FP64 log and FP64 exp options are independent.
    Other backends retain log_unit64. The MetaX frontend exposes its native
    FP64 log through the CUDA libdevice compatibility module.
    """
    if native_log:
        from triton.language.extra.cuda import libdevice

        log64 = libdevice.log
        exp_function = exp64_metax_finite if finite_exp else exp64_metax
    else:
        log64 = log_unit64
        exp_function = exp64

    if ppu_native_log64:
        from ._mtgp32_ppu_math import _ppu_log64

        log64 = _ppu_log64

    if ppu_native_exp64:
        from ._mtgp32_ppu_math import _ppu_exp64

        exp_function = _ppu_exp64

    if ppu_fast_log:
        from ._mtgp32_ppu_math import _ppu_log32

        log32 = _ppu_log32
    else:
        log32 = tl.log

    def normal_kernel(
        RAW,
        OUT,
        N,
        MU_BITS,
        SIGMA_BITS,
        F64: tl.constexpr,
        LOGNORMAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = 2 * i < N
        if F64:
            a = tl.load(RAW + 4 * i, active, other=0).to(tl.uint32, bitcast=True)
            b = tl.load(RAW + 4 * i + 1, active, other=0).to(tl.uint32, bitcast=True)
            c = tl.load(RAW + 4 * i + 2, active, other=0).to(tl.uint32, bitcast=True)
            d = tl.load(RAW + 4 * i + 3, active, other=0).to(tl.uint32, bitcast=True)
            u1 = _uniform64(a, b)
            u2 = _uniform64(c, d)
            radius = sqrt_radial64(-2.0 * log64(u1))
            mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
            sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
        else:
            a = tl.load(RAW + 2 * i, active, other=0).to(tl.uint32, bitcast=True)
            b = tl.load(RAW + 2 * i + 1, active, other=0).to(tl.uint32, bitcast=True)
            u1 = ((a >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
            u2 = ((b >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
            radius = tl.sqrt(-2.0 * log32(u1))
            mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
            sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
        sine, cosine = sincos_turns(u2, F64)
        a = mu + sigma * radius * cosine
        b = mu + sigma * radius * sine
        if LOGNORMAL:
            if F64:
                a = exp_function(a)
                b = exp_function(b)
            else:
                a = tl.exp(a)
                b = tl.exp(b)
        tl.store(OUT + 2 * i, a, active)
        tl.store(OUT + 2 * i + 1, b, 2 * i + 1 < N)

    return bind_kernel(normal_kernel)


_normal_kernel = build_normal_kernel()
