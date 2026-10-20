"""MUSA pi-scaled trigonometric candidates using the installed libdevice."""

import triton
from triton.language import core

SINPI32_TYPES: core.constexpr = {(core.float32,): ("__mt_sinpi_f32", core.float32)}
COSPI32_TYPES: core.constexpr = {(core.float32,): ("__mt_cospi_f32", core.float32)}
EXTERN_LIBRARY: core.constexpr = ""
EXTERN_PURE: core.constexpr = True


@triton.jit
def sinpi32(x):
    return core.extern_elementwise(
        EXTERN_LIBRARY,
        EXTERN_LIBRARY,
        [x],
        SINPI32_TYPES,
        is_pure=EXTERN_PURE,
    )


@triton.jit
def cospi32(x):
    # libdevice.31.bc defines this symbol; the installed Python wrapper omits it.
    return core.extern_elementwise(
        EXTERN_LIBRARY,
        EXTERN_LIBRARY,
        [x],
        COSPI32_TYPES,
        is_pure=EXTERN_PURE,
    )


@triton.jit
def sincos_turns32(u):
    twice_turns = u * 2.0
    return sinpi32(twice_turns), cospi32(twice_turns)
