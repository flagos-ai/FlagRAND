"""PPU math helpers whose JIT identities distinguish backend implementations."""

import triton
from triton.language.extra.ppu import libdevice as ppu_libdevice


@triton.jit
def _ppu_log32(value):
    return ppu_libdevice.fast_logf(value)


@triton.jit
def _ppu_log64(value):
    return ppu_libdevice.log(value)


@triton.jit
def _ppu_exp64(value):
    return ppu_libdevice.exp(value)
