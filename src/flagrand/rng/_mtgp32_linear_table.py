"""Four-basis form of the MTGP32 GF(2) recurrence and tempering tables."""

import triton
import triton.language as tl


@triton.jit
def _load_linear_table_basis(TABLE, base):
    basis0 = tl.load(TABLE + base + 1).to(tl.uint32, bitcast=True)
    basis1 = tl.load(TABLE + base + 2).to(tl.uint32, bitcast=True)
    basis2 = tl.load(TABLE + base + 4).to(tl.uint32, bitcast=True)
    basis3 = tl.load(TABLE + base + 8).to(tl.uint32, bitcast=True)
    return basis0, basis1, basis2, basis3


@triton.jit
def _linear_table_value(index, basis0, basis1, basis2, basis3):
    return (
        tl.where((index & 1) != 0, basis0, 0)
        ^ tl.where((index & 2) != 0, basis1, 0)
        ^ tl.where((index & 4) != 0, basis2, 0)
        ^ tl.where((index & 8) != 0, basis3, 0)
    )


@triton.jit
def _temper_linear(value, t, basis0, basis1, basis2, basis3):
    t = t ^ (t >> 16)
    t = t ^ (t >> 8)
    return value ^ _linear_table_value(t, basis0, basis1, basis2, basis3)
