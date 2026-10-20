"""Constant-table gather candidate for Ascend software-f64 Horner loops."""

import triton
import triton.language as tl

from .soft64_math import COS_COEFF, EXP_COEFF, LOG_COEFF, SIN_COEFF


@triton.jit
def _polynomial_coefficient(k, KIND: tl.constexpr, COMPONENT: tl.constexpr):
    lane = tl.arange(0, 32)
    table = tl.full((32,), 0.0, tl.float32)
    if KIND == 0:
        for j in tl.static_range(0, 18):
            table = tl.where(lane == j, LOG_COEFF[j][COMPONENT], table)
    elif KIND == 1:
        for j in tl.static_range(0, 13):
            table = tl.where(lane == j, SIN_COEFF[j][COMPONENT], table)
    elif KIND == 2:
        for j in tl.static_range(0, 14):
            table = tl.where(lane == j, COS_COEFF[j][COMPONENT], table)
    else:
        for j in tl.static_range(0, 25):
            table = tl.where(lane == j, EXP_COEFF[j][COMPONENT], table)
    value = tl.gather(table, k + tl.arange(0, 1), axis=0)
    return tl.reshape(value, ())
