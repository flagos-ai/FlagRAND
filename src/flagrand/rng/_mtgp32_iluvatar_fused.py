"""CoreX complete-block FP32 transforms using adjacent-word reshape/split.

The installed CoreX Triton 3.1 frontend has no tl.gather. Each MTGP32
256-word block is reshaped to 128 adjacent pairs without changing its order.
"""

import triton
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.runtime.compiled_launcher import CachedKernelLauncher

from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear
from ._mtgp32_uniform_f32 import _uniform32
from ._xorwow.native_math import sincos_turns


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_iluvatar_fused_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START_BLOCK,
    BLOCK_COUNT,
    N,
    MU_BITS,
    SIGMA_BITS,
    THRESHOLD,
    ALIASES,
    N_STATES: tl.constexpr,
    MODE: tl.constexpr,
    F64: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    LINEAR_TABLE: tl.constexpr,
):
    p = tl.program_id(0)
    first_state = (START_BLOCK % N_STATES).to(tl.int32)
    state_id = (first_state + p) % N_STATES
    first_round = ((START_BLOCK // N_STATES) & 3).to(tl.int32)
    first_round += (first_state + p) // N_STATES
    iterations = tl.cdiv(BLOCK_COUNT - p, N_STATES)
    pos = tl.load(POS + state_id)
    sh1 = tl.load(SH1 + state_id)
    sh2 = tl.load(SH2 + state_id)
    s_base = state_id * (STATE_MASK + 1)
    p_base = state_id * 16
    lane = tl.arange(0, BLOCK_SIZE)
    if LINEAR_TABLE:
        p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
        t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)

    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(STATE + s_base + ((lane + offset) & STATE_MASK)).to(tl.uint32, bitcast=True)
        x2 = tl.load(STATE + s_base + ((lane + offset + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = tl.load(STATE + s_base + ((lane + offset + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        if LINEAR_TABLE:
            matrix = _linear_table_value(y, p0, p1, p2, p3)
        else:
            matrix = tl.load(PARAM + p_base + (y & 15)).to(tl.uint32, bitcast=True)
        value = y ^ matrix
        t = tl.load(STATE + s_base + ((lane + offset + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        if LINEAR_TABLE:
            word = _temper_linear(value, t, t0, t1, t2, t3)
        else:
            word = _mtgp32_temper(value, t, TEMPER + p_base)
        tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
        block = k.to(tl.int64) * N_STATES + p

        if MODE == 1:
            tl.store(OUT + block * BLOCK_SIZE + lane, _uniform32(word))
        else:
            j = tl.arange(0, BLOCK_SIZE // 2)
            a, b = tl.split(tl.reshape(word, (BLOCK_SIZE // 2, 2)))
            output_pair = block * (BLOCK_SIZE // 2) + j
            if MODE == 4:
                result = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
                tl.store(OUT + output_pair, result)
            else:
                u1 = _uniform32(a)
                u2 = _uniform32(b)
                radius = tl.sqrt(-2.0 * tl.log(u1))
                mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sine, cosine = sincos_turns(u2, False)
                a = mu + sigma * radius * cosine
                b = mu + sigma * radius * sine
                if MODE == 3:
                    a = tl.exp(a)
                    b = tl.exp(b)
                tl.store(OUT + 2 * output_pair, a, 2 * output_pair < N)
                tl.store(OUT + 2 * output_pair + 1, b, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()


_iluvatar_fused_launcher = CachedKernelLauncher(
    _mtgp32_iluvatar_fused_kernel,
    constexpr_names=(
        "N_STATES",
        "MODE",
        "F64",
        "FIRST",
        "SIZE",
        "BLOCK_SIZE",
        "STATE_MASK",
        "MASK",
        "N_RECUR",
        "LINEAR_TABLE",
    ),
    option_names=("num_warps", "enable_fp_fusion"),
)
