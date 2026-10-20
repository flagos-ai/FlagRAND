"""Single-change MTGP32 candidate: static FP64 raw-word unpacking.

Compared with _mtgp32_fused_kernel, only the four fixed-stride gathers in
FP64 normal/lognormal become reshape+split. The 64-pair loop is unchanged.
This candidate is not used by the public generator.
"""

import triton
import triton.language as tl

from ._mtgp32_fused import (
    _mtgp32_recurrence,
    _mtgp32_temper,
    _sample_alias,
    _uniform32,
    _uniform64,
    exp64,
    log_unit64,
    sincos_turns,
    sqrt_radial64,
)


@triton.jit
def _split_f64_words(word, BLOCK_SIZE: tl.constexpr):
    # Row-major [pair, 2, 2] contains [[a,b],[c,d]]. split removes the
    # final dimension, first producing [a,c] and [b,d], then four vectors.
    packed = tl.reshape(word, (BLOCK_SIZE // 4, 2, 2))
    ac, bd = tl.split(packed)
    a, c = tl.split(ac)
    b, d = tl.split(bd)
    return a, b, c, d


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_split_kernel(
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

    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(STATE + s_base + ((lane + offset) & STATE_MASK)).to(tl.uint32, bitcast=True)
        x2 = tl.load(STATE + s_base + ((lane + offset + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = tl.load(STATE + s_base + ((lane + offset + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        matrix = tl.load(PARAM + p_base + (y & 15)).to(tl.uint32, bitcast=True)
        value = y ^ matrix
        t = tl.load(STATE + s_base + ((lane + offset + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        word = _mtgp32_temper(value, t, TEMPER + p_base)
        tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
        block = k.to(tl.int64) * N_STATES + p

        if MODE == 1:
            if F64:
                j = tl.arange(0, BLOCK_SIZE // 2)
                a = tl.gather(word, 2 * j, 0)
                b = tl.gather(word, 2 * j + 1, 0)
                tl.store(OUT + block * (BLOCK_SIZE // 2) + j, _uniform64(a, b))
            else:
                tl.store(OUT + block * BLOCK_SIZE + lane, _uniform32(word))
        elif MODE == 4:
            j = tl.arange(0, BLOCK_SIZE // 2)
            a = tl.gather(word, 2 * j, 0)
            b = tl.gather(word, 2 * j + 1, 0)
            result = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
            tl.store(OUT + block * (BLOCK_SIZE // 2) + j, result)
        else:
            if F64:
                j = tl.arange(0, BLOCK_SIZE // 4)
                a, b, c, d = _split_f64_words(word, BLOCK_SIZE)
                u1 = _uniform64(a, b)
                u2 = _uniform64(c, d)
                radius = sqrt_radial64(-2.0 * log_unit64(u1))
                mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 4) + j
            else:
                j = tl.arange(0, BLOCK_SIZE // 2)
                a = tl.gather(word, 2 * j, 0)
                b = tl.gather(word, 2 * j + 1, 0)
                u1 = _uniform32(a)
                u2 = _uniform32(b)
                radius = tl.sqrt(-2.0 * tl.log(u1))
                mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 2) + j
            sine, cosine = sincos_turns(u2, F64)
            a = mu + sigma * radius * cosine
            b = mu + sigma * radius * sine
            if MODE == 3:
                if F64:
                    a = exp64(a)
                    b = exp64(b)
                else:
                    a = tl.exp(a)
                    b = tl.exp(b)
            tl.store(OUT + 2 * output_pair, a, 2 * output_pair < N)
            tl.store(OUT + 2 * output_pair + 1, b, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()
