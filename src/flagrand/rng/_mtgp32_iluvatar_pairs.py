"""CoreX MTGP32 FP32 normal pairs with one adjacent raw pair per lane."""

import triton
import triton.language as tl

from flagrand.rng._mtgp32_data import MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from flagrand.rng._mtgp32_direct_pairs import _pair_linear
from flagrand.rng._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from flagrand.rng._mtgp32_linear_table import _load_linear_table_basis
from flagrand.rng._mtgp32_sequence import advance_to_block_start, ensure_working_state
from flagrand.rng._mtgp32_uniform_f32 import _uniform32
from flagrand.rng._xorwow.native_math import sincos_turns
from flagrand.rng.mtgp32 import _float_bits
from flagrand.runtime.compiled_launcher import CachedKernelLauncher


@triton.jit
def _pair_plain(
    STATE, PARAM, TEMPER, s_base, p_base, index, pos, sh1, sh2, STATE_MASK: tl.constexpr, MASK: tl.constexpr
):
    # Same six state reads as the existing _pair_linear. x1 is shared by
    # adjacent recurrences, and y0 is the tempering input for the second word.
    x0 = tl.load(STATE + s_base + (index & STATE_MASK)).to(tl.uint32, bitcast=True)
    x1 = tl.load(STATE + s_base + ((index + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    x2 = tl.load(STATE + s_base + ((index + 2) & STATE_MASK)).to(tl.uint32, bitcast=True)
    t = tl.load(STATE + s_base + ((index + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y0 = tl.load(STATE + s_base + ((index + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y1 = tl.load(STATE + s_base + ((index + pos + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    r0 = _mtgp32_recurrence(x0, x1, y0, sh1, sh2, MASK)
    r1 = _mtgp32_recurrence(x1, x2, y1, sh1, sh2, MASK)
    value_a = r0 ^ tl.load(PARAM + p_base + (r0 & 15)).to(tl.uint32, bitcast=True)
    value_b = r1 ^ tl.load(PARAM + p_base + (r1 & 15)).to(tl.uint32, bitcast=True)
    a = _mtgp32_temper(value_a, t, TEMPER + p_base)
    b = _mtgp32_temper(value_b, y0, TEMPER + p_base)
    return value_a, a, value_b, b


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_corex_pair_kernel(
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
    N_STATES: tl.constexpr,
    LOGNORMAL: tl.constexpr,
    LINEAR_TABLE: tl.constexpr,
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
    if LINEAR_TABLE:
        p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
        t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
    pair = tl.arange(0, BLOCK_SIZE // 2)
    mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
    sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        index = offset + 2 * pair
        if LINEAR_TABLE:
            value_a, a, value_b, b = _pair_linear(
                STATE,
                s_base,
                index,
                pos,
                sh1,
                sh2,
                p0,
                p1,
                p2,
                p3,
                t0,
                t1,
                t2,
                t3,
                STATE_MASK,
                MASK,
            )
        else:
            value_a, a, value_b, b = _pair_plain(
                STATE,
                PARAM,
                TEMPER,
                s_base,
                p_base,
                index,
                pos,
                sh1,
                sh2,
                STATE_MASK,
                MASK,
            )
        tl.store(STATE + s_base + ((index + N_RECUR) & STATE_MASK), value_a.to(tl.int32, bitcast=True))
        tl.store(STATE + s_base + ((index + 1 + N_RECUR) & STATE_MASK), value_b.to(tl.int32, bitcast=True))
        u1, u2 = _uniform32(a), _uniform32(b)
        radius = tl.sqrt(-2.0 * tl.log(u1))
        sine, cosine = sincos_turns(u2, False)
        x = mu + sigma * radius * cosine
        y = mu + sigma * radius * sine
        if LOGNORMAL:
            x = tl.exp(x)
            y = tl.exp(y)
        block = k.to(tl.int64) * N_STATES + p
        output_pair = block * (BLOCK_SIZE // 2) + pair
        tl.store(OUT + 2 * output_pair, x, 2 * output_pair < N)
        tl.store(OUT + 2 * output_pair + 1, y, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()


_launcher = CachedKernelLauncher(
    _mtgp32_corex_pair_kernel,
    constexpr_names=("N_STATES", "LOGNORMAL", "LINEAR_TABLE", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
    option_names=("num_warps", "enable_fp_fusion"),
)


def generate(engine, out, api="normal", *, mean=0.0, stddev=1.0, linear_table=False):
    """Explicit complete-256-word entry; the public partial path is unchanged."""
    n = out.numel()
    words = 2 * ((n + 1) // 2)
    device = str(out.device)
    ensure_working_state(engine, engine.seed, device)
    if engine._ws_next_block_start < engine.offset:
        # A direct seed change rebuilt the state while retaining its logical offset.
        advance_to_block_start(engine, engine.offset, device, 4)
    states, blocks = engine._working_state.shape[0], words // 256
    args = (
        out,
        engine._working_state,
        *engine._ws_params,
        engine.offset // 256,
        blocks,
        n,
        _float_bits(mean, False),
        _float_bits(stddev, False),
    )
    constants = (states, api == "lognormal", linear_table, 256, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
    _launcher.launch((min(blocks, states),), args, constants, (2, False))
    engine.offset += words
    engine._ws_next_block_start += words
    return out
