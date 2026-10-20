"""C550 FP64 normal/lognormal fusion using four 64-lane word groups.

The split control and direct candidate share native FP64 log and one warp.
Public dispatch selects the direct path for aligned calls up to 65536 outputs.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from ._mtgp32_sequence import ensure_working_state
from ._mtgp32_split import _split_f64_words
from ._mtgp32_transforms import _uniform64, exp_finite_range
from ._xorwow.metax_math import sqrt_radial64
from ._xorwow.native_math import sincos_turns
from ._xorwow.range_math64 import exp64_metax, exp64_metax_finite
from .mtgp32 import _float_bits


@triton.jit
def _generate_words(
    STATE,
    PARAM,
    TEMPER,
    s_base,
    p_base,
    index,
    pos,
    sh1,
    sh2,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    x1 = tl.load(STATE + s_base + (index & STATE_MASK)).to(tl.uint32, bitcast=True)
    x2 = tl.load(STATE + s_base + ((index + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y = tl.load(STATE + s_base + ((index + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
    matrix = tl.load(PARAM + p_base + (y & 15)).to(tl.uint32, bitcast=True)
    value = y ^ matrix
    t = tl.load(STATE + s_base + ((index + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    word = _mtgp32_temper(value, t, TEMPER + p_base)
    tl.store(STATE + s_base + ((index + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
    return word


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_direct_words_kernel(
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
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    DIRECT: tl.constexpr,
    LOGNORMAL: tl.constexpr,
    FINITE_EXP: tl.constexpr = False,
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
    pair = tl.arange(0, BLOCK_SIZE // 4)
    mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
    sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)

    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        if DIRECT:
            # Old inputs lie in offset+[0,350], new state in offset+[351,606].
            # These disjoint regions remain disjoint modulo the 1024-word ring.
            a = _generate_words(
                STATE, PARAM, TEMPER, s_base, p_base, offset + 4 * pair, pos, sh1, sh2, STATE_MASK, MASK, N_RECUR
            )
            b = _generate_words(
                STATE, PARAM, TEMPER, s_base, p_base, offset + 4 * pair + 1, pos, sh1, sh2, STATE_MASK, MASK, N_RECUR
            )
            c = _generate_words(
                STATE, PARAM, TEMPER, s_base, p_base, offset + 4 * pair + 2, pos, sh1, sh2, STATE_MASK, MASK, N_RECUR
            )
            d = _generate_words(
                STATE, PARAM, TEMPER, s_base, p_base, offset + 4 * pair + 3, pos, sh1, sh2, STATE_MASK, MASK, N_RECUR
            )
        else:
            word = _generate_words(
                STATE,
                PARAM,
                TEMPER,
                s_base,
                p_base,
                offset + tl.arange(0, BLOCK_SIZE),
                pos,
                sh1,
                sh2,
                STATE_MASK,
                MASK,
                N_RECUR,
            )
            a, b, c, d = _split_f64_words(word, BLOCK_SIZE)
        u1 = _uniform64(a, b)
        u2 = _uniform64(c, d)
        radius = sqrt_radial64(-2.0 * libdevice.log(u1))
        sine, cosine = sincos_turns(u2, True)
        a = mu + sigma * radius * cosine
        b = mu + sigma * radius * sine
        if LOGNORMAL:
            if FINITE_EXP:
                a = exp64_metax_finite(a)
                b = exp64_metax_finite(b)
            else:
                a = exp64_metax(a)
                b = exp64_metax(b)
        block = k.to(tl.int64) * N_STATES + p
        output_pair = block * (BLOCK_SIZE // 4) + pair
        tl.store(OUT + 2 * output_pair, a, 2 * output_pair < N)
        tl.store(OUT + 2 * output_pair + 1, b, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()


_launcher = CachedKernelLauncher(
    _mtgp32_direct_words_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR", "DIRECT", "LOGNORMAL", "FINITE_EXP"),
    option_names=("num_warps", "enable_fp_fusion"),
)


def generate_direct_words(
    engine,
    out,
    *,
    direct=True,
    mean=0.0,
    stddev=1.0,
    lognormal=False,
    keyed_launch=True,
    cached_launch=True,
    cache_tensor_signature=True,
):
    """Generate an aligned call consuming whole 256-word blocks."""
    device = str(out.device)
    ensure_working_state(engine, engine.seed, device)
    pos, sh1, sh2, param, temper = engine._ws_params
    n = out.numel()
    words = 4 * ((n + 1) // 2)
    blocks = words // MTGP32_BLOCK_SIZE
    states = engine.state.shape[0]
    arguments = (
        out,
        engine.state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        engine.offset // MTGP32_BLOCK_SIZE,
        blocks,
        n,
        _float_bits(mean, True),
        _float_bits(stddev, True),
    )
    finite_exp = lognormal and exp_finite_range(mean, stddev)
    constants = (states, MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N, direct, lognormal, finite_exp)
    options = (1, False)
    if cached_launch:
        tensor_signatures = (
            ((out.dtype, out.data_ptr() % 16 == 0),) + engine._ws_tensor_signatures
            if keyed_launch and cache_tensor_signature
            else None
        )
        key = (
            launch_specialization_key(
                arguments, 7, constants, options, _mtgp32_direct_words_kernel.debug, tensor_signatures=tensor_signatures
            )
            if keyed_launch
            else None
        )
        compiled = _launcher.launch(
            (min(blocks, states),),
            arguments,
            constants,
            options,
            specialization_key=key,
        )
    else:
        keywords = dict(zip(_launcher._constexpr_names, constants))
        keywords.update(zip(_launcher._option_names, options))
        compiled = _mtgp32_direct_words_kernel[(min(blocks, states),)](*arguments, **keywords)
    engine.offset += words
    engine._ws_next_block_start += words
    return compiled
