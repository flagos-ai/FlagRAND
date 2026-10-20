"""Independent two-word MTGP fusion with a matched linear-table control.

Each direct lane owns adjacent raw words. The control generates the same
256-word round as one vector and gathers pairs. Both use the same GF(2) table
bases, transform, two warps and host wrapper. MetaX public dispatch selects
the direct path for aligned uniform_f64 and Poisson calls, and for aligned
normal_f32/lognormal_f32 calls up to 1048576 outputs.
"""

from functools import lru_cache

import triton
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.runtime.backend import get_vendor_info
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _metax_tle_launcher, _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear
from ._mtgp32_sequence import ensure_working_state
from ._mtgp32_transforms import _uniform64
from ._mtgp32_uniform_f32 import _uniform32
from ._xorwow.native_math import sincos_turns
from ._xorwow.native_trig32 import sincos_turns_native32
from .mtgp32 import _float_bits


@triton.jit
def _word_linear(
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
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    x1 = tl.load(STATE + s_base + (index & STATE_MASK)).to(tl.uint32, bitcast=True)
    x2 = tl.load(STATE + s_base + ((index + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y = tl.load(STATE + s_base + ((index + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
    value = y ^ _linear_table_value(y, p0, p1, p2, p3)
    t = tl.load(STATE + s_base + ((index + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    word = _temper_linear(value, t, t0, t1, t2, t3)
    return value, word


@triton.jit
def _pair_linear(
    STATE, s_base, index, pos, sh1, sh2, p0, p1, p2, p3, t0, t1, t2, t3, STATE_MASK: tl.constexpr, MASK: tl.constexpr
):
    x0 = tl.load(STATE + s_base + (index & STATE_MASK)).to(tl.uint32, bitcast=True)
    x1 = tl.load(STATE + s_base + ((index + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    x2 = tl.load(STATE + s_base + ((index + 2) & STATE_MASK)).to(tl.uint32, bitcast=True)
    t = tl.load(STATE + s_base + ((index + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y0 = tl.load(STATE + s_base + ((index + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
    y1 = tl.load(STATE + s_base + ((index + pos + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
    r0 = _mtgp32_recurrence(x0, x1, y0, sh1, sh2, MASK)
    r1 = _mtgp32_recurrence(x1, x2, y1, sh1, sh2, MASK)
    value_a = r0 ^ _linear_table_value(r0, p0, p1, p2, p3)
    value_b = r1 ^ _linear_table_value(r1, p0, p1, p2, p3)
    a = _temper_linear(value_a, t, t0, t1, t2, t3)
    b = _temper_linear(value_b, y0, t0, t1, t2, t3)
    return value_a, a, value_b, b


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_direct_pairs_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    THRESHOLD,
    ALIASES,
    START_BLOCK,
    BLOCK_COUNT,
    N,
    MU_BITS,
    SIGMA_BITS,
    N_STATES: tl.constexpr,
    MODE: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    DIRECT: tl.constexpr,
    NATIVE_TRIG: tl.constexpr,
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
    p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
    t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
    pair = tl.arange(0, BLOCK_SIZE // 2)
    mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
    sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        if DIRECT:
            # All round inputs precede the disjoint [351,606] write interval
            # in the 1024-word ring, so these two groups can execute in order.
            value_a, a, value_b, b = _pair_linear(
                STATE,
                s_base,
                offset + 2 * pair,
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
            # All six distinct inputs are read before either state write.
            # Y(a) is also T(b); spelling out that reuse avoids a duplicate
            # global load left by the MetaX modular-index lowering.
            tl.store(STATE + s_base + ((offset + 2 * pair + N_RECUR) & STATE_MASK), value_a.to(tl.int32, bitcast=True))
            tl.store(
                STATE + s_base + ((offset + 2 * pair + 1 + N_RECUR) & STATE_MASK), value_b.to(tl.int32, bitcast=True)
            )
        else:
            lane = tl.arange(0, BLOCK_SIZE)
            value, word = _word_linear(
                STATE, s_base, offset + lane, pos, sh1, sh2, p0, p1, p2, p3, t0, t1, t2, t3, STATE_MASK, MASK, N_RECUR
            )
            tl.store(STATE + s_base + ((offset + lane + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
            a = tl.gather(word, 2 * pair, 0)
            b = tl.gather(word, 2 * pair + 1, 0)
        block = k.to(tl.int64) * N_STATES + p
        output_pair = block * (BLOCK_SIZE // 2) + pair
        if MODE == 1:
            tl.store(OUT + output_pair, _uniform64(a, b), output_pair < N)
        elif MODE == 4:
            value = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
            tl.store(OUT + output_pair, value, output_pair < N)
        else:
            u1, u2 = _uniform32(a), _uniform32(b)
            radius = tl.sqrt(-2.0 * tl.log(u1))
            if NATIVE_TRIG:
                sine, cosine = sincos_turns_native32(u2)
            else:
                sine, cosine = sincos_turns(u2, False)
            x = mu + sigma * radius * cosine
            y = mu + sigma * radius * sine
            if MODE == 3:
                x, y = tl.exp(x), tl.exp(y)
            tl.store(OUT + 2 * output_pair, x, 2 * output_pair < N)
            tl.store(OUT + 2 * output_pair + 1, y, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()


_launcher = CachedKernelLauncher(
    _mtgp32_direct_pairs_kernel,
    constexpr_names=(
        "N_STATES",
        "MODE",
        "FIRST",
        "SIZE",
        "BLOCK_SIZE",
        "STATE_MASK",
        "MASK",
        "N_RECUR",
        "DIRECT",
        "NATIVE_TRIG",
    ),
    option_names=("num_warps", "enable_fp_fusion"),
)


@lru_cache(maxsize=1)
def _metax_shared_pairs():
    if _metax_tle_launcher(get_vendor_info().vendor_name) is None:
        return None
    from ._mtgp32_tle_pairs import generate_tle_pairs

    return generate_tle_pairs


@lru_cache(maxsize=1)
def _metax_shared_normal():
    if _metax_tle_launcher(get_vendor_info().vendor_name) is None:
        return None
    from ._mtgp32_tle_normal import generate_tle_normal

    return generate_tle_normal


def generate_direct_pairs(
    engine,
    out,
    distribution="uniform",
    *,
    direct=True,
    mean=0.0,
    stddev=1.0,
    threshold=None,
    aliases=None,
    first=0,
    size=1,
    cached_launch=True,
    keyed_launch=True,
    cache_tensor_signature=True,
    num_warps=2,
    native_trig=False,
):
    """Generate complete 256-raw-word rounds from an aligned engine offset.

    Uniform is FP64, normal/lognormal FP32, and Poisson int32. Partial raw-word
    rounds remain the public buffered path's responsibility. Return the actual
    compiled kernel for the independent experiment's compiler evidence.
    """
    mode = {"uniform": 1, "normal": 2, "lognormal": 3, "poisson": 4}[distribution]
    n = out.numel()
    words = 2 * n if mode in (1, 4) else 2 * ((n + 1) // 2)
    if direct and mode in (1, 4) and num_warps == 2 and words >= 1048576:
        shared_pairs = _metax_shared_pairs()
        if shared_pairs is not None:
            return shared_pairs(
                engine,
                out,
                distribution,
                threshold=threshold,
                aliases=aliases,
                first=first,
                size=size,
                cached_launch=cached_launch,
                keyed_launch=keyed_launch,
                cache_tensor_signature=cache_tensor_signature,
                num_warps=num_warps,
            )
    if direct and mode in (2, 3) and not native_trig and num_warps == 2 and words >= 1048576 and n <= 1048576:
        # Shared state helps at the existing fusion limit; larger public
        # normal calls retain independent raw generation and transformation.
        shared_normal = _metax_shared_normal()
        if shared_normal is not None:
            return shared_normal(
                engine,
                out,
                distribution,
                mean=mean,
                stddev=stddev,
                cached_launch=cached_launch,
                keyed_launch=keyed_launch,
                cache_tensor_signature=cache_tensor_signature,
                num_warps=num_warps,
            )
    ensure_working_state(engine, engine.seed, str(out.device))
    pos, sh1, sh2, param, temper = engine._ws_params
    blocks, states = words // MTGP32_BLOCK_SIZE, engine.state.shape[0]
    arguments = (
        out,
        engine.state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        out if threshold is None else threshold,
        out if aliases is None else aliases,
        engine.offset // MTGP32_BLOCK_SIZE,
        blocks,
        n,
        _float_bits(mean, False),
        _float_bits(stddev, False),
    )
    constants = (
        states,
        mode,
        first,
        size,
        MTGP32_BLOCK_SIZE,
        MTGP32_STATE_MASK,
        MTGP32_MASK,
        MTGPDC_N,
        direct,
        native_trig,
    )
    options = (num_warps, mode not in (2, 3))
    grid = (min(blocks, states),)
    if cached_launch:
        signatures = None
        if keyed_launch and cache_tensor_signature:
            out_signature = (out.dtype, out.data_ptr() % 16 == 0)
            signatures = (
                (out_signature,)
                + engine._ws_tensor_signatures
                + (
                    out_signature if threshold is None else (threshold.dtype, threshold.data_ptr() % 16 == 0),
                    out_signature if aliases is None else (aliases.dtype, aliases.data_ptr() % 16 == 0),
                )
            )
        key = (
            launch_specialization_key(
                arguments, 9, constants, options, _mtgp32_direct_pairs_kernel.debug, tensor_signatures=signatures
            )
            if keyed_launch
            else None
        )
        compiled = _launcher.launch(grid, arguments, constants, options, specialization_key=key)
    else:
        keywords = dict(zip(_launcher._constexpr_names, constants))
        keywords.update(zip(_launcher._option_names, options))
        compiled = _mtgp32_direct_pairs_kernel[grid](*arguments, **keywords)
    engine.offset += words
    engine._ws_next_block_start += words
    return compiled
