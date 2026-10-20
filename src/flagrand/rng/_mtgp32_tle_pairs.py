"""MTGP32 uniform FP64 and Poisson using persistent shared state on MetaX TLE."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear
from ._mtgp32_sequence import ensure_working_state
from ._mtgp32_transforms import _uniform64


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N"])
def _mtgp32_tle_pairs_kernel(
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
    N_STATES: tl.constexpr,
    MODE: tl.constexpr,
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
    p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
    t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
    pair = tl.arange(0, BLOCK_SIZE // 2)
    ring_lane = tl.arange(0, STATE_MASK + 1)
    buffer = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
    ring_ptr = tle.gpu.local_ptr(buffer, indices=(ring_lane,))
    initial = tl.load(STATE + s_base + ring_lane).to(tl.uint32, bitcast=True)
    tl.store(ring_ptr, initial)
    if MODE == 4 and SIZE <= 1024:
        table_lane = tl.arange(0, SIZE)
        cutoffs = tle.gpu.alloc([SIZE], tl.uint32, nv_mma_shared_layout=False)
        aliases = tle.gpu.alloc([SIZE], tl.int32, nv_mma_shared_layout=False)
        tl.store(
            tle.gpu.local_ptr(cutoffs, indices=(table_lane,)),
            tl.load(THRESHOLD + table_lane).to(tl.uint32, bitcast=True),
        )
        tl.store(tle.gpu.local_ptr(aliases, indices=(table_lane,)), tl.load(ALIASES + table_lane))
    tl.debug_barrier()
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        index = offset + 2 * pair
        x0 = tl.load(tle.gpu.local_ptr(buffer, indices=(index & STATE_MASK,)))
        x1 = tl.load(tle.gpu.local_ptr(buffer, indices=((index + 1) & STATE_MASK,)))
        x2 = tl.load(tle.gpu.local_ptr(buffer, indices=((index + 2) & STATE_MASK,)))
        t = tl.load(tle.gpu.local_ptr(buffer, indices=((index + pos - 1) & STATE_MASK,)))
        y0 = tl.load(tle.gpu.local_ptr(buffer, indices=((index + pos) & STATE_MASK,)))
        y1 = tl.load(tle.gpu.local_ptr(buffer, indices=((index + pos + 1) & STATE_MASK,)))
        r0 = _mtgp32_recurrence(x0, x1, y0, sh1, sh2, MASK)
        r1 = _mtgp32_recurrence(x1, x2, y1, sh1, sh2, MASK)
        value_a = r0 ^ _linear_table_value(r0, p0, p1, p2, p3)
        value_b = r1 ^ _linear_table_value(r1, p0, p1, p2, p3)
        a = _temper_linear(value_a, t, t0, t1, t2, t3)
        b = _temper_linear(value_b, y0, t0, t1, t2, t3)
        # All six inputs, including Y(a) reused as T(b), precede the writes.
        dest_a = tle.gpu.local_ptr(buffer, indices=((offset + 2 * pair + N_RECUR) & STATE_MASK,))
        dest_b = tle.gpu.local_ptr(buffer, indices=((offset + 2 * pair + 1 + N_RECUR) & STATE_MASK,))
        tl.store(dest_a, value_a)
        tl.store(dest_b, value_b)
        block = k.to(tl.int64) * N_STATES + p
        output_pair = block * (BLOCK_SIZE // 2) + pair
        if MODE == 1:
            tl.store(OUT + output_pair, _uniform64(a, b), output_pair < N)
        else:
            if SIZE <= 1024:
                column = a.to(tl.uint32) & (SIZE - 1)
                coin = b.to(tl.uint32)
                cutoff = tl.load(tle.gpu.local_ptr(cutoffs, indices=(column,)))
                other = tl.load(tle.gpu.local_ptr(aliases, indices=(column,)))
                value = FIRST + tl.where(coin < cutoff, column, other)
            else:
                value = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
            tl.store(OUT + output_pair, value, output_pair < N)
        # Makes each update visible to the next step and final state writeback.
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + s_base + ring_lane, final.to(tl.int32, bitcast=True))


_launcher = CachedKernelLauncher(
    _mtgp32_tle_pairs_kernel,
    constexpr_names=("N_STATES", "MODE", "FIRST", "SIZE", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
    option_names=("num_warps", "enable_fp_fusion"),
)


def generate_tle_pairs(
    engine,
    out,
    distribution="uniform",
    *,
    threshold=None,
    aliases=None,
    first=0,
    size=1,
    cached_launch=True,
    keyed_launch=True,
    cache_tensor_signature=True,
    num_warps=2,
):
    """Generate aligned uniform FP64 or Poisson using persistent shared state."""
    mode = {"uniform": 1, "poisson": 4}[distribution]
    ensure_working_state(engine, engine.seed, str(out.device))
    pos, sh1, sh2, param, temper = engine._ws_params
    n = out.numel()
    words = 2 * n
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
    )
    constants = (states, mode, first, size, MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
    options = (num_warps, True)
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
                arguments, 9, constants, options, _mtgp32_tle_pairs_kernel.debug, tensor_signatures=signatures
            )
            if keyed_launch
            else None
        )
        compiled = _launcher.launch(grid, arguments, constants, options, specialization_key=key)
    else:
        keywords = dict(zip(_launcher._constexpr_names, constants))
        keywords.update(zip(_launcher._option_names, options))
        compiled = _mtgp32_tle_pairs_kernel[grid](*arguments, **keywords)
    engine.offset += words
    engine._ws_next_block_start += words
    return compiled
