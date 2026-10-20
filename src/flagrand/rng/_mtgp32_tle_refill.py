"""Shared-state MTGP32 refills for large transformed calls on MetaX TLE."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher

from ._mtgp32_buffered import _write_refill_output
from ._mtgp32_kernel import _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear


@triton.jit(do_not_specialize=["OLD_OFFSET", "PREFIX", "START_BLOCK", "BLOCK_COUNT", "N"])
def _tle_refill_kernel(
    OUT,
    OLD,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    PUBLIC_OUT,
    OLD_OFFSET,
    PREFIX,
    START_BLOCK,
    BLOCK_COUNT,
    N,
    OUTPUT_MODE: tl.constexpr,
    N_STATES: tl.constexpr,
    BLOCK: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    p = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    copy_index = p * BLOCK + lane
    old = tl.load(OLD + OLD_OFFSET + copy_index, copy_index < PREFIX, other=0)
    tl.store(OUT + copy_index, old, copy_index < PREFIX)
    if OUTPUT_MODE >= 0:
        _write_refill_output(PUBLIC_OUT, copy_index, old, PREFIX, OUTPUT_MODE)

    if p < tl.minimum(BLOCK_COUNT, N_STATES):
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
        ring_lane = tl.arange(0, STATE_MASK + 1)
        buffer = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
        ring_ptr = tle.gpu.local_ptr(buffer, indices=(ring_lane,))
        initial = tl.load(STATE + s_base + ring_lane).to(tl.uint32, bitcast=True)
        tl.store(ring_ptr, initial)
        tl.debug_barrier()
        for k in range(iterations):
            offset = ((first_round + k) * BLOCK) & STATE_MASK
            x1 = tl.load(tle.gpu.local_ptr(buffer, indices=((lane + offset) & STATE_MASK,)))
            x2 = tl.load(tle.gpu.local_ptr(buffer, indices=((lane + offset + 1) & STATE_MASK,)))
            y = tl.load(tle.gpu.local_ptr(buffer, indices=((lane + offset + pos) & STATE_MASK,)))
            y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
            value = y ^ _linear_table_value(y, p0, p1, p2, p3)
            t = tl.load(tle.gpu.local_ptr(buffer, indices=((lane + offset + pos - 1) & STATE_MASK,)))
            word = _temper_linear(value, t, t0, t1, t2, t3)
            destination = tle.gpu.local_ptr(buffer, indices=((lane + offset + N_RECUR) & STATE_MASK,))
            tl.store(destination, value)
            index = PREFIX + (k.to(tl.int64) * N_STATES + p) * BLOCK + lane
            tl.store(OUT + index, word.to(tl.int32, bitcast=True))
            if OUTPUT_MODE >= 0:
                _write_refill_output(PUBLIC_OUT, index, word, N, OUTPUT_MODE)
            tl.debug_barrier()
        final = tl.load(ring_ptr)
        tl.store(STATE + s_base + ring_lane, final.to(tl.int32, bitcast=True))


_refill_launcher = CachedKernelLauncher(
    _tle_refill_kernel,
    constexpr_names=("OUTPUT_MODE", "N_STATES", "BLOCK", "STATE_MASK", "MASK", "N_RECUR"),
)
