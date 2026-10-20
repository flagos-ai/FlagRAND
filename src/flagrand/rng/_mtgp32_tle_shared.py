"""MTGP32 persistent shared state using the MetaX FlagTree TLE backend."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT"])
def _mtgp32_tle_shared_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START_BLOCK,
    BLOCK_COUNT,
    N_STATES: tl.constexpr,
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
    lane = tl.arange(0, BLOCK_SIZE)
    ring_lane = tl.arange(0, STATE_MASK + 1)
    buffer = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
    ring_ptr = tle.gpu.local_ptr(buffer, indices=(ring_lane,))
    initial = tl.load(STATE + s_base + ring_lane).to(tl.uint32, bitcast=True)
    tl.store(ring_ptr, initial)
    tl.debug_barrier()
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane) & STATE_MASK,)))
        x2 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + 1) & STATE_MASK,)))
        y = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos) & STATE_MASK,)))
        t = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos - 1) & STATE_MASK,)))
        r = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        value = r ^ _linear_table_value(r, p0, p1, p2, p3)
        word = _temper_linear(value, t, t0, t1, t2, t3)
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane
        tl.store(OUT + index, word.to(tl.int32, bitcast=True))
        destination = tle.gpu.local_ptr(buffer, indices=((offset + lane + N_RECUR) & STATE_MASK,))
        tl.store(destination, value)
        # All reads for this step precede its state stores. This synchronizes
        # the next step and also the final complete backing-state writeback.
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + s_base + ring_lane, final.to(tl.int32, bitcast=True))


_launcher = CachedKernelLauncher(
    _mtgp32_tle_shared_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
)


def launch_tle_shared(
    out,
    state,
    pos,
    sh1,
    sh2,
    param,
    temper,
    *,
    start_block,
    block_count,
    num_warps,
    cached_launch=True,
    keyed_launch=True,
    linear_table=True,
    tensor_signatures=None,
):
    arguments = (out, state, pos, sh1, sh2, param, temper, start_block, block_count)
    constants = (state.shape[0], MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
    grid = (min(block_count, state.shape[0]),)
    if cached_launch:
        key = (
            launch_specialization_key(
                arguments,
                7,
                constants,
                (num_warps,),
                _mtgp32_tle_shared_kernel.debug,
                tensor_signatures=tensor_signatures,
            )
            if keyed_launch
            else None
        )
        return _launcher.launch(grid, arguments, constants, (num_warps,), specialization_key=key)
    return _mtgp32_tle_shared_kernel[grid](
        *arguments,
        N_STATES=constants[0],
        BLOCK_SIZE=constants[1],
        STATE_MASK=constants[2],
        MASK=constants[3],
        N_RECUR=constants[4],
        num_warps=num_warps,
    )
