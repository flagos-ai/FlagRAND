"""PPU MTGP32 shared state with the original global parameter tables."""

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from ._mtgp32_uniform_f32 import _uniform32


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT"])
def _mtgp32_thead_shared_kernel(
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
    OUTPUT_MODE: tl.constexpr,
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
    state_base = state_id * (STATE_MASK + 1)
    param_base = state_id * 16
    lane = tl.arange(0, BLOCK_SIZE)
    ring_lane = tl.arange(0, STATE_MASK + 1)
    ring = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
    ring_ptr = tle.gpu.local_ptr(ring, indices=(ring_lane,))
    initial = tl.load(STATE + state_base + ring_lane).to(tl.uint32, bitcast=True)
    tl.store(ring_ptr, initial)
    tl.debug_barrier()
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(tle.gpu.local_ptr(ring, indices=((lane + offset) & STATE_MASK,)))
        x2 = tl.load(tle.gpu.local_ptr(ring, indices=((lane + offset + 1) & STATE_MASK,)))
        y = tl.load(tle.gpu.local_ptr(ring, indices=((lane + offset + pos) & STATE_MASK,)))
        y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        matrix = tl.load(PARAM + param_base + (y & 15)).to(tl.uint32, bitcast=True)
        value = y ^ matrix
        t = tl.load(tle.gpu.local_ptr(ring, indices=((lane + offset + pos - 1) & STATE_MASK,)))
        word = _mtgp32_temper(value, t, TEMPER + param_base)
        destination = tle.gpu.local_ptr(ring, indices=((lane + offset + N_RECUR) & STATE_MASK,))
        # Current 192 parameter sets have pos in [3, 93]. Across the CTA,
        # this step reads [0, 348] and writes [351, 606] before ring rotation.
        # The sets are disjoint; the explicit barrier preserves the next step.
        # A tied, unused result satisfies elementwise asm's output requirement.
        tl.inline_asm_elementwise(
            "ppu.st.shared.b32 [ $2 + 0 ], { $1 };",
            constraints="=r,0,l,~{memory}",
            args=[value, destination],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane
        if OUTPUT_MODE == 1:
            tl.store(OUT + index, _uniform32(word))
        else:
            tl.store(OUT + index, word.to(tl.int32, bitcast=True))
        # Includes the final step before the complete physical-state writeback.
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + state_base + ring_lane, final.to(tl.int32, bitcast=True))


_launcher = CachedKernelLauncher(
    _mtgp32_thead_shared_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR", "OUTPUT_MODE"),
)


def launch_thead_shared(
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
    tensor_signatures=None,
    output_mode=0,
):
    arguments = (out, state, pos, sh1, sh2, param, temper, start_block, block_count)
    constants = (state.shape[0], MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N, output_mode)
    grid = (min(block_count, state.shape[0]),)
    if cached_launch:
        key = (
            launch_specialization_key(
                arguments,
                7,
                constants,
                (num_warps,),
                _mtgp32_thead_shared_kernel.debug,
                tensor_signatures=tensor_signatures,
            )
            if keyed_launch
            else None
        )
        return _launcher.launch(grid, arguments, constants, (num_warps,), specialization_key=key)
    return _mtgp32_thead_shared_kernel[grid](
        *arguments,
        N_STATES=constants[0],
        BLOCK_SIZE=constants[1],
        STATE_MASK=constants[2],
        MASK=constants[3],
        N_RECUR=constants[4],
        OUTPUT_MODE=constants[5],
        num_warps=num_warps,
    )
