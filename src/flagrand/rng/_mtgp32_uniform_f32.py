"""Aligned MTGP32 uniform-f32 fusion, also callable for direct comparisons.

Inputs consume complete 256-word MTGP blocks and start at a block boundary.
The caller selects this domain; the ordinary cached path handles partial
blocks. State IDs, output-block order and word consumption are unchanged.
"""

import triton
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper, mtgp32_launch_warps
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear
from ._mtgp32_sequence import ensure_working_state


@triton.jit
def _uniform32(word):
    # Identical to _mtgp32_transforms._uniform_kernel's f32 expression:
    # top 24 random bits plus one produce the grid {1,...,2**24}/2**24.
    return ((word >> 8) + 1).to(tl.float32) * 5.960464477539063e-8


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT"])
def _mtgp32_uniform_f32_kernel(
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
    LINEAR_TABLE: tl.constexpr = False,
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
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane
        tl.store(OUT + index, _uniform32(word))
        if k + 1 < iterations:
            tl.debug_barrier()


_uniform_f32_launcher = CachedKernelLauncher(
    _mtgp32_uniform_f32_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR", "LINEAR_TABLE"),
)


def generate_uniform_f32_aligned(
    engine, out, *, num_warps=8, cached_launch=True, keyed_launch=True, linear_table=False, cache_tensor_signature=True
):
    """Generate aligned f32 output, advancing device state and word counters."""
    device = str(out.device)
    ensure_working_state(engine, engine.seed, device)
    pos, sh1, sh2, param, temper = engine._ws_params
    blocks = out.numel() // MTGP32_BLOCK_SIZE
    states = engine._working_state.shape[0]
    arguments = (
        out,
        engine._working_state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        engine.offset // MTGP32_BLOCK_SIZE,
        blocks,
    )
    warps = mtgp32_launch_warps(blocks, (blocks + states - 1) // states, num_warps)
    if cached_launch:
        constants = (states, MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N, linear_table)
        if keyed_launch:
            signatures = (
                ((out.dtype, out.data_ptr() % 16 == 0),) + engine._ws_tensor_signatures
                if cache_tensor_signature
                else None
            )
            key = launch_specialization_key(
                arguments,
                7,
                constants,
                (warps,),
                _mtgp32_uniform_f32_kernel.debug,
                tensor_signatures=signatures,
            )
        else:
            key = None
        _uniform_f32_launcher.launch(
            (min(blocks, states),),
            arguments,
            constants,
            (warps,),
            specialization_key=key,
        )
    else:
        _mtgp32_uniform_f32_kernel[(min(blocks, states),)](
            *arguments,
            N_STATES=states,
            BLOCK_SIZE=MTGP32_BLOCK_SIZE,
            STATE_MASK=MTGP32_STATE_MASK,
            MASK=MTGP32_MASK,
            N_RECUR=MTGPDC_N,
            LINEAR_TABLE=linear_table,
            num_warps=warps,
        )
    engine.offset += out.numel()
    engine._ws_next_block_start += out.numel()
    return out
