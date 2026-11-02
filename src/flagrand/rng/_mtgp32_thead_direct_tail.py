"""PPU MTGP32 raw refill with direct output and a bounded unconsumed tail."""

import torch
import triton
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from . import _mtgp32_thead_buffered as buffered
from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from ._mtgp32_sequence import clear_chunk_cache


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "PREFIX", "N"])
def _raw_pointer_tail_kernel(
    OUT,
    TAIL,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START_BLOCK,
    BLOCK_COUNT,
    PREFIX,
    N,
    N_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    # Identical global-state recurrence to _mtgp32_contiguous_kernel,
    # LINEAR_TABLE=False, num_warps=8. Only the output destinations differ.
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
        output = _mtgp32_temper(value, t, TEMPER + p_base)
        tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane + PREFIX
        destination = tl.where(index < N, OUT + index, TAIL + index - N)
        tl.store(destination, output.to(tl.int32, bitcast=True))
        if k + 1 < iterations:
            tl.debug_barrier()


_launcher = CachedKernelLauncher(
    _raw_pointer_tail_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
)


def launch_direct_tail(out, tail, state, pos, sh1, sh2, param, temper, *, start_block, block_count, prefix, n):
    arguments = (out, tail, state, pos, sh1, sh2, param, temper, start_block, block_count, prefix, n)
    constants = (state.shape[0], MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
    options = (8,)
    key = launch_specialization_key(arguments, 8, constants, options, _raw_pointer_tail_kernel.debug)
    return _launcher.launch((min(block_count, state.shape[0]),), arguments, constants, options, specialization_key=key)


def refill_plan(engine, n, buffer):
    cache = engine._chunk_cache
    consumed = 0 if cache is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if cache is None else cache.numel() - consumed
    target = max(n, min(buffer.prefetch_calls * n, buffer.max_prefetch_words))
    blocks = (target - remaining + 255) // 256
    pending = remaining + blocks * 256 - n
    return cache, consumed, remaining, blocks, pending


def generate_direct_tail(engine, out, n):
    buffer = buffered._buffer_state(engine)
    _, _, remaining, blocks, pending = refill_plan(engine, n, buffer)
    # Keep cache hits, larger prefetched tails and the existing shared-state
    # recurrence region on their original paths. The measured candidate is
    # the global-state refill with at most one unconsumed 256-word block.
    if remaining >= n or pending > 256 or blocks >= 65536:
        return False
    buffer = buffered.prepare_stream(engine, out.device)
    cache, consumed, remaining, blocks, pending = refill_plan(engine, n, buffer)
    if remaining:
        buffered._copy_tensor(buffer, out, cache, consumed, remaining)
    # A zero logical tail still receives a valid, unused pointer. It is not
    # retained in the generator cache after the call.
    tail = torch.empty(max(1, pending), dtype=torch.int32, device=out.device)
    launch_direct_tail(
        out,
        tail,
        engine._working_state,
        *engine._ws_params,
        start_block=engine._ws_next_block_start // 256,
        block_count=blocks,
        prefix=remaining,
        n=n,
    )
    out.record_stream(buffer.stream)
    engine._ws_next_block_start += blocks * 256
    engine.offset += n
    if pending:
        engine._chunk_cache = tail
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(out.device))
    else:
        clear_chunk_cache(engine)
    buffer.values = None
    return True
