"""Research: shared MTGP32 recurrence, direct paired FP64 output and raw pending tail."""

import torch

import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from flagrand.rng._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from flagrand.rng._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper
from flagrand.rng._mtgp32_sequence import clear_chunk_cache
from flagrand.rng import _mtgp32_thead_buffered as buffered
from flagrand.rng._mtgp32_transforms import _uniform64


@triton.jit(do_not_specialize=["CACHE_OFFSET", "START_BLOCK", "BLOCK_COUNT", "PREFIX", "WORDS"])
def _uniform64_shared_tail_kernel(
    OUT,
    TAIL,
    CACHE,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    CACHE_OFFSET,
    START_BLOCK,
    BLOCK_COUNT,
    PREFIX,
    WORDS,
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
    state_base = state_id * (STATE_MASK + 1)
    param_base = state_id * 16
    lane = tl.arange(0, BLOCK_SIZE)
    # PREFIX is even and at most one block. No pair crosses an old/new
    # raw boundary in this candidate. Odd raw offsets keep the original path.
    if PREFIX > 0:
        if p == 0:
            old_word = tl.load(CACHE + CACHE_OFFSET + lane, lane < PREFIX, other=0).to(tl.uint32, bitcast=True)
            old_partner = tl.inline_asm_elementwise(
                "ppu.shfl.sync.bfly.b32 $0, $1, 1, 31, -1;",
                constraints="=r,r",
                args=[old_word],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            tl.store(OUT + lane // 2, _uniform64(old_word, old_partner), ((lane & 1) == 0) & (lane < PREFIX))
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
        word_index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane + PREFIX
        # Every lane participates before the even-lane output predicate.
        partner = tl.inline_asm_elementwise(
            "ppu.shfl.sync.bfly.b32 $0, $1, 1, 31, -1;",
            constraints="=r,r",
            args=[word],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        tl.store(OUT + (word_index >> 1), _uniform64(word, partner), ((lane & 1) == 0) & (word_index < WORDS))
        # Keep every unused word, including both members of unused pairs.
        # The tail is raw32, so later distribution changes reuse exact bits.
        tl.store(TAIL + word_index - WORDS, word.to(tl.int32, bitcast=True), word_index >= WORDS)
        # Includes the final step before the complete physical-state writeback.
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + state_base + ring_lane, final.to(tl.int32, bitcast=True))


_launcher = CachedKernelLauncher(
    _uniform64_shared_tail_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
)


def launch_tail(
    out, tail, cache, cache_offset, state, pos, sh1, sh2, param, temper, *, start_block, block_count, prefix, words
):
    arguments = (
        out,
        tail,
        cache,
        state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        cache_offset,
        start_block,
        block_count,
        prefix,
        words,
    )
    constants = (state.shape[0], MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
    key = launch_specialization_key(arguments, 9, constants, (8,), _uniform64_shared_tail_kernel.debug)
    return _launcher.launch((min(block_count, state.shape[0]),), arguments, constants, (8,), specialization_key=key)


def refill_plan(engine, words, buffer):
    cache = getattr(engine, "_chunk_cache", None)
    consumed = 0 if cache is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if cache is None else cache.numel() - consumed
    target = max(words, min(buffer.prefetch_calls * words, buffer.max_prefetch_words))
    blocks = (target - remaining + 255) // 256
    pending = remaining + blocks * 256 - words
    return cache, consumed, remaining, blocks, pending


def generate_tail(engine, out, n):
    words = 2 * n
    # The established aligned, no-cache fused kernel is the unchanged control.
    if getattr(engine, "_chunk_cache", None) is None and words % 256 == 0 and engine.offset % 256 == 0:
        return False
    buffer = buffered._buffer_state(engine)
    cache, _, remaining, blocks, pending = refill_plan(engine, words, buffer)
    # Keep the original prefetch formula, odd-start fallback and short prefix.
    # Only a live raw cache may retain a larger tail within its configured cap.
    if (
        remaining >= words
        or remaining > 256
        or remaining % 2
        or engine.offset % 2
        or (pending > 256 and (cache is None or pending > buffer.max_prefetch_words))
        or blocks >= 65536
    ):
        return False
    buffer = buffered.prepare_stream(engine, out.device)
    cache, consumed, remaining, blocks, pending = refill_plan(engine, words, buffer)
    tail = torch.empty(max(1, pending), dtype=torch.int32, device=out.device)
    launch_tail(
        out,
        tail,
        cache if remaining else tail,
        consumed,
        engine._working_state,
        *engine._ws_params,
        start_block=engine._ws_next_block_start // 256,
        block_count=blocks,
        prefix=remaining,
        words=words,
    )
    out.record_stream(buffer.stream)
    engine._ws_next_block_start += blocks * 256
    engine.offset += words
    if pending:
        engine._chunk_cache = tail
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(out.device))
    else:
        clear_chunk_cache(engine)
    buffer.values = None
    return True
