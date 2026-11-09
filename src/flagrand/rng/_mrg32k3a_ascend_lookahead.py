"""Bounded standard FP64 normal lookahead with immediate real-state commits.

Each window advances a stream at most once. Every public call copies its
outputs and commits its consumed streams; unused future state is never exposed.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flagrand.runtime.ascend_launcher import PreparedAscendLauncher
from flagrand.runtime.specialization import integer_specialization_key

PREFETCH_CALLS = 64
MAX_PAIRS = 131072
MAX_STREAMS = 131072
MAX_OUTPUTS = 16384
BLOCK = 256


@triton.jit(do_not_specialize=["PAIRS", "CURSOR", "START_PAIR"])
def commit_kernel(
    STATE,
    FUTURE,
    VALUES,
    OUT,
    PAIRS,
    CURSOR,
    START_PAIR,
    S: tl.constexpr,
    B: tl.constexpr,
    WORDS_PER_PAIR: tl.constexpr,
):
    # State and output have independent CTA ownership. Every selected physical
    # state tile has one owner, including a wrap through a partial final tile.
    pid = tl.program_id(0)
    tile = (CURSOR // B + pid) % tl.cdiv(S, B)
    stream = tile * B + tl.arange(0, B)
    delta = stream - CURSOR
    logical = tl.where(delta < 0, delta + S, delta)
    present = stream < S
    consumed = logical < PAIRS
    for row in tl.static_range(0, 6):
        original = tl.load(STATE + row * S + stream, present, other=0)
        future = tl.load(FUTURE + row * S + stream, present, other=0)
        value = tl.where(consumed, future, original)
        # Whole physical tile, with a prefix-only mask at the last state row
        # tile. Inactive streams retain their real state rather than lookahead.
        tl.store(STATE + row * S + stream, value, present)

    # tile_count(S, CURSOR, PAIRS, B) >= ceil(PAIRS/B). An extra state-only CTA
    # has an empty output mask. No floating-point operation touches the bits.
    word = WORDS_PER_PAIR * pid * B + tl.arange(0, WORDS_PER_PAIR * B)
    active_word = word < WORDS_PER_PAIR * PAIRS
    bits = tl.load(VALUES + WORDS_PER_PAIR * START_PAIR + word, active_word, other=0)
    tl.store(OUT + word, bits, active_word)


class Normal64Window:
    def __init__(self, engine, out, distribution, calls, refill, launcher, tile_count, block, stream):
        from flagrand.rng.mrg32k3a import Mrg32k3aGenerator

        self.n = out.numel()
        self.pairs = self.n // 2
        self.f64 = out.dtype == torch.float64
        self.words_per_pair = 4 if self.f64 else 2
        self.distribution = distribution
        self.device = out.device
        self.state = engine.ensure_state(out.device)
        self.seed = engine.seed
        self.streams = engine.nstreams
        self.expected_offset = engine.offset
        self.expected_cursor = engine._cursor
        self.calls = min(calls, 64, self.streams // self.pairs, MAX_PAIRS // self.pairs)
        self.position = 0

        # offset is a raw-word count, not a snapshot of mixed-call interleaving.
        # Clone the actual state and copy its cursor explicitly before prefill.
        temporary = Mrg32k3aGenerator(seed=engine.seed, offset=engine.offset, nstreams=self.streams)
        temporary._state = self.state.clone()
        temporary._state_seed = engine._state_seed
        temporary._state_offset = engine._state_offset
        temporary._cursor = engine._cursor
        values = torch.empty(self.calls * self.n, dtype=out.dtype, device=out.device)
        refill(temporary, values, distribution)
        self.future = temporary._state
        self.values = values.view(torch.int32)
        self.launcher = launcher
        self.constants = self.streams, block, self.words_per_pair
        self.options = 4, False, False
        self.stream = stream
        signatures = tuple((tensor.dtype, tensor.data_ptr() % 16) for tensor in (self.state, self.future, self.values))
        prefix = out.device, signatures, self.pairs, self.constants
        self.steps = []
        for index in range(self.calls):
            cursor = (engine._cursor + index * self.pairs) % self.streams
            start = index * self.pairs
            scalars = self.pairs, cursor, start
            key = prefix, tuple(integer_specialization_key(value) for value in scalars)
            grid = (tile_count(self.streams, cursor, self.pairs, block),)
            self.steps.append((grid, scalars, key))
        self.out = None
        self.out_pointer = None
        self.target = None
        self.target_alignment = None
        # Only future state and values survive this constructor; production
        # radius/angle/raw/standard scratch belongs to the temporary engine.

    def matches(self, engine, out, distribution, stream):
        return (
            self.position < self.calls
            and self.n == out.numel()
            and self.f64 == (out.dtype == torch.float64)
            and self.distribution == distribution
            and self.device == out.device
            and self.state is engine._state
            and self.seed == engine.seed == engine._state_seed
            and self.streams == engine.nstreams
            and self.expected_offset == engine.offset == engine._state_offset
            and self.expected_cursor == engine._cursor
            and self.stream == stream
        )

    def commit(self, engine, out):
        pointer = out.data_ptr()
        if self.out is not out or self.out_pointer != pointer:
            self.out = out
            self.out_pointer = pointer
            self.target = out.view(torch.int32)
            self.target_alignment = pointer % 16
        grid, scalars, key = self.steps[self.position]
        self.launcher.launch(
            grid,
            (self.state, self.future, self.values, self.target, *scalars),
            self.constants,
            self.options,
            specialization_key=(key, self.target_alignment),
        )
        # The same queued kernel writes output and every state row for exactly
        # this call's streams. Public state never points to the future window.
        self.position += 1
        self.expected_offset += self.words_per_pair * self.pairs
        self.expected_cursor = (self.expected_cursor + self.pairs) % self.streams
        engine.offset = self.expected_offset
        engine._state_offset = self.expected_offset
        engine._cursor = self.expected_cursor
        engine._normal_has_spare[self.f64] = False
        return out


@lru_cache(maxsize=1)
def lookahead_launcher():
    return PreparedAscendLauncher(
        commit_kernel,
        constexpr_names=("S", "B", "WORDS_PER_PAIR"),
        option_names=("num_warps", "enable_fp_fusion", "sync_solver"),
    )
