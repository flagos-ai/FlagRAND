"""MRG32k3a raw-pair generation fused with the existing alias conversion.

The current stream cursor defines the first raw word. Adjacent raw words
stay adjacent when pairing, including a pair that crosses the stream wrap.
The caller advances the scalar cursor and consumed-word count.
"""

import triton
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.rng.mrg32k3a import _mrg32k3a_step


@triton.jit(do_not_specialize=["N", "CURSOR"])
def _mrg32k3a_poisson(
    STATE,
    OUT,
    THRESHOLD,
    ALIASES,
    N,
    CURSOR,
    NSTREAMS: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    stream = (CURSOR + slot) % NSTREAMS
    present = (slot < NSTREAMS) & (slot < 2 * N)
    x0 = tl.load(STATE + stream, present, other=1).to(tl.uint32).to(tl.int64)
    x1 = tl.load(STATE + NSTREAMS + stream, present, other=1).to(tl.uint32).to(tl.int64)
    x2 = tl.load(STATE + 2 * NSTREAMS + stream, present, other=1).to(tl.uint32).to(tl.int64)
    y0 = tl.load(STATE + 3 * NSTREAMS + stream, present, other=1).to(tl.uint32).to(tl.int64)
    y1 = tl.load(STATE + 4 * NSTREAMS + stream, present, other=1).to(tl.uint32).to(tl.int64)
    y2 = tl.load(STATE + 5 * NSTREAMS + stream, present, other=1).to(tl.uint32).to(tl.int64)
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for batch in range(tl.cdiv(2 * N, NSTREAMS)):
        active = present & (batch * NSTREAMS + slot < 2 * N)
        word, a, b, c, d, e, f = _mrg32k3a_step(x0, x1, x2, y0, y1, y2)
        first, second = tl.split(tl.reshape(word, (BLOCK, 2)))
        value = _sample_alias(first, second, THRESHOLD, ALIASES, FIRST, SIZE, 4294967087)
        index = batch * (NSTREAMS // 2) + pair
        tl.store(OUT + index, value, (pair < NSTREAMS // 2) & (index < N))
        x0 = tl.where(active, a, x0)
        x1 = tl.where(active, b, x1)
        x2 = tl.where(active, c, x2)
        y0 = tl.where(active, d, y0)
        y1 = tl.where(active, e, y1)
        y2 = tl.where(active, f, y2)
    tl.store(STATE + stream, x0.to(tl.uint32), present)
    tl.store(STATE + NSTREAMS + stream, x1.to(tl.uint32), present)
    tl.store(STATE + 2 * NSTREAMS + stream, x2.to(tl.uint32), present)
    tl.store(STATE + 3 * NSTREAMS + stream, y0.to(tl.uint32), present)
    tl.store(STATE + 4 * NSTREAMS + stream, y1.to(tl.uint32), present)
    tl.store(STATE + 5 * NSTREAMS + stream, y2.to(tl.uint32), present)


def launch_mrg32k3a_poisson(engine, out, threshold, aliases, first, size, block_size=128, num_warps=4):
    state = engine.ensure_state(out.device)
    n = out.numel()
    _mrg32k3a_poisson[(triton.cdiv(min(n, engine.nstreams // 2), block_size),)](
        state, out, threshold, aliases, n, engine._cursor, engine.nstreams, first, size, block_size, num_warps=num_warps
    )
    return out
