"""Ascend MRG Poisson from consecutive interleaved raw-stream words.

Even stream counts use physical pair tiles. An odd starting cursor places
only the last physical pair across the S-1/0 boundary, with a scalar owner
for state zero. Odd stream counts use the ordinary raw kernel then alias.
Offset and cursor advancement belong to the outer Poisson API.
"""

import triton
import triton.language as tl

from ._mrg32k3a_ascend_active import _add_words, _multiply_words, _unsigned_less, step_bounded
from ._mrg32k3a_ascend_store import _store_rotation


@triton.jit
def _fold(high, low, C: tl.constexpr):
    h, l = _multiply_words(high, C)
    return _add_words(h, l, 0, low)


@triton.jit
def _alias(a, b, threshold, aliases, FIRST: tl.constexpr, SIZE: tl.constexpr):
    # floor((a-1)*SIZE/M1). Each high-word fold adds that high word to
    # the quotient because 2**32=M1+209. This also avoids float binning.
    high, low = _multiply_words((a - 1).to(tl.int32), SIZE)
    column = high
    high, low = _fold(high, low, 209)
    column += high
    high, low = _fold(high, low, 209)
    column += high
    high, low = _fold(high, low, 209)
    column += tl.where(_unsigned_less(low, -209), 0, 1)
    cutoff = tl.load(threshold + column)
    other = tl.load(aliases + column)
    return FIRST + tl.where(_unsigned_less((b - 1).to(tl.int32), cutoff), column, other)


@triton.jit
def _load_pair_row(ROW, BASE, S: tl.constexpr, BLOCK: tl.constexpr, CURSOR_ODD: tl.constexpr):
    lanes = tl.arange(0, 2 * BLOCK)
    stream = BASE + int(CURSOR_ODD) + lanes
    present = lanes < S - BASE
    values = tl.load(ROW + stream, present & (stream < S), other=1)
    if CURSOR_ODD:
        # Only the last physical pair is (S-1, 0). Keep the main vector
        # address unit-stride and load its wrapped final word as a scalar.
        last_tile = BASE + 2 * BLOCK >= S
        wrapped = tl.load(ROW, last_tile, other=1)
        values = tl.where(lanes == S - BASE - 1, wrapped, values)
    return values


@triton.jit
def _store_pair_row(ROW, BASE, VALUES, S: tl.constexpr, BLOCK: tl.constexpr, CURSOR_ODD: tl.constexpr):
    lanes = tl.arange(0, 2 * BLOCK)
    stream = BASE + int(CURSOR_ODD) + lanes
    present = lanes < S - BASE
    tl.store(ROW + stream, VALUES, present & (stream < S))
    if CURSOR_ODD:
        # One int32 lane carries the wrapped state word. Integer reduction
        # selects its exact bits; a scalar store has one physical owner.
        wrapped = tl.sum(tl.where(lanes == S - BASE - 1, VALUES, 0), 0)
        tl.store(ROW, wrapped, BASE + 2 * BLOCK >= S)


@triton.jit
def even_pairs_kernel(
    STATE,
    OUT,
    THRESHOLD,
    ALIASES,
    N,
    CURSOR,
    S: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    CURSOR_ODD: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
):
    pairs: tl.constexpr = S // 2
    pair_cursor = CURSOR // 2
    physical_tile = (pair_cursor // BLOCK + tl.program_id(0)) % tl.cdiv(pairs, BLOCK)
    pair_base = physical_tile * BLOCK
    pair = pair_base + tl.arange(0, BLOCK)
    rotated = pair - pair_cursor
    slot = tl.where(rotated < 0, rotated + pairs, rotated)
    word_base = 2 * pair_base
    x0 = _load_pair_row(STATE, word_base, S, BLOCK, CURSOR_ODD)
    x1 = _load_pair_row(STATE + S, word_base, S, BLOCK, CURSOR_ODD)
    x2 = _load_pair_row(STATE + 2 * S, word_base, S, BLOCK, CURSOR_ODD)
    y0 = _load_pair_row(STATE + 3 * S, word_base, S, BLOCK, CURSOR_ODD)
    y1 = _load_pair_row(STATE + 4 * S, word_base, S, BLOCK, CURSOR_ODD)
    y2 = _load_pair_row(STATE + 5 * S, word_base, S, BLOCK, CURSOR_ODD)
    present = pair < pairs
    for batch in range(tl.cdiv(N, pairs)):
        index = slot + batch * pairs
        active_pair = present & (index < N)
        active_word = tl.interleave(active_pair, active_pair)
        word, ax, bx, cx, dy, ey, fy = step_bounded(x0, x1, x2, y0, y1, y2)
        a, b = tl.split(tl.reshape(word, (BLOCK, 2)))
        value = _alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE)
        _store_rotation(value, OUT, N, pair_base, pair_cursor, batch, pairs, BLOCK, 1, 0, CURSOR_ALIGNED)
        x0, x1, x2 = tl.where(active_word, ax, x0), tl.where(active_word, bx, x1), tl.where(active_word, cx, x2)
        y0, y1, y2 = tl.where(active_word, dy, y0), tl.where(active_word, ey, y1), tl.where(active_word, fy, y2)
    _store_pair_row(STATE, word_base, x0, S, BLOCK, CURSOR_ODD)
    _store_pair_row(STATE + S, word_base, x1, S, BLOCK, CURSOR_ODD)
    _store_pair_row(STATE + 2 * S, word_base, x2, S, BLOCK, CURSOR_ODD)
    _store_pair_row(STATE + 3 * S, word_base, y0, S, BLOCK, CURSOR_ODD)
    _store_pair_row(STATE + 4 * S, word_base, y1, S, BLOCK, CURSOR_ODD)
    _store_pair_row(STATE + 5 * S, word_base, y2, S, BLOCK, CURSOR_ODD)


@triton.jit
def alias_kernel(RAW, OUT, THRESHOLD, ALIASES, N, FIRST: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(RAW + 2 * i, i < N, other=1)
    b = tl.load(RAW + 2 * i + 1, i < N, other=1)
    value = _alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE)
    tl.store(OUT + i, value, i < N)
