"""Exact MRG32K3A recurrence and contiguous active-state tiles for Ascend.

The high-word bounds are specific to the two MRG moduli and coefficients.
A float32 estimate and the exact low word recover coefficient products or
their fixed pair sums before the integer state recurrence continues.
"""

import triton
import triton.language as tl

from ._mrg32k3a_ascend_math import sincos_turns
from ._mrg32k3a_ascend_store import _store_rotation


def tile_count(streams, cursor, count, block):
    total_tiles = (streams + block - 1) // block
    active_streams = min(count, streams)
    if active_streams == 0:
        # A normal call consuming only its cached spare still needs one CTA.
        return 1
    end = cursor + active_streams
    if end <= streams:
        return (cursor % block + active_streams + block - 1) // block
    head_tiles = total_tiles - cursor // block
    tail_tiles = (end - streams + block - 1) // block
    # Start/end can occupy the same tile; that tile must have one owner.
    return min(total_tiles, head_tiles + tail_tiles)


@triton.jit
def _signed_key(a):
    # Int32 wraparound flips only the sign bit; Ascend lowers this to one add.
    return (a + tl.full((), -2147483648, tl.int32)).to(tl.int32)


@triton.jit
def _unsigned_less(a, b):
    return _signed_key(a) < _signed_key(b)


@triton.jit
def _multiply_words(a, B: tl.constexpr):
    a0, a1 = a & 65535, (a >> 16) & 65535
    w0 = a0 * (B & 65535)
    t = a1 * (B & 65535) + ((w0 >> 16) & 65535)
    w1 = a0 * (B >> 16) + (t & 65535)
    high = a1 * (B >> 16) + ((t >> 16) & 65535) + ((w1 >> 16) & 65535)
    low = (w1 << 16) | (w0 & 65535)
    return high.to(tl.int32), low.to(tl.int32)


@triton.jit
def _multiply_coefficient(a, B: tl.constexpr):
    # Only the four recurrence coefficients use this path: 1 <= B < 2**21.
    # RN32 errors before the final TRUNC cast are < 9/32 + 3*2**-26 < 1/2
    # around the exact integer high word plus 0.5.
    low = (a * B).to(tl.int32)
    unsigned_a = a.to(tl.float32) + tl.where(a < 0, 4294967296.0, 0.0)
    estimate = unsigned_a * (B * (2.0**-32))
    unsigned_low = low.to(tl.float32) + tl.where(low < 0, 4294967296.0, 0.0)
    fraction = unsigned_low * (2.0**-32)
    high = (estimate - fraction + 0.5).to(tl.int32)
    return high, low


@triton.jit
def _sum_coefficient_products(a, b, K: tl.constexpr, L: tl.constexpr):
    # Only the two MRG coefficient pairs use this helper. RN32 puts the
    # estimate minus the low-word fraction within 15/32 of the integer high
    # word; adding 0.5 followed by TRUNC recovers that word exactly.
    low = (a * K + b * L).to(tl.int32)
    # Centering before conversion keeps the unsigned estimate within 192.
    # This approximation is only for high-word recovery, never output uniform.
    unsigned_a = _signed_key(a).to(tl.float32) + 2147483648.0
    unsigned_b = _signed_key(b).to(tl.float32) + 2147483648.0
    first = unsigned_a * (K * (2.0**-32))
    second = unsigned_b * (L * (2.0**-32))
    estimate = first + second
    unsigned_low = _signed_key(low).to(tl.float32) + 2147483648.0
    fraction = unsigned_low * (2.0**-32)
    high = (estimate - fraction + 0.5).to(tl.int32)
    return high, low


@triton.jit
def _add_words(ah, al, bh, bl):
    low = (al + bl).to(tl.int32)
    return ah + bh + _unsigned_less(low, al).to(tl.int32), low


@triton.jit
def _uniform32(a):
    # Both terms fit positive int32; no unsigned float conversion is needed.
    value = ((a >> 16) & 65535).to(tl.float32) * 65536.0 + (a & 65535).to(tl.float32)
    return value * (1.0 / 4294967087.0)


@triton.jit
def _finish_small_high(high, low, C: tl.constexpr):
    """Reduce low + high*C when high*C < 2**31 and at most one modulus fits."""
    correction = high * C
    result = (low + correction).to(tl.int32)
    subtract_modulus = _unsigned_less(result, low) | (~_unsigned_less(result, -C))
    return tl.where(subtract_modulus, result + C, result).to(tl.int32)


@triton.jit
def _reduce_m2(high, low):
    # high <= 1898190, so low16*22853 < 2**31 and high16*22853 < 2**20.
    lower = (high & 65535) * 22853
    upper = (high >> 16) * 22853
    folded_low = ((upper << 16) + lower).to(tl.int32)
    folded_high = (upper >> 16) + _unsigned_less(folded_low, lower).to(tl.int32)
    result = (low + folded_low).to(tl.int32)
    folded_high += _unsigned_less(result, low).to(tl.int32)
    # folded_high <= 11; the last fold/canonical reduction needs one multiply.
    return _finish_small_high(folded_high, result, 22853)


@triton.jit
def step_bounded(x0, x1, x2, y0, y1, y2, JOINT: tl.constexpr = False):
    if JOINT:
        high, low = _sum_coefficient_products(x1, (-209 - x0).to(tl.int32), 1403580, 810728)
    else:
        h0, l0 = _multiply_coefficient(x1, 1403580)
        h1, l1 = _multiply_coefficient((-209 - x0).to(tl.int32), 810728)
        high, low = _add_words(h0, l0, h1, l1)
    # high <= 2214307, so high*209 <= 462790163 < 2**29.
    x = _finish_small_high(high, low, 209)
    if JOINT:
        high, low = _sum_coefficient_products(y2, (-22853 - y0).to(tl.int32), 527612, 1370589)
    else:
        h0, l0 = _multiply_coefficient(y2, 527612)
        h1, l1 = _multiply_coefficient((-22853 - y0).to(tl.int32), 1370589)
        high, low = _add_words(h0, l0, h1, l1)
    y = _reduce_m2(high, low)
    z = (x - y).to(tl.int32)
    z = tl.where(_unsigned_less(y, x), z, z - 209).to(tl.int32)
    return z, x1, x2, x, y1, y2, y


@triton.jit
def _generate32_round(
    x0,
    x1,
    x2,
    y0,
    y1,
    y2,
    OUT,
    SPARE,
    N,
    CURSOR,
    SPARE_INDEX,
    MEAN,
    STD,
    physical_tile,
    physical_base,
    slot,
    present,
    batch,
    count,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    DIST: tl.constexpr,
    HAS_SPARE: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
    EMIT_SPARE: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr,
    JOINT: tl.constexpr,
    COMPLETE: tl.constexpr,
):
    index = slot + batch * S
    active = present & (index < count)
    (a, ax, bx, cx, dy, ey, fy) = step_bounded(x0, x1, x2, y0, y1, y2, JOINT)
    if DIST == 0 or DIST == 1:
        value = a if DIST == 0 else _uniform32(a)
        _store_rotation(value, OUT, N, physical_base, CURSOR, batch, S, BLOCK, 1, 0, CURSOR_ALIGNED, OUTPUT_BLOCK)
    else:
        (b, ax, bx, cx, dy, ey, fy) = step_bounded(ax, bx, cx, dy, ey, fy, JOINT)
        radius = tl.sqrt(-2.0 * tl.log(_uniform32(a)))
        (sine, cosine) = sincos_turns(_uniform32(b), False)
        (z0, z1) = (radius * cosine, radius * sine)
        (v0, v1) = (z0 * STD + MEAN, z1 * STD + MEAN)
        if DIST == 3:
            (v0, v1) = (tl.exp(v0), tl.exp(v1))
        i = 2 * index + int(HAS_SPARE)
        _store_rotation(
            tl.interleave(v0, v1),
            OUT,
            N,
            physical_base,
            CURSOR,
            batch,
            S,
            BLOCK,
            2,
            int(HAS_SPARE),
            CURSOR_ALIGNED,
            OUTPUT_BLOCK,
        )
        if EMIT_SPARE:
            spare_lane = active & (i + 1 == N)
            spare_bits = tl.sum(tl.where(spare_lane, z1.to(tl.int32, bitcast=True), 0), 0)
            spare_slot = (N - int(HAS_SPARE)) // 2
            spare_stream = (CURSOR + spare_slot) % S
            owns_spare = ((N - int(HAS_SPARE)) % 2 != 0) & (physical_tile == spare_stream // BLOCK)
            owns_spare = owns_spare & (batch == spare_slot // S)
            tl.store(SPARE + 1 - SPARE_INDEX, spare_bits.to(tl.float32, bitcast=True), owns_spare)
    if COMPLETE:
        (x0, x1, x2, y0, y1, y2) = (ax, bx, cx, dy, ey, fy)
    else:
        (x0, x1, x2) = (tl.where(active, ax, x0), tl.where(active, bx, x1), tl.where(active, cx, x2))
        (y0, y1, y2) = (tl.where(active, dy, y0), tl.where(active, ey, y1), tl.where(active, fy, y2))
    return (x0, x1, x2, y0, y1, y2)


@triton.jit
def generate32_kernel(
    STATE,
    OUT,
    SPARE,
    N,
    CURSOR,
    SPARE_INDEX,
    MEAN,
    STD,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    DIST: tl.constexpr,
    HAS_SPARE: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
    EMIT_SPARE: tl.constexpr,
    FULL_ROUNDS: tl.constexpr = False,
    OUTPUT_BLOCK: tl.constexpr = 0,
    JOINT: tl.constexpr = False,
):
    # Scalar tile rotation; every state row remains unit-stride within a CTA.
    physical_tile = (CURSOR // BLOCK + tl.program_id(0)) % tl.cdiv(S, BLOCK)
    physical_base = physical_tile * BLOCK
    stream = physical_base + tl.arange(0, BLOCK)
    rotated = stream - CURSOR
    slot = tl.where(rotated < 0, rotated + S, rotated)
    if DIST >= 2:
        count = (N - int(HAS_SPARE) + 1) // 2
        if HAS_SPARE:
            z = tl.load(SPARE + SPARE_INDEX)
            v = z * STD + MEAN
            if DIST == 3:
                v = tl.exp(v)
            tl.store(OUT, v, tl.program_id(0) == 0)
    else:
        count = N
    present = stream < S
    x0 = tl.load(STATE + stream, present, other=1)
    x1 = tl.load(STATE + S + stream, present, other=1)
    x2 = tl.load(STATE + 2 * S + stream, present, other=1)
    y0 = tl.load(STATE + 3 * S + stream, present, other=1)
    y1 = tl.load(STATE + 4 * S + stream, present, other=1)
    y2 = tl.load(STATE + 5 * S + stream, present, other=1)
    if JOINT and not FULL_ROUNDS:
        full_batches = count // S
        for batch in range(full_batches):
            (x0, x1, x2, y0, y1, y2) = _generate32_round(
                x0,
                x1,
                x2,
                y0,
                y1,
                y2,
                OUT,
                SPARE,
                N,
                CURSOR,
                SPARE_INDEX,
                MEAN,
                STD,
                physical_tile,
                physical_base,
                slot,
                present,
                batch,
                count,
                S,
                BLOCK,
                DIST,
                HAS_SPARE,
                CURSOR_ALIGNED,
                EMIT_SPARE,
                OUTPUT_BLOCK,
                JOINT,
                True,
            )
        if not FULL_ROUNDS:
            remainder = count - full_batches * S
            tail_end = CURSOR + remainder
            physical_end = tl.minimum(physical_base + BLOCK, S)
            intersects = (physical_base < tl.minimum(tail_end, S)) & (physical_end > CURSOR) | (tail_end > S) & (
                physical_base < tail_end - S
            )
            if (remainder > 0) & intersects:
                (x0, x1, x2, y0, y1, y2) = _generate32_round(
                    x0,
                    x1,
                    x2,
                    y0,
                    y1,
                    y2,
                    OUT,
                    SPARE,
                    N,
                    CURSOR,
                    SPARE_INDEX,
                    MEAN,
                    STD,
                    physical_tile,
                    physical_base,
                    slot,
                    present,
                    full_batches,
                    count,
                    S,
                    BLOCK,
                    DIST,
                    HAS_SPARE,
                    CURSOR_ALIGNED,
                    EMIT_SPARE,
                    OUTPUT_BLOCK,
                    JOINT,
                    False,
                )
    else:
        for batch in range(tl.cdiv(count, S)):
            index = slot + batch * S
            active = present & (index < count)
            a, ax, bx, cx, dy, ey, fy = step_bounded(x0, x1, x2, y0, y1, y2, JOINT)
            if DIST == 0 or DIST == 1:
                value = a if DIST == 0 else _uniform32(a)
                _store_rotation(
                    value, OUT, N, physical_base, CURSOR, batch, S, BLOCK, 1, 0, CURSOR_ALIGNED, OUTPUT_BLOCK
                )
            else:
                b, ax, bx, cx, dy, ey, fy = step_bounded(ax, bx, cx, dy, ey, fy, JOINT)
                radius = tl.sqrt(-2.0 * tl.log(_uniform32(a)))
                sine, cosine = sincos_turns(_uniform32(b), False)
                z0, z1 = radius * cosine, radius * sine
                v0, v1 = z0 * STD + MEAN, z1 * STD + MEAN
                if DIST == 3:
                    v0, v1 = tl.exp(v0), tl.exp(v1)
                i = 2 * index + int(HAS_SPARE)
                _store_rotation(
                    tl.interleave(v0, v1),
                    OUT,
                    N,
                    physical_base,
                    CURSOR,
                    batch,
                    S,
                    BLOCK,
                    2,
                    int(HAS_SPARE),
                    CURSOR_ALIGNED,
                    OUTPUT_BLOCK,
                )
                if EMIT_SPARE:
                    # At most one lane owns the spare. Sum integer bits with zeros,
                    # preserving every FP32 bit including the sign of zero.
                    spare_lane = active & (i + 1 == N)
                    spare_bits = tl.sum(tl.where(spare_lane, z1.to(tl.int32, bitcast=True), 0), 0)
                    spare_slot = (N - int(HAS_SPARE)) // 2
                    spare_stream = (CURSOR + spare_slot) % S
                    owns_spare = ((N - int(HAS_SPARE)) % 2 != 0) & (physical_tile == spare_stream // BLOCK)
                    owns_spare = owns_spare & (batch == spare_slot // S)
                    tl.store(SPARE + 1 - SPARE_INDEX, spare_bits.to(tl.float32, bitcast=True), owns_spare)
            if FULL_ROUNDS:
                x0, x1, x2, y0, y1, y2 = ax, bx, cx, dy, ey, fy
            else:
                x0, x1, x2 = (
                    tl.where(active, ax, x0),
                    tl.where(active, bx, x1),
                    tl.where(active, cx, x2),
                )
                y0, y1, y2 = (
                    tl.where(active, dy, y0),
                    tl.where(active, ey, y1),
                    tl.where(active, fy, y2),
                )
    tl.store(STATE + stream, x0, present)
    tl.store(STATE + S + stream, x1, present)
    tl.store(STATE + 2 * S + stream, x2, present)
    tl.store(STATE + 3 * S + stream, y0, present)
    tl.store(STATE + 4 * S + stream, y1, present)
    tl.store(STATE + 5 * S + stream, y2, present)
