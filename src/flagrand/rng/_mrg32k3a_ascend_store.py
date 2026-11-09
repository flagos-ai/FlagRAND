"""A generic 1D physical-to-logical store with lane-zero source masks.

WIDTH is one for raw/planes/Poisson or two for paired values/binary64 words.
N and H are in output words/elements. No mathematical operation is changed.
"""

import triton
import triton.language as tl


@triton.jit
def _store_rotation(
    values,
    OUT,
    N,
    physical_base,
    CURSOR,
    batch,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
    OUTPUT_BLOCK: tl.constexpr = 0,
):
    lanes = tl.arange(0, WIDTH * BLOCK)
    available = WIDTH * tl.minimum(BLOCK, S - physical_base)
    if CURSOR_ALIGNED:
        logical_base = tl.where(physical_base < CURSOR, physical_base - CURSOR + S, physical_base - CURSOR)
        output_index = WIDTH * (logical_base + batch * S) + H + lanes
        tl.store(OUT + output_index, values, (lanes < available) & (output_index < N))
    else:
        split = (physical_base <= CURSOR) & (CURSOR < physical_base + BLOCK)
        if split:
            boundary = WIDTH * (CURSOR - physical_base)
            G: tl.constexpr = OUTPUT_BLOCK if OUTPUT_BLOCK > 0 else WIDTH * BLOCK
            head_lanes = tl.arange(0, G)
            indices = (head_lanes + boundary) & (WIDTH * BLOCK - 1)
            # The installed Ascend gather accepts fp32 payloads but rejects
            # int32. These casts only reinterpret the 32-bit payload; gather
            # performs no floating arithmetic on raw or packed binary64 words.
            rotated = tl.gather(values.to(tl.float32, bitcast=True), indices, axis=0).to(values.dtype, bitcast=True)
            head_index = WIDTH * batch * S + H + head_lanes
            tl.store(
                OUT + head_index,
                rotated,
                (head_lanes < available - boundary) & (head_index < N),
            )
            tail_index = WIDTH * ((batch + 1) * S) + H - boundary + lanes
            tl.store(OUT + tail_index, values, (lanes < boundary) & (tail_index < N))
        else:
            logical_base = tl.where(
                physical_base < CURSOR,
                physical_base - CURSOR + S,
                physical_base - CURSOR,
            )
            output_index = WIDTH * (logical_base + batch * S) + H + lanes
            tl.store(OUT + output_index, values, (lanes < available) & (output_index < N))
