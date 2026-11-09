"""MRG32K3A on Ascend using exact signed-32-bit modular arithmetic.

Each state word stores an unsigned residue in an int32 tensor. Products use
16-bit digits; folding 2**32 == c modulo 2**32-c avoids vector int64 casts.
The stream, cursor and paired-normal cache follow the common generator.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._xorwow.native_math import sincos_turns


@triton.jit
def _unsigned_less(a, b):
    return (a ^ -2147483648) < (b ^ -2147483648)


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
def _add_words(ah, al, bh, bl):
    low = (al + bl).to(tl.int32)
    return ah + bh + _unsigned_less(low, al).to(tl.int32), low


@triton.jit
def _fold(high, low, C: tl.constexpr):
    h, l = _multiply_words(high, C)
    return _add_words(h, l, 0, low)


@triton.jit
def _reduce(high, low, C: tl.constexpr, FOLDS: tl.constexpr):
    for _ in tl.static_range(FOLDS):
        high, low = _fold(high, low, C)
    # The recurrence bounds leave high=0 and low < 2**32 after these folds.
    return tl.where(_unsigned_less(low, -C), low, low + C).to(tl.int32)


@triton.jit
def _step(x0, x1, x2, y0, y1, y2):
    h0, l0 = _multiply_words(x1, 1403580)
    h1, l1 = _multiply_words((-209 - x0).to(tl.int32), 810728)
    h, l = _add_words(h0, l0, h1, l1)
    x = _reduce(h, l, 209, 2)
    h0, l0 = _multiply_words(y2, 527612)
    h1, l1 = _multiply_words((-22853 - y0).to(tl.int32), 1370589)
    h, l = _add_words(h0, l0, h1, l1)
    y = _reduce(h, l, 22853, 3)
    z = (x - y).to(tl.int32)
    z = tl.where(_unsigned_less(y, x), z, z - 209).to(tl.int32)
    return z, x1, x2, x, y1, y2, y


@triton.jit
def _uniform32(a):
    # Both terms fit positive int32; no unsigned float conversion is needed.
    value = ((a >> 16) & 65535).to(tl.float32) * 65536.0 + (a & 65535).to(tl.float32)
    return value * (1.0 / 4294967087.0)


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
def _generate32(
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
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
    stream = (CURSOR + slot) % S
    present = (slot < S) & (slot < count)
    x0 = tl.load(STATE + stream, present, other=1)
    x1 = tl.load(STATE + S + stream, present, other=1)
    x2 = tl.load(STATE + 2 * S + stream, present, other=1)
    y0 = tl.load(STATE + 3 * S + stream, present, other=1)
    y1 = tl.load(STATE + 4 * S + stream, present, other=1)
    y2 = tl.load(STATE + 5 * S + stream, present, other=1)
    for batch in range(tl.cdiv(count, S)):
        index = slot + batch * S
        active = present & (index < count)
        a, ax, bx, cx, dy, ey, fy = _step(x0, x1, x2, y0, y1, y2)
        if DIST == 0:
            tl.store(OUT + index, a, active)
        elif DIST == 1:
            tl.store(OUT + index, _uniform32(a), active)
        else:
            b, ax, bx, cx, dy, ey, fy = _step(ax, bx, cx, dy, ey, fy)
            radius = tl.sqrt(-2.0 * tl.log(_uniform32(a)))
            sine, cosine = sincos_turns(_uniform32(b), False)
            z0, z1 = radius * cosine, radius * sine
            v0, v1 = z0 * STD + MEAN, z1 * STD + MEAN
            if DIST == 3:
                v0, v1 = tl.exp(v0), tl.exp(v1)
            i = 2 * index + int(HAS_SPARE)
            tl.store(OUT + i, v0, active & (i < N))
            tl.store(OUT + i + 1, v1, active & (i + 1 < N))
            tl.store(SPARE + 1 - SPARE_INDEX + tl.zeros((BLOCK,), tl.int32), z1, active & (i + 1 == N))
        x0, x1, x2 = tl.where(active, ax, x0), tl.where(active, bx, x1), tl.where(active, cx, x2)
        y0, y1, y2 = tl.where(active, dy, y0), tl.where(active, ey, y1), tl.where(active, fy, y2)
    tl.store(STATE + stream, x0, present)
    tl.store(STATE + S + stream, x1, present)
    tl.store(STATE + 2 * S + stream, x2, present)
    tl.store(STATE + 3 * S + stream, y0, present)
    tl.store(STATE + 4 * S + stream, y1, present)
    tl.store(STATE + 5 * S + stream, y2, present)


@triton.jit
def _poisson(
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
):
    slot = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    stream = (CURSOR + slot) % S
    present = (slot < S) & (slot < 2 * N)
    x0 = tl.load(STATE + stream, present, other=1)
    x1 = tl.load(STATE + S + stream, present, other=1)
    x2 = tl.load(STATE + 2 * S + stream, present, other=1)
    y0 = tl.load(STATE + 3 * S + stream, present, other=1)
    y1 = tl.load(STATE + 4 * S + stream, present, other=1)
    y2 = tl.load(STATE + 5 * S + stream, present, other=1)
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for batch in range(tl.cdiv(2 * N, S)):
        active = present & (batch * S + slot < 2 * N)
        word, a, b, c, d, e, f = _step(x0, x1, x2, y0, y1, y2)
        first, second = tl.split(tl.reshape(word, (BLOCK, 2)))
        value = _alias(first, second, THRESHOLD, ALIASES, FIRST, SIZE)
        index = batch * (S // 2) + pair
        tl.store(OUT + index, value, (pair < S // 2) & (index < N))
        x0, x1, x2 = tl.where(active, a, x0), tl.where(active, b, x1), tl.where(active, c, x2)
        y0, y1, y2 = tl.where(active, d, y0), tl.where(active, e, y1), tl.where(active, f, y2)
    tl.store(STATE + stream, x0, present)
    tl.store(STATE + S + stream, x1, present)
    tl.store(STATE + 2 * S + stream, x2, present)
    tl.store(STATE + 3 * S + stream, y0, present)
    tl.store(STATE + 4 * S + stream, y1, present)
    tl.store(STATE + 5 * S + stream, y2, present)


@triton.jit
def _raw_poisson(RAW, OUT, THRESHOLD, ALIASES, N, FIRST: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(RAW + 2 * i, i < N, other=1)
    b = tl.load(RAW + 2 * i + 1, i < N, other=1)
    value = _alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE)
    tl.store(OUT + i, value, i < N)


def _launch_poisson_fallback(engine, out, threshold, aliases, first, size, block_size=128, num_warps=4):
    state = engine.ensure_state(out.device)
    if engine.nstreams % 2:
        # An odd stream count pairs across alternating batch boundaries.
        # Generate the exact interleaved raw sequence before alias conversion.
        raw = torch.empty(2 * out.numel(), dtype=torch.int32, device=out.device)
        _generate32[(triton.cdiv(min(raw.numel(), engine.nstreams), block_size),)](
            state,
            raw,
            state,
            raw.numel(),
            engine._cursor,
            0,
            0.0,
            1.0,
            engine.nstreams,
            block_size,
            0,
            False,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        _raw_poisson[(triton.cdiv(out.numel(), block_size),)](
            raw, out, threshold, aliases, out.numel(), first, size, block_size, num_warps=num_warps
        )
        return out
    _poisson[(triton.cdiv(min(out.numel(), engine.nstreams // 2), block_size),)](
        state,
        out,
        threshold,
        aliases,
        out.numel(),
        engine._cursor,
        engine.nstreams,
        first,
        size,
        block_size,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _unpack_standard(low, high):
    # A Box-Muller standard normal is zero or a normal binary64 value.
    # Retain all 53 significand bits as exact 21+16+16-bit components.
    nonzero = ((high & 0x7FFFFFFF) != 0) | (low != 0)
    sign = tl.where(high < 0, -1.0, 1.0)
    a = ((high & 0xFFFFF) | 0x100000).to(tl.float32) * (2.0**-20)
    b = ((low >> 16) & 65535).to(tl.float32) * (2.0**-36)
    c = (low & 65535).to(tl.float32) * (2.0**-52)
    exponent = ((high >> 20) & 2047) - 1023
    return tl.where(nonzero, a * sign, 0.0), b * sign, c * sign, tl.where(nonzero, exponent, 0)


@lru_cache(maxsize=1)
def _soft_kernels():
    from ._xorwow._binding import bind_kernel
    from ._xorwow.ascend_soft64_math import bind_ascend_soft64

    soft = bind_ascend_soft64(runtime_loops=True, direct_product=True)
    normalize64, pack64 = soft.normalize_scaled, soft.pack_binary64
    log64, sqrt64, sincos64 = soft.log, soft.sqrt, soft.sincos_turns
    mul_scaled64, affine64, exp64 = soft.mul_scaled, soft.affine_scaled, soft.exp_from_scaled

    # The same normalized rational grid serves standard and affine FP64 paths.
    from ._mrg32k3a_ascend_soft64 import rational_uniform

    uniform64 = rational_uniform()

    def transform(low, high, PARAMETERS, DIST: tl.constexpr, STANDARD: tl.constexpr):
        z0, z1, z2, ze = _unpack_standard(low, high)
        if STANDARD:
            a, b, c, e = z0, z1, z2, ze
        else:
            mu0 = tl.load(PARAMETERS).to(tl.float32, bitcast=True)
            mu1 = tl.load(PARAMETERS + 1).to(tl.float32, bitcast=True)
            mu2 = tl.load(PARAMETERS + 2).to(tl.float32, bitcast=True)
            mue = tl.load(PARAMETERS + 3)
            sd0 = tl.load(PARAMETERS + 4).to(tl.float32, bitcast=True)
            sd1 = tl.load(PARAMETERS + 5).to(tl.float32, bitcast=True)
            sd2 = tl.load(PARAMETERS + 6).to(tl.float32, bitcast=True)
            sde = tl.load(PARAMETERS + 7)
            a, b, c, e = affine64(z0, z1, z2, ze, mu0, mu1, mu2, mue, sd0, sd1, sd2, sde)
        if DIST == 3:
            a, b, c, e = exp64(a, b, c, e)
        return pack64(a, b, c, e)

    transform = bind_kernel(transform)

    def normal_words(a, b, c, d, PARAMETERS, DIST: tl.constexpr, STANDARD: tl.constexpr):
        u0, u1, u2 = uniform64(a, b)
        t0, t1, t2 = uniform64(c, d)
        l0, l1, l2 = log64(u0, u1, u2)
        r0, r1, r2 = sqrt64(-2.0 * l0, -2.0 * l1, -2.0 * l2)
        s0, s1, s2, c0, c1, c2 = sincos64(t0, t1, t2)
        z0, z1, z2, ze = mul_scaled64(r0, r1, r2, c0, c1, c2)
        w0, w1, w2, we = mul_scaled64(r0, r1, r2, s0, s1, s2)
        # Cache and immediate outputs use the same rounded standard normal.
        low0, high0 = pack64(z0, z1, z2, ze)
        low1, high1 = pack64(w0, w1, w2, we)
        if STANDARD and DIST == 2:
            lo0, hi0, lo1, hi1 = low0, high0, low1, high1
        else:
            # Keep each elementwise software-f64 transform one-dimensional.
            lo0, hi0 = transform(low0, high0, PARAMETERS, DIST, STANDARD)
            lo1, hi1 = transform(low1, high1, PARAMETERS, DIST, STANDARD)
        return lo0, hi0, lo1, hi1, low1, high1

    normal_words = bind_kernel(normal_words)

    def generate64(
        STATE,
        OUT,
        SPARE,
        PARAMETERS,
        N,
        CURSOR,
        SPARE_INDEX,
        S: tl.constexpr,
        BLOCK: tl.constexpr,
        DIST: tl.constexpr,
        HAS_SPARE: tl.constexpr,
        STANDARD: tl.constexpr,
    ):
        slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if DIST >= 2:
            count = (N - int(HAS_SPARE) + 1) // 2
        else:
            count = N
        stream = (CURSOR + slot) % S
        present = (slot < S) & (slot < count)
        x0 = tl.load(STATE + stream, present, other=1)
        x1 = tl.load(STATE + S + stream, present, other=1)
        x2 = tl.load(STATE + 2 * S + stream, present, other=1)
        y0 = tl.load(STATE + 3 * S + stream, present, other=1)
        y1 = tl.load(STATE + 4 * S + stream, present, other=1)
        y2 = tl.load(STATE + 5 * S + stream, present, other=1)
        for batch in range(tl.cdiv(count, S)):
            index = slot + batch * S
            active = present & (index < count)
            a, ax, bx, cx, dy, ey, fy = _step(x0, x1, x2, y0, y1, y2)
            b, ax, bx, cx, dy, ey, fy = _step(ax, bx, cx, dy, ey, fy)
            if DIST == 1:
                u0, u1, u2 = uniform64(a, b)
                u0, u1, u2, exponent = normalize64(u0, u1, u2, 0)
                low, high = pack64(u0, u1, u2, exponent)
                tl.store(OUT + 2 * index, low, active)
                tl.store(OUT + 2 * index + 1, high, active)
            else:
                c, ax, bx, cx, dy, ey, fy = _step(ax, bx, cx, dy, ey, fy)
                d, ax, bx, cx, dy, ey, fy = _step(ax, bx, cx, dy, ey, fy)
                lo0, hi0, lo1, hi1, cache_lo, cache_hi = normal_words(a, b, c, d, PARAMETERS, DIST, STANDARD)
                i = 2 * index + int(HAS_SPARE)
                tl.store(OUT + 2 * i, lo0, active & (i < N))
                tl.store(OUT + 2 * i + 1, hi0, active & (i < N))
                tl.store(OUT + 2 * i + 2, lo1, active & (i + 1 < N))
                tl.store(OUT + 2 * i + 3, hi1, active & (i + 1 < N))
                cache_lane = tl.zeros((BLOCK,), tl.int32)
                tl.store(SPARE + 2 * (1 - SPARE_INDEX) + cache_lane, cache_lo, active & (i + 1 == N))
                tl.store(SPARE + 2 * (1 - SPARE_INDEX) + 1 + cache_lane, cache_hi, active & (i + 1 == N))
            x0, x1, x2 = tl.where(active, ax, x0), tl.where(active, bx, x1), tl.where(active, cx, x2)
            y0, y1, y2 = tl.where(active, dy, y0), tl.where(active, ey, y1), tl.where(active, fy, y2)
        tl.store(STATE + stream, x0, present)
        tl.store(STATE + S + stream, x1, present)
        tl.store(STATE + 2 * S + stream, x2, present)
        tl.store(STATE + 3 * S + stream, y0, present)
        tl.store(STATE + 4 * S + stream, y1, present)
        tl.store(STATE + 5 * S + stream, y2, present)

    def spare64(OUT, SPARE, PARAMETERS, SPARE_INDEX, DIST: tl.constexpr, STANDARD: tl.constexpr):
        i = tl.arange(0, 32)
        low = tl.load(SPARE + 2 * SPARE_INDEX + tl.zeros((32,), tl.int32))
        high = tl.load(SPARE + 2 * SPARE_INDEX + 1 + tl.zeros((32,), tl.int32))
        if not (STANDARD and DIST == 2):
            low, high = transform(low, high, PARAMETERS, DIST, STANDARD)
        tl.store(OUT + 2 * i, low, i == 0)
        tl.store(OUT + 2 * i + 1, high, i == 0)

    return (
        bind_kernel(generate64, do_not_specialize=["N", "CURSOR", "SPARE_INDEX"]),
        bind_kernel(spare64, do_not_specialize=["SPARE_INDEX"]),
    )


def _generate_fallback(engine, out, distribution, *, mean=0.0, stddev=1.0, **kwargs):
    n, f64, paired = out.numel(), out.dtype == torch.float64, distribution >= 2
    state = engine.ensure_state(out.device)
    has_spare, spare_index, spare = False, 0, state
    if paired:
        if f64 not in engine._normal_cache:
            engine._normal_cache[f64] = torch.empty(2, dtype=out.dtype, device=out.device)
        spare = engine._normal_cache[f64]
        has_spare = engine._normal_has_spare.get(f64, False)
        spare_index = engine._normal_spare_index.get(f64, 0)
    count = (n - int(has_spare) + 1) // 2 if paired else n
    block = kwargs.get("block_size", kwargs.get("block", (32 if paired else 64) if f64 else (64 if paired else 256)))
    grid = (triton.cdiv(max(1, min(count, engine.nstreams)), block),)
    options = dict(num_warps=kwargs.get("num_warps", 4), enable_fp_fusion=False, sync_solver=False)
    if f64:
        from ._xorwow.api import _float_bits, _soft_parameters

        kernel, spare_kernel = _soft_kernels()
        standard = mean == 0.0 and stddev == 1.0
        parameters = _soft_parameters(_float_bits(mean, True), _float_bits(stddev, True), str(out.device))
        words = out.view(torch.int32)
        spare_words = spare.view(torch.int32)
        if has_spare:
            spare_kernel[(1,)](words, spare_words, parameters, spare_index, distribution, standard, **options)
        if count:
            kernel[grid](
                state,
                words,
                spare_words,
                parameters,
                n,
                engine._cursor,
                spare_index,
                engine.nstreams,
                block,
                distribution,
                has_spare,
                standard,
                **options,
            )
    else:
        _generate32[grid](
            state,
            out,
            spare,
            n,
            engine._cursor,
            spare_index,
            float(mean),
            float(stddev),
            engine.nstreams,
            block,
            distribution,
            has_spare,
            **options,
        )
    if paired:
        new_spare = (n - int(has_spare)) % 2 != 0
        engine._normal_has_spare[f64] = new_spare
        if new_spare:
            engine._normal_spare_index[f64] = 1 - spare_index
    engine.offset += count * (2 if paired else 1) * (2 if f64 else 1)
    engine._state_offset = engine.offset
    engine._cursor = (engine._cursor + count) % engine.nstreams
    return out


def generate(engine, out, distribution, *, mean=0.0, stddev=1.0, **kwargs):
    from ._mrg32k3a_ascend_api import generate as launch

    return launch(engine, out, distribution, mean=mean, stddev=stddev, **kwargs)


def launch_poisson(engine, out, threshold, aliases, first, size, block_size=512, num_warps=4):
    from ._mrg32k3a_ascend_api import launch_poisson as launch

    return launch(engine, out, threshold, aliases, first, size, block_size, num_warps)
