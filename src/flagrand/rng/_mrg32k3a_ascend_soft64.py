"""Ascend MRG rational FP64 uniforms and staged standard normal transforms.

Each normal pair consumes four consecutive steps in one stream. Rounded
binary64 standard normals are shared by immediate output and the spare;
lognormal applies exp only after that rounding.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from ._mrg32k3a_ascend_active import step_bounded
from ._mrg32k3a_ascend_store import _store_rotation
from ._xorwow._binding import bind_kernel
from ._xorwow.ascend_bounded_math import bind_bounded_math


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
def rational_uniform():
    math = bind_bounded_math()
    mul, add, renorm = math.mul, math.add, math.renorm

    def uniform(a, b):
        endpoint = (a == -209) & (b == -209)
        a = (a - 1).to(tl.int32)
        ah = ((a >> 16) & 65535).to(tl.float32) * (2.0**-16)
        al = (a & 65535).to(tl.float32) * (2.0**-32)
        bh = ((b >> 16) & 65535).to(tl.float32) * (2.0**-16)
        bl = (b & 65535).to(tl.float32) * (2.0**-32)
        zero = tl.full(a.shape, 0.0, tl.float32)
        c0 = tl.full(a.shape, 1.0, tl.float32)
        c1 = tl.full(a.shape, 4.866160807637243e-8, tl.float32)
        c2 = tl.full(a.shape, -1.1847618181044166e-15, tl.float32)
        # Digit splitting is exact but not a normalized floating expansion.
        # In particular, a leading zero must not leave the value in its middle
        # component when the multiplication drops fourth-order cross terms.
        a0, a1, a2 = renorm(ah, al, zero)
        b0, b1, b2 = renorm(bh, bl, zero)
        b0, b1, b2 = mul(b0, b1, b2, c0, c1, c2)
        u0, u1, u2 = add(a0, a1, a2, b0 * (2.0**-32), b1 * (2.0**-32), b2 * (2.0**-32))
        u0, u1, u2 = mul(u0, u1, u2, c0, c1, c2)
        return tl.where(endpoint, 1.0, u0), tl.where(endpoint, 0.0, u1), tl.where(endpoint, 0.0, u2)

    return bind_kernel(uniform)


@lru_cache(maxsize=1)
def uniform64_kernel():
    soft = bind_bounded_math()
    uniform64 = rational_uniform()
    normalize64, pack64 = soft.normalize_scaled, soft.pack_binary64

    def uniform_segments(
        STATE,
        OUT,
        N,
        CURSOR,
        S: tl.constexpr,
        BLOCK: tl.constexpr,
        CURSOR_ALIGNED: tl.constexpr,
    ):
        physical_tile = (CURSOR // BLOCK + tl.program_id(0)) % tl.cdiv(S, BLOCK)
        physical_base = physical_tile * BLOCK
        stream = physical_base + tl.arange(0, BLOCK)
        rotated = stream - CURSOR
        slot = tl.where(rotated < 0, rotated + S, rotated)
        present = stream < S
        x0 = tl.load(STATE + stream, present, other=1)
        x1 = tl.load(STATE + S + stream, present, other=1)
        x2 = tl.load(STATE + 2 * S + stream, present, other=1)
        y0 = tl.load(STATE + 3 * S + stream, present, other=1)
        y1 = tl.load(STATE + 4 * S + stream, present, other=1)
        y2 = tl.load(STATE + 5 * S + stream, present, other=1)
        for batch in range(tl.cdiv(N, S)):
            index = slot + batch * S
            active = present & (index < N)
            a, ax, bx, cx, dy, ey, fy = step_bounded(x0, x1, x2, y0, y1, y2)
            b, ax, bx, cx, dy, ey, fy = step_bounded(ax, bx, cx, dy, ey, fy)
            # Preserve the production ordering and the original software-f64
            # rational uniform, normalization and final binary64 rounding.
            u0, u1, u2 = uniform64(a, b)
            u0, u1, u2, exponent = normalize64(u0, u1, u2, 0)
            low, high = pack64(u0, u1, u2, exponent)
            _store_rotation(
                tl.interleave(low, high), OUT, 2 * N, physical_base, CURSOR, batch, S, BLOCK, 2, 0, CURSOR_ALIGNED
            )
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

    return bind_kernel(uniform_segments, do_not_specialize=["N", "CURSOR"])


@triton.jit
def _store_plane(
    OUT,
    value,
    P,
    CURSOR,
    physical_base,
    batch,
    S: tl.constexpr,
    B: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
):
    _store_rotation(value, OUT, P, physical_base, CURSOR, batch, S, B, 1, 0, CURSOR_ALIGNED)


@triton.jit(do_not_specialize=["P", "CURSOR"])
def raw_four_kernel(
    STATE,
    RAW,
    P,
    CURSOR,
    S: tl.constexpr,
    B: tl.constexpr,
    CURSOR_ALIGNED: tl.constexpr,
):
    physical_tile = (CURSOR // B + tl.program_id(0)) % tl.cdiv(S, B)
    physical_base = physical_tile * B
    stream = physical_base + tl.arange(0, B)
    rotated = stream - CURSOR
    slot = tl.where(rotated < 0, rotated + S, rotated)
    present = stream < S
    x0 = tl.load(STATE + stream, present, other=1)
    x1 = tl.load(STATE + S + stream, present, other=1)
    x2 = tl.load(STATE + 2 * S + stream, present, other=1)
    y0 = tl.load(STATE + 3 * S + stream, present, other=1)
    y1 = tl.load(STATE + 4 * S + stream, present, other=1)
    y2 = tl.load(STATE + 5 * S + stream, present, other=1)
    for batch in range(tl.cdiv(P, S)):
        index = slot + batch * S
        active = present & (index < P)
        a, ax, bx, cx, dy, ey, fy = step_bounded(x0, x1, x2, y0, y1, y2)
        b, ax, bx, cx, dy, ey, fy = step_bounded(ax, bx, cx, dy, ey, fy)
        c, ax, bx, cx, dy, ey, fy = step_bounded(ax, bx, cx, dy, ey, fy)
        d, ax, bx, cx, dy, ey, fy = step_bounded(ax, bx, cx, dy, ey, fy)
        _store_plane(RAW, a, P, CURSOR, physical_base, batch, S, B, CURSOR_ALIGNED)
        _store_plane(RAW + P, b, P, CURSOR, physical_base, batch, S, B, CURSOR_ALIGNED)
        _store_plane(RAW + 2 * P, c, P, CURSOR, physical_base, batch, S, B, CURSOR_ALIGNED)
        _store_plane(RAW + 3 * P, d, P, CURSOR, physical_base, batch, S, B, CURSOR_ALIGNED)
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


@lru_cache(maxsize=1)
def normal64_kernels():
    soft = bind_bounded_math()
    uniform = rational_uniform()
    log, sqrt, sincos = soft.log, soft.sqrt, soft.sincos_turns
    mul_scaled, pack, exp = soft.mul_scaled, soft.pack_binary64, soft.exp_from_scaled

    def radius(RAW, R, P, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        a = tl.load(RAW + i, i < P, other=1)
        b = tl.load(RAW + P + i, i < P, other=1)
        u0, u1, u2 = uniform(a, b)
        l0, l1, l2 = log(u0, u1, u2)
        r0, r1, r2 = sqrt(-2.0 * l0, -2.0 * l1, -2.0 * l2)
        tl.store(R + i, r0, i < P)
        tl.store(R + P + i, r1, i < P)
        tl.store(R + 2 * P + i, r2, i < P)

    def angle(RAW, T, P, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        c = tl.load(RAW + 2 * P + i, i < P, other=1)
        d = tl.load(RAW + 3 * P + i, i < P, other=1)
        t0, t1, t2 = uniform(c, d)
        # MRG retains its rational turn grid and the original quadrant policy.
        s0, s1, s2, c0, c1, c2 = sincos(t0, t1, t2)
        tl.store(T + i, s0, i < P)
        tl.store(T + P + i, s1, i < P)
        tl.store(T + 2 * P + i, s2, i < P)
        tl.store(T + 3 * P + i, c0, i < P)
        tl.store(T + 4 * P + i, c1, i < P)
        tl.store(T + 5 * P + i, c2, i < P)

    def rounded_output(
        R,
        T,
        OUT,
        SPARE,
        N,
        P,
        SPARE_INDEX,
        B: tl.constexpr,
        HAS_SPARE: tl.constexpr,
        EMIT_SPARE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        i = pid * B + tl.arange(0, B)
        r0 = tl.load(R + i, i < P, other=0)
        r1 = tl.load(R + P + i, i < P, other=0)
        r2 = tl.load(R + 2 * P + i, i < P, other=0)
        s0 = tl.load(T + i, i < P, other=0)
        s1 = tl.load(T + P + i, i < P, other=0)
        s2 = tl.load(T + 2 * P + i, i < P, other=0)
        c0 = tl.load(T + 3 * P + i, i < P, other=0)
        c1 = tl.load(T + 4 * P + i, i < P, other=0)
        c2 = tl.load(T + 5 * P + i, i < P, other=0)
        z0, z1, z2, ze = mul_scaled(r0, r1, r2, c0, c1, c2)
        w0, w1, w2, we = mul_scaled(r0, r1, r2, s0, s1, s2)
        low0, high0 = pack(z0, z1, z2, ze)
        low1, high1 = pack(w0, w1, w2, we)
        words = tl.interleave(tl.interleave(low0, low1), tl.interleave(high0, high1))
        output_index = 4 * pid * B + 2 * int(HAS_SPARE) + tl.arange(0, 4 * B)
        tl.store(OUT + output_index, words, output_index < 2 * N)
        if HAS_SPARE:
            # Old and new spare slots differ, so no CTA waits on another CTA.
            old_low = tl.load(SPARE + 2 * SPARE_INDEX)
            old_high = tl.load(SPARE + 2 * SPARE_INDEX + 1)
            tl.store(OUT, old_low, pid == 0)
            tl.store(OUT + 1, old_high, pid == 0)
        if EMIT_SPARE:
            # One pair contributes integer words; one CTA performs scalar stores.
            cache_low = tl.sum(tl.where(i == P - 1, low1, 0), 0)
            cache_high = tl.sum(tl.where(i == P - 1, high1, 0), 0)
            owns_spare = pid == (P - 1) // B
            tl.store(SPARE + 2 * (1 - SPARE_INDEX), cache_low, owns_spare)
            tl.store(SPARE + 2 * (1 - SPARE_INDEX) + 1, cache_high, owns_spare)

    def lognormal(STANDARD, OUT, N, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        word_index = tl.program_id(0) * (2 * B) + tl.arange(0, 2 * B)
        words = tl.load(STANDARD + word_index, word_index < 2 * N, other=0)
        low, high = tl.split(tl.reshape(words, (B, 2)))
        z0, z1, z2, ze = _unpack_standard(low, high)
        e0, e1, e2, exponent = exp(z0, z1, z2, ze)
        low, high = pack(e0, e1, e2, exponent)
        tl.store(OUT + word_index, tl.interleave(low, high), word_index < 2 * N)

    return (
        raw_four_kernel,
        bind_kernel(radius, do_not_specialize=["P"]),
        bind_kernel(angle, do_not_specialize=["P"]),
        bind_kernel(rounded_output, do_not_specialize=["N", "P", "SPARE_INDEX"]),
        bind_kernel(lognormal, do_not_specialize=["N"]),
    )


class Scratch:
    """One current shape per engine; allocations stay in the public call."""

    def __init__(self, n, pairs, is_log, device):
        self.key = (n, pairs, is_log, device)
        self.raw = torch.empty((4, max(1, pairs)), dtype=torch.int32, device=device)
        self.radius = torch.empty((3, max(1, pairs)), dtype=torch.float32, device=device)
        self.angle = torch.empty((6, max(1, pairs)), dtype=torch.float32, device=device)
        self.standard = torch.empty(2 * n, dtype=torch.int32, device=device) if is_log else None
        self.stream = torch.npu.current_stream(device)

    def prepare_stream(self):
        current = torch.npu.current_stream(self.raw.device)
        if current != self.stream:
            self.raw.record_stream(current)
            self.radius.record_stream(current)
            self.angle.record_stream(current)
            if self.standard is not None:
                self.standard.record_stream(current)
            self.stream = current
