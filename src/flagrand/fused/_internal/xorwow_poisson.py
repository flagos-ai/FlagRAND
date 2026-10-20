"""Fuse the existing XORWOW raw-word ordering with Poisson alias sampling.

For an even S streams, output i uses raw positions 2*i and 2*i+1, hence
streams 2*(i % (S/2)) and 2*(i % (S/2))+1 at round i//(S/2). Each lane
retains these two independent six-word states. The caller owns the scalar
offset update; this launcher only advances device state.
"""

import triton
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.rng._xorwow.kernels import _next


@triton.jit(do_not_specialize=["N"])
def _xorwow_poisson(
    STATE,
    OUT,
    THRESHOLD,
    ALIASES,
    N,
    NSTREAMS: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = 2 * p
    present = (p < NSTREAMS // 2) & (p < N)
    x0 = tl.load(STATE + s, present, other=0).to(tl.uint32, bitcast=True)
    y0 = tl.load(STATE + NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
    z0 = tl.load(STATE + 2 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
    w0 = tl.load(STATE + 3 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
    v0 = tl.load(STATE + 4 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
    d0 = tl.load(STATE + 5 * NSTREAMS + s, present, other=0).to(tl.uint32, bitcast=True)
    x1 = tl.load(STATE + s + 1, present, other=0).to(tl.uint32, bitcast=True)
    y1 = tl.load(STATE + NSTREAMS + s + 1, present, other=0).to(tl.uint32, bitcast=True)
    z1 = tl.load(STATE + 2 * NSTREAMS + s + 1, present, other=0).to(tl.uint32, bitcast=True)
    w1 = tl.load(STATE + 3 * NSTREAMS + s + 1, present, other=0).to(tl.uint32, bitcast=True)
    v1 = tl.load(STATE + 4 * NSTREAMS + s + 1, present, other=0).to(tl.uint32, bitcast=True)
    d1 = tl.load(STATE + 5 * NSTREAMS + s + 1, present, other=0).to(tl.uint32, bitcast=True)

    for batch in range(tl.cdiv(N, NSTREAMS // 2)):
        i = batch.to(tl.int64) * (NSTREAMS // 2) + p
        active = present & (i < N)
        a, nx0, ny0, nz0, nw0, nv0, nd0 = _next(x0, y0, z0, w0, v0, d0)
        b, nx1, ny1, nz1, nw1, nv1, nd1 = _next(x1, y1, z1, w1, v1, d1)
        value = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
        tl.store(OUT + i, value, active)
        x0 = tl.where(active, nx0, x0)
        y0 = tl.where(active, ny0, y0)
        z0 = tl.where(active, nz0, z0)
        w0 = tl.where(active, nw0, w0)
        v0 = tl.where(active, nv0, v0)
        d0 = tl.where(active, nd0, d0)
        x1 = tl.where(active, nx1, x1)
        y1 = tl.where(active, ny1, y1)
        z1 = tl.where(active, nz1, z1)
        w1 = tl.where(active, nw1, w1)
        v1 = tl.where(active, nv1, v1)
        d1 = tl.where(active, nd1, d1)

    tl.store(STATE + s, x0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + NSTREAMS + s, y0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 2 * NSTREAMS + s, z0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 3 * NSTREAMS + s, w0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 4 * NSTREAMS + s, v0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 5 * NSTREAMS + s, d0.to(tl.int32, bitcast=True), present)
    tl.store(STATE + s + 1, x1.to(tl.int32, bitcast=True), present)
    tl.store(STATE + NSTREAMS + s + 1, y1.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 2 * NSTREAMS + s + 1, z1.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 3 * NSTREAMS + s + 1, w1.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 4 * NSTREAMS + s + 1, v1.to(tl.int32, bitcast=True), present)
    tl.store(STATE + 5 * NSTREAMS + s + 1, d1.to(tl.int32, bitcast=True), present)


def launch_xorwow_poisson(engine, out, threshold, aliases, first, size, block_size=256, num_warps=4):
    """Launch for an even number of fixed XORWOW streams; do not change offset."""
    state = engine.ensure_state(out.device)
    n = out.numel()
    grid = (triton.cdiv(min(n, engine.nstreams // 2), block_size),)
    _xorwow_poisson[grid](
        state,
        out,
        threshold,
        aliases,
        n,
        engine.nstreams,
        first,
        size,
        block_size,
        num_warps=num_warps,
    )
    return out
