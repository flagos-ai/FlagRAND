"""Two independent MRG streams per lane avoid shared layout conversion."""

from dataclasses import dataclass, field
from functools import lru_cache

import triton
import triton.language as tl
from triton.runtime.jit import mangle_type

from flagrand.rng.mrg32k3a import _mrg32k3a_step, _mrg32k3a_step_u32
from flagrand.runtime.compiled_launcher import CachedKernelLauncher


@triton.jit
def sample_alias(r0, r1, threshold, aliases, FIRST: tl.constexpr, SIZE: tl.constexpr, RAW_RANGE: tl.constexpr):
    if RAW_RANGE == 4294967296:
        column = r0.to(tl.uint32) & (SIZE - 1)
        coin = r1.to(tl.uint32)
    elif RAW_RANGE == 4294967087 and SIZE <= 16777216:
        product = (r0.to(tl.uint64) - 1) * SIZE
        quotient = (product >> 32).to(tl.uint32)
        remainder = product.to(tl.uint32)
        # product = quotient * 2**32 + remainder. With q*209 < M1,
        # floor(product/M1) differs from quotient by at most one.
        column = quotient + (remainder >= 4294967087 - quotient * 209).to(tl.uint32)
        coin = r1.to(tl.uint32) - 1
    else:
        column = (((r0.to(tl.uint64) - 1) * SIZE) // RAW_RANGE).to(tl.int32)
        coin = r1.to(tl.uint32) - 1
    cutoff = tl.load(threshold + column).to(tl.uint32)
    other = tl.load(aliases + column)
    return FIRST + tl.where(coin < cutoff, column, other)


@triton.jit(do_not_specialize=["N", "CURSOR"])
def mrg_poisson_pair(
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
    STATE_U32: tl.constexpr = True,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s0 = (CURSOR + 2 * pair) % NSTREAMS
    s1 = (s0 + 1) % NSTREAMS
    present = (pair < NSTREAMS // 2) & (pair < N)
    a0 = tl.load(STATE + s0, present, other=1).to(tl.uint32)
    a1 = tl.load(STATE + NSTREAMS + s0, present, other=1).to(tl.uint32)
    a2 = tl.load(STATE + 2 * NSTREAMS + s0, present, other=1).to(tl.uint32)
    b0 = tl.load(STATE + 3 * NSTREAMS + s0, present, other=1).to(tl.uint32)
    b1 = tl.load(STATE + 4 * NSTREAMS + s0, present, other=1).to(tl.uint32)
    b2 = tl.load(STATE + 5 * NSTREAMS + s0, present, other=1).to(tl.uint32)
    c0 = tl.load(STATE + s1, present, other=1).to(tl.uint32)
    c1 = tl.load(STATE + NSTREAMS + s1, present, other=1).to(tl.uint32)
    c2 = tl.load(STATE + 2 * NSTREAMS + s1, present, other=1).to(tl.uint32)
    d0 = tl.load(STATE + 3 * NSTREAMS + s1, present, other=1).to(tl.uint32)
    d1 = tl.load(STATE + 4 * NSTREAMS + s1, present, other=1).to(tl.uint32)
    d2 = tl.load(STATE + 5 * NSTREAMS + s1, present, other=1).to(tl.uint32)
    if not STATE_U32:
        a0, a1, a2, b0, b1, b2 = (
            a0.to(tl.int64),
            a1.to(tl.int64),
            a2.to(tl.int64),
            b0.to(tl.int64),
            b1.to(tl.int64),
            b2.to(tl.int64),
        )
        c0, c1, c2, d0, d1, d2 = (
            c0.to(tl.int64),
            c1.to(tl.int64),
            c2.to(tl.int64),
            d0.to(tl.int64),
            d1.to(tl.int64),
            d2.to(tl.int64),
        )
    for batch in range(tl.cdiv(N, NSTREAMS // 2)):
        index = batch * (NSTREAMS // 2) + pair
        active = present & (index < N)
        if STATE_U32:
            w0, na0, na1, na2, nb0, nb1, nb2 = _mrg32k3a_step_u32(a0, a1, a2, b0, b1, b2)
            w1, nc0, nc1, nc2, nd0, nd1, nd2 = _mrg32k3a_step_u32(c0, c1, c2, d0, d1, d2)
        else:
            w0, na0, na1, na2, nb0, nb1, nb2 = _mrg32k3a_step(a0, a1, a2, b0, b1, b2)
            w1, nc0, nc1, nc2, nd0, nd1, nd2 = _mrg32k3a_step(c0, c1, c2, d0, d1, d2)
        value = sample_alias(w0, w1, THRESHOLD, ALIASES, FIRST, SIZE, 4294967087)
        tl.store(OUT + index, value, active)
        a0, a1, a2 = tl.where(active, na0, a0), tl.where(active, na1, a1), tl.where(active, na2, a2)
        b0, b1, b2 = tl.where(active, nb0, b0), tl.where(active, nb1, b1), tl.where(active, nb2, b2)
        c0, c1, c2 = tl.where(active, nc0, c0), tl.where(active, nc1, c1), tl.where(active, nc2, c2)
        d0, d1, d2 = tl.where(active, nd0, d0), tl.where(active, nd1, d1), tl.where(active, nd2, d2)
    tl.store(STATE + s0, a0.to(tl.uint32), present)
    tl.store(STATE + NSTREAMS + s0, a1.to(tl.uint32), present)
    tl.store(STATE + 2 * NSTREAMS + s0, a2.to(tl.uint32), present)
    tl.store(STATE + 3 * NSTREAMS + s0, b0.to(tl.uint32), present)
    tl.store(STATE + 4 * NSTREAMS + s0, b1.to(tl.uint32), present)
    tl.store(STATE + 5 * NSTREAMS + s0, b2.to(tl.uint32), present)
    tl.store(STATE + s1, c0.to(tl.uint32), present)
    tl.store(STATE + NSTREAMS + s1, c1.to(tl.uint32), present)
    tl.store(STATE + 2 * NSTREAMS + s1, c2.to(tl.uint32), present)
    tl.store(STATE + 3 * NSTREAMS + s1, d0.to(tl.uint32), present)
    tl.store(STATE + 4 * NSTREAMS + s1, d1.to(tl.uint32), present)
    tl.store(STATE + 5 * NSTREAMS + s1, d2.to(tl.uint32), present)


@dataclass
class _PreparedPairLaunch:
    launcher: CachedKernelLauncher
    grid: tuple
    n: int
    constants: tuple
    options: tuple
    context: object = field(default_factory=object)

    def launch(self, state, out, threshold, aliases, cursor):
        tensors = (state, out, threshold, aliases)
        args = tensors + (self.n, cursor)
        signature = (
            out.device,
            tuple((tensor.dtype, tensor.data_ptr()) for tensor in tensors),
            mangle_type(cursor),
            self.launcher._kernel.debug,
        )
        return self.launcher.launch(
            self.grid, args, self.constants, self.options, specialization_key=(self.context, signature)
        )


@lru_cache(maxsize=128)
def _prepared(device, n, nstreams, first, size, block_size, num_warps):
    launcher = CachedKernelLauncher(
        mrg_poisson_pair,
        constexpr_names=("NSTREAMS", "FIRST", "SIZE", "BLOCK", "STATE_U32"),
        option_names=("num_warps",),
    )
    return _PreparedPairLaunch(
        launcher,
        (triton.cdiv(min(n, nstreams // 2), block_size),),
        n,
        (nstreams, first, size, block_size, True),
        (num_warps,),
    )


def launch_mrg_poisson(engine, out, threshold, aliases, first, size, block_size, num_warps, *, capture=None):
    state = engine.ensure_state(out.device)
    plan = _prepared(out.device, out.numel(), engine.nstreams, first, size, block_size, num_warps)
    cursor = engine._cursor
    compiled = plan.launch(state, out, threshold, aliases, cursor)
    if capture is not None:
        capture(plan, compiled, (state, out, threshold, aliases, plan.n, cursor), 1, 5)
    return compiled
