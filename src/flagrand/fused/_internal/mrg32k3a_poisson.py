"""MRG32k3a raw-pair generation fused with the existing alias conversion.

The current stream cursor defines the first raw word. Adjacent raw words
stay adjacent when pairing, including a pair that crosses the stream wrap.
The caller advances the scalar cursor and consumed-word count.
"""

from dataclasses import dataclass
from functools import lru_cache

import triton
import triton.language as tl
from triton.runtime.driver import driver

from flagrand.fused.poisson import _sample_alias
from flagrand.rng.mrg32k3a import _mrg32k3a_step, _mrg32k3a_step_u32
from flagrand.runtime.backend import get_vendor_info
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key, launch_specialization_key


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
    UINT32_STATE: tl.constexpr = False,
):
    slot = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    stream = (CURSOR + slot) % NSTREAMS
    present = (slot < NSTREAMS) & (slot < 2 * N)
    x0 = tl.load(STATE + stream, present, other=1).to(tl.uint32)
    x1 = tl.load(STATE + NSTREAMS + stream, present, other=1).to(tl.uint32)
    x2 = tl.load(STATE + 2 * NSTREAMS + stream, present, other=1).to(tl.uint32)
    y0 = tl.load(STATE + 3 * NSTREAMS + stream, present, other=1).to(tl.uint32)
    y1 = tl.load(STATE + 4 * NSTREAMS + stream, present, other=1).to(tl.uint32)
    y2 = tl.load(STATE + 5 * NSTREAMS + stream, present, other=1).to(tl.uint32)
    if not UINT32_STATE:
        x0, x1, x2 = x0.to(tl.int64), x1.to(tl.int64), x2.to(tl.int64)
        y0, y1, y2 = y0.to(tl.int64), y1.to(tl.int64), y2.to(tl.int64)
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for batch in range(tl.cdiv(2 * N, NSTREAMS)):
        active = present & (batch * NSTREAMS + slot < 2 * N)
        if UINT32_STATE:
            word, a, b, c, d, e, f = _mrg32k3a_step_u32(x0, x1, x2, y0, y1, y2)
        else:
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


@lru_cache(None)
def _launcher():
    return CachedKernelLauncher(
        _mrg32k3a_poisson,
        constexpr_names=("NSTREAMS", "FIRST", "SIZE", "BLOCK", "UINT32_STATE"),
        option_names=("num_warps",),
    )


@dataclass
class _PreparedPoissonLaunch:
    launcher: CachedKernelLauncher
    grid: tuple
    n: int
    constants: tuple
    options: tuple
    _signature: object = None
    _key: object = None

    def launch(self, state, out, threshold, aliases, cursor):
        tensors = (state, out, threshold, aliases)
        arguments = tensors + (self.n, cursor)
        signatures = tuple((value.dtype, value.data_ptr() % 16 == 0) for value in tensors)
        debug = self.launcher._kernel.debug
        signature = (driver.active.get_current_device(), signatures, integer_specialization_key(cursor), debug)
        if signature != self._signature:
            self._key = launch_specialization_key(
                arguments, 4, self.constants, self.options, debug, tensor_signatures=signatures
            )
            self._signature = signature
        # Cursor and all four tensor pointers are read from this call. The
        # shared launcher retains dynamic mode checks and current-stream lookup.
        return self.launcher.launch(self.grid, arguments, self.constants, self.options, specialization_key=self._key)


@lru_cache(maxsize=128)
def _prepared_launch(device, n, nstreams, first, size, block_size, num_warps):
    grid = (triton.cdiv(min(n, nstreams // 2), block_size),)
    return _PreparedPoissonLaunch(_launcher(), grid, n, (nstreams, first, size, block_size, True), (num_warps,))


def launch_mrg32k3a_poisson(engine, out, threshold, aliases, first, size, block_size=128, num_warps=4):
    state = engine.ensure_state(out.device)
    n = out.numel()
    if get_vendor_info().vendor_name == "thead":
        prepared = _prepared_launch(out.device, n, engine.nstreams, first, size, block_size, num_warps)
        prepared.launch(state, out, threshold, aliases, engine._cursor)
        return out
    _mrg32k3a_poisson[(triton.cdiv(min(n, engine.nstreams // 2), block_size),)](
        state,
        out,
        threshold,
        aliases,
        n,
        engine._cursor,
        engine.nstreams,
        first,
        size,
        block_size,
        num_warps=num_warps,
    )
    return out
