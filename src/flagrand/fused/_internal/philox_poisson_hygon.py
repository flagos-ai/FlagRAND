"""Reuse Hygon's compiled Philox Poisson specialization for repeated calls."""

from functools import lru_cache

import triton

from flagrand.runtime.compiled_launcher import CachedKernelLauncher


@lru_cache(maxsize=1)
def _launcher():
    from flagrand.fused.poisson import _philox_poisson

    kernel = triton.jit(_philox_poisson.fn, do_not_specialize=["counter"])
    return CachedKernelLauncher(
        kernel,
        constexpr_names=("SHIFT", "FIRST", "SIZE", "BLOCK"),
        option_names=("num_warps",),
    )


def launch_philox_poisson(engine, out, threshold, aliases, first, size, block_size, num_warps, *, capture=None):
    n = out.numel()
    launcher = _launcher()
    grid = (triton.cdiv(n, 2 * block_size),)
    args = (out, threshold, aliases, engine.seed, engine.offset // 4, n)
    constants = (engine.offset % 4, first, size, block_size)
    options = (num_warps,)
    compiled = launcher.launch(grid, args, constants, options)
    if capture is not None:
        capture(launcher, grid, args, constants, options, compiled)
    return compiled
