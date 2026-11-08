"""CoreX prepared launch for the fused Philox Poisson kernel."""

from dataclasses import dataclass, field
from functools import lru_cache

import triton
from triton.runtime.jit import compute_spec_key, mangle_type

from flagrand.runtime.compiled_launcher import CachedKernelLauncher


@dataclass
class _PreparedPoissonLaunch:
    launcher: CachedKernelLauncher
    grid: tuple
    seed: int
    n: int
    constants: tuple
    options: tuple
    context: object = field(default_factory=object)

    def launch(self, out, threshold, aliases, counter):
        tensors = (out, threshold, aliases)
        args = tensors + (self.seed, counter, self.n)
        # The fixed plan owns all scalars except counter. CoreX 4.4's actual
        # binder helpers retain the counter ABI and divisibility specialization.
        signature = (
            out.device,
            tuple((tensor.dtype, tensor.data_ptr()) for tensor in tensors),
            mangle_type(counter),
            compute_spec_key(counter),
            self.launcher._kernel.debug,
        )
        return self.launcher.launch(
            self.grid, args, self.constants, self.options, specialization_key=(self.context, signature)
        )


@lru_cache(maxsize=1)
def _corex_kernel():
    from flagrand.fused.poisson import _philox_poisson

    # Counter divisibility does not change the algorithm. Keep its runtime ABI
    # while avoiding new binaries as a persistent handle advances its counter.
    return triton.jit(_philox_poisson.fn, do_not_specialize=["counter"])


@lru_cache(maxsize=128)
def _prepared(device, seed, n, shift, first, size, block_size, num_warps):
    launcher = CachedKernelLauncher(
        _corex_kernel(),
        constexpr_names=("SHIFT", "FIRST", "SIZE", "BLOCK"),
        option_names=("num_warps",),
    )
    return _PreparedPoissonLaunch(
        launcher, (triton.cdiv(n, 2 * block_size),), seed, n, (shift, first, size, block_size), (num_warps,)
    )


def launch_philox_poisson(engine, out, threshold, aliases, first, size, block_size, num_warps, *, capture=None):
    prepared = _prepared(out.device, engine.seed, out.numel(), engine.offset % 4, first, size, block_size, num_warps)
    counter = engine.offset // 4
    compiled = prepared.launch(out, threshold, aliases, counter)
    if capture is not None:
        capture(prepared, compiled, (out, threshold, aliases, prepared.seed, counter, prepared.n), 0, 4)
    return compiled
