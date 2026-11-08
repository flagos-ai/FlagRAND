"""CoreX MRG handle-local reuse of a compiled public-call configuration."""

import os
from dataclasses import dataclass, field

import torch
from triton.runtime.driver import driver

from flagrand.runtime.backend import supports_cached_binary_launch


@dataclass
class CompiledCall:
    compiled: object
    kernel: object
    run: object
    function: object
    metadata: object
    grid: tuple
    device_index: int
    stream_reader: object
    parameters: tuple
    debug: object
    portable_value: object


@dataclass
class HandlePlan:
    n: int
    dtype: object
    device: object
    distribution: int
    nstreams: int
    mean: float
    stddev: float
    state: object
    calls: list = field(default_factory=lambda: [None, None])

    def matches(self, engine, n, dtype, device, distribution, mean, stddev):
        return (
            self.n == n
            and self.dtype == dtype
            and self.device == device
            and self.distribution == distribution
            and self.nstreams == engine.nstreams
            and self.mean is mean
            and self.stddev is stddev
            and self.state is engine._state
            and engine.seed == engine._state_seed
            and engine.offset == engine._state_offset
        )


def capture(engine, out, distribution, mean, stddev, backend, prepared, compiled, has_spare):
    # This is the actual result of this handle's preceding prepared.launch.
    # No ordering assumption about a module-global cache is involved.
    if backend != "iluvatar" or compiled is None or not supports_cached_binary_launch():
        engine._corex_recent_plan = None
        return
    dtype = out.dtype
    if type(mean) is not float or type(stddev) is not float:
        engine._corex_recent_plan = None
        return
    if dtype != (torch.int32 if distribution == 0 else torch.float32):
        engine._corex_recent_plan = None
        return
    kernel = prepared.launcher._kernel
    if (
        kernel.debug
        or getattr(compiled, "launch_enter_hook", None) is not None
        or getattr(compiled, "launch_exit_hook", None) is not None
    ):
        engine._corex_recent_plan = None
        return
    n, device = prepared.n, out.device
    plan = engine._corex_recent_plan
    if plan is None or not plan.matches(engine, n, dtype, device, distribution, mean, stddev):
        plan = HandlePlan(n, dtype, device, distribution, engine.nstreams, mean, stddev, engine._state)
        engine._corex_recent_plan = plan
    compiled._init_handles()
    grid = prepared.grid
    plan.calls[int(has_spare)] = CompiledCall(
        compiled,
        kernel,
        compiled.run.launch,
        compiled.function,
        compiled.packed_metadata,
        (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1),
        driver.active.get_current_device(),
        driver.active.get_current_stream,
        prepared.parameters,
        kernel.debug,
        os.environ.get("FLAGRAND_FORCE_PORTABLE"),
    )


_direct_getattribute = object.__getattribute__


def generate_hot(engine, out, distribution, *, seed=None, offset=None, mean=0.0, stddev=1.0, **kwargs):
    plan = engine._corex_recent_plan
    if seed is not None or offset is not None or kwargs:
        engine._corex_recent_plan = None
        return engine._generate_slow(out, distribution, seed=seed, offset=offset, mean=mean, stddev=stddev, **kwargs)
    n, dtype, device = out.numel(), out.dtype, out.device
    if plan is not None and plan.matches(engine, n, dtype, device, distribution, mean, stddev):
        paired = distribution >= 2
        has_spare = engine._normal_has_spare.get(False, False) if paired else False
        entry = plan.calls[int(has_spare)]
        if entry is not None:
            compiled = entry.compiled
            if (
                os.environ.get("FLAGRAND_FORCE_PORTABLE") == entry.portable_value
                and entry.kernel.debug == entry.debug
                and _direct_getattribute(compiled, "launch_enter_hook") is None
                and _direct_getattribute(compiled, "launch_exit_hook") is None
            ):
                state = engine._state
                if paired:
                    spare_index = engine._normal_spare_index.get(False, 0)
                    spare = engine._normal_cache[False]
                    arguments = (state, out, spare, n, engine._cursor, spare_index) + entry.parameters
                else:
                    arguments = (state, out, n, engine._cursor) + entry.parameters
                # Tensor pointers are resolved by the existing C launcher now;
                # the stream is read now, including after a stream switch.
                stream = entry.stream_reader(entry.device_index)
                entry.run(*entry.grid, stream, entry.function, entry.metadata, None, None, None, *arguments)
                if paired:
                    pairs = (n - int(has_spare) + 1) // 2
                    has_new_spare = (n - int(has_spare)) % 2 != 0
                    engine._normal_has_spare[False] = has_new_spare
                    if has_new_spare:
                        engine._normal_spare_index[False] = 1 - spare_index
                    engine.offset += pairs * 2
                    engine._state_offset = engine.offset
                    engine._cursor = (engine._cursor + pairs) % engine.nstreams
                else:
                    engine.offset += n
                    engine._state_offset = engine.offset
                    engine._cursor = (engine._cursor + n) % engine.nstreams
                return out
            engine._corex_recent_plan = None
    return engine._generate_slow(out, distribution, mean=mean, stddev=stddev)
