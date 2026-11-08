"""CoreX Poisson fixed-configuration plans from actual cold public calls."""

import os
from dataclasses import dataclass, field

import torch
from triton.runtime.driver import driver
from triton.runtime.jit import compute_spec_key, mangle_type

from flagrand import _curand_handle as curand
from flagrand.fused.poisson import generate_poisson as _base_poisson
from flagrand.runtime.backend import supports_cached_binary_launch

_direct_getattribute = object.__getattribute__

KINDS = {
    curand.CURAND_RNG_PSEUDO_PHILOX4_32_10: "philox",
    curand.CURAND_RNG_PSEUDO_XORWOW: "xorwow",
    curand.CURAND_RNG_PSEUDO_MRG32K3A: "mrg32k3a",
}


def _original(handle, out, lam, **kwargs):
    return _base_poisson(handle.engine, out, lam, **kwargs)


def _normalize_grid(grid):
    return (grid, 1, 1) if isinstance(grid, int) else tuple(grid) + (1,) * (3 - len(grid))


@dataclass
class Entry:
    compiled: object
    kernel: object
    launch: object
    function: int
    metadata: tuple
    grid: tuple
    device: int
    stream_reader: object
    debug: object
    args: tuple
    out_index: int
    moving_index: int | None
    out_specialized: bool
    moving_specialized: bool

    def active(self):
        return (
            self.kernel.debug == self.debug
            and (not self.kernel.pre_run_hooks)
            and (self.kernel.launch_metadata is None)
            and (self.compiled.launch_enter_hook is None)
            and (self.compiled.launch_exit_hook is None)
        )

    def key(self, out, moving, shift):
        out_key = compute_spec_key(out) if self.out_specialized else None
        if self.moving_index is None:
            return out_key
        return (shift, out_key, mangle_type(moving), compute_spec_key(moving) if self.moving_specialized else None)

    def run(self, values):
        stream = self.stream_reader(self.device)
        self.launch(*self.grid, stream, self.function, self.metadata, None, None, None, *values)


@dataclass
class Context:
    engine: object
    kind: str
    config: tuple
    expected_offset: int
    portable_value: object
    state: object = None
    entries: dict = field(default_factory=dict)
    exemplar: object = None

    def continuation(self, engine):
        if engine is not self.engine or engine.offset != self.expected_offset:
            return False
        if self.kind == "philox":
            return True
        return (
            engine._state is self.state
            and engine._state_seed == engine.seed
            and (engine._state_offset == engine.offset)
        )

    def moving(self, engine):
        if self.kind == "philox":
            return (engine.offset // 4, engine.offset % 4)
        if self.kind == "mrg32k3a":
            return (engine._cursor, None)
        return (None, None)

    def values(self, entry, engine, out, moving):
        if self.kind == "philox":
            _, threshold, aliases, seed, _, n = entry.args
            return (out, threshold, aliases, seed, moving, n)
        if self.kind == "mrg32k3a":
            _, _, threshold, aliases, n, _ = entry.args
            return (engine._state, out, threshold, aliases, n, moving)
        _, _, threshold, aliases, n = entry.args
        return (engine._state, out, threshold, aliases, n)

    def advance(self, engine, n):
        engine.offset += 2 * n
        if self.kind != "philox":
            engine._state_offset = engine.offset
            if self.kind == "mrg32k3a":
                engine._cursor = (engine._cursor + 2 * n) % engine.nstreams
        self.expected_offset = engine.offset


def _record(prepared, compiled, values, out_index, moving_index):
    compiled._init_handles()
    kernel = prepared.launcher._kernel
    params = kernel.params
    out_specialized = not params[out_index].do_not_specialize
    moving_specialized = moving_index is not None and (not params[moving_index].do_not_specialize)
    return Entry(
        compiled,
        kernel,
        compiled.run.launch,
        compiled.function,
        tuple(compiled.packed_metadata),
        _normalize_grid(prepared.grid),
        driver.active.get_current_device(),
        driver.active.get_current_stream,
        kernel.debug,
        tuple((None if i == out_index else value for i, value in enumerate(values))),
        out_index,
        moving_index,
        out_specialized,
        moving_specialized,
    )


def _configuration(engine, out, lam, n):
    return (n, lam, engine.seed, out.device, out.dtype, getattr(engine, "nstreams", None))


def generate_poisson(handle, out, lam, **kwargs):
    kind = KINDS.get(handle.rng_type)
    if kwargs or kind is None:
        handle._iluvatar_poisson_context = None
        return _original(handle, out, lam, **kwargs)
    engine = handle.engine
    n = out.numel()
    if n == 0:
        return _original(handle, out, lam)
    lam = float(lam)
    config = (n, lam, engine.seed, out.device, out.dtype, getattr(engine, "nstreams", None))
    context = getattr(handle, "_iluvatar_poisson_context", None)
    portable_value = os.environ.get("FLAGRAND_FORCE_PORTABLE")
    moving = engine.offset // 4 if kind == "philox" else engine._cursor if kind == "mrg32k3a" else None
    shift = engine.offset % 4 if kind == "philox" else None
    if (
        context is not None
        and context.portable_value == portable_value
        and context.config == config
        and engine is context.engine
        and engine.offset == context.expected_offset
        and (
            kind == "philox"
            or (
                engine._state is context.state
                and engine._state_seed == engine.seed
                and engine._state_offset == engine.offset
            )
        )
    ):
        exemplar = context.exemplar
        out_key = compute_spec_key(out) if exemplar.out_specialized else None
        key = (
            out_key
            if exemplar.moving_index is None
            else (
                shift,
                out_key,
                mangle_type(moving),
                compute_spec_key(moving) if exemplar.moving_specialized else None,
            )
        )
        entry = context.entries.get(key)
        if (
            entry is not None
            and entry.kernel.debug == entry.debug
            and not entry.kernel.pre_run_hooks
            and entry.kernel.launch_metadata is None
            and _direct_getattribute(entry.compiled, "launch_enter_hook") is None
            and _direct_getattribute(entry.compiled, "launch_exit_hook") is None
            and driver.active.get_current_device() == entry.device
        ):
            if kind == "philox":
                _, threshold, aliases, seed, _, saved_n = entry.args
                values = (out, threshold, aliases, seed, moving, saved_n)
            elif kind == "mrg32k3a":
                _, _, threshold, aliases, saved_n, _ = entry.args
                values = (engine._state, out, threshold, aliases, saved_n, moving)
            else:
                _, _, threshold, aliases, saved_n = entry.args
                values = (engine._state, out, threshold, aliases, saved_n)
            stream = entry.stream_reader(entry.device)
            entry.launch(*entry.grid, stream, entry.function, entry.metadata, None, None, None, *values)
            engine.offset += 2 * n
            if kind != "philox":
                engine._state_offset = engine.offset
                if kind == "mrg32k3a":
                    engine._cursor = (engine._cursor + 2 * n) % engine.nstreams
            context.expected_offset = engine.offset
            return out
        if not exemplar.active():
            handle._iluvatar_poisson_context = None
            return _original(handle, out, lam)
    else:
        context = Context(engine, kind, config, engine.offset, portable_value)
    if not supports_cached_binary_launch():
        handle._iluvatar_poisson_context = None
        return _original(handle, out, lam)
    records = []

    def capture(prepared, compiled, values, out_index, moving_index):
        records.append(_record(prepared, compiled, values, out_index, moving_index))

    result = _base_poisson(engine, out, lam, _capture=capture)
    if records:
        entry = records[0]
        context.state = None if kind == "philox" else entry.args[0]
        context.exemplar = entry
        observed_moving = None if entry.moving_index is None else entry.args[entry.moving_index]
        key = entry.key(out, observed_moving, shift)
        context.entries[key] = entry
        context.expected_offset = engine.offset
        handle._iluvatar_poisson_context = context
    else:
        handle._iluvatar_poisson_context = None
    return result
