"""Hygon Poisson plans retain tables and the actual compiled specialization."""

from dataclasses import dataclass, field

from flagrand import _curand_handle as curand
from flagrand.fused.poisson import generate_poisson as _base_poisson
from flagrand.runtime.specialization import integer_specialization_key


@dataclass
class Entry:
    bound: object
    args: tuple
    constants: tuple
    options: tuple
    key: object


@dataclass
class Context:
    engine: object
    config: tuple
    offset: int
    state: object
    entries: dict = field(default_factory=dict)


def _output_signature(out):
    # HCU adds the S tensor attribute when the entire storage fits in 31 bits.
    return (out.data_ptr() % 16 == 0, out.untyped_storage().size() <= (1 << 31) - 1)


def generate_poisson(handle, out, lam, **kwargs):
    philox = handle.rng_type == curand.CURAND_RNG_PSEUDO_PHILOX4_32_10
    xorwow = handle.rng_type == curand.CURAND_RNG_PSEUDO_XORWOW
    if kwargs or not (philox or xorwow):
        return _base_poisson(handle.engine, out, lam, **kwargs)
    engine = handle.engine
    n = out.numel()
    if n == 0 or (xorwow and engine.nstreams % 2):
        return _base_poisson(engine, out, lam)
    config = (n, float(lam), engine.seed, out.device, out.dtype, getattr(engine, "nstreams", None))
    context = getattr(handle, "_hygon_poisson_context", None)
    if (
        context is None
        or context.engine is not engine
        or context.config != config
        or context.offset != engine.offset
        or (xorwow and context.state is not engine._state)
    ):
        context = Context(engine, config, engine.offset, getattr(engine, "_state", None))
        handle._hygon_poisson_context = context
    counter = engine.offset // 4 if philox else None
    key = (_output_signature(out), integer_specialization_key(counter), engine.offset % 4) if philox else _output_signature(out)
    entry = context.entries.get(key)
    if entry is None:
        def capture(launcher, grid, args, constants, options, compiled):
            context.entries[key] = Entry(launcher.prepare(grid), args, constants, options, object())

        _base_poisson(engine, out, lam, _capture=capture)
        context.state = getattr(engine, "_state", None)
        context.offset = engine.offset
        return out
    if philox:
        _, threshold, aliases, seed, _, saved_n = entry.args
        values = (out, threshold, aliases, seed, counter, saved_n)
    else:
        _, _, threshold, aliases, saved_n = entry.args
        values = (engine._state, out, threshold, aliases, saved_n)
    entry.bound.launch(values, entry.constants, entry.options, specialization_key=entry.key)
    engine.offset += 2 * n
    if xorwow:
        engine._state_offset = engine.offset
    context.offset = engine.offset
    return out
