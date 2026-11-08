"""Hygon MRG32k3a host plans using the existing recurrence and transforms."""

from dataclasses import dataclass, field

import torch

from flagrand.runtime.specialization import integer_specialization_key

from ._hygon_launch import prepare_hygon


@dataclass
class Entry:
    bound: object
    parameters: tuple
    constants: tuple
    options: tuple
    key: object = field(default_factory=object)


@dataclass
class Plan:
    config: tuple
    state: object
    mean: object
    stddev: object
    entries: dict = field(default_factory=dict)


def generate(engine, out, distribution, *, seed=None, offset=None, mean=0.0, stddev=1.0, **kwargs):
    if seed is not None or offset is not None or kwargs:
        return engine._generate_slow(out, distribution, seed=seed, offset=offset, mean=mean, stddev=stddev, **kwargs)
    n, device, dtype = out.numel(), out.device, out.dtype
    if n == 0:
        return out
    config = (n, dtype, device, distribution, engine.nstreams)
    plan = getattr(engine, "_hygon_plan", None)
    state = engine._state
    if not (
        plan is not None
        and plan.config == config
        and plan.state is state
        and type(mean) is float
        and type(stddev) is float
        and plan.mean is mean
        and plan.stddev is stddev
        and engine.seed == engine._state_seed
        and engine.offset == engine._state_offset
    ):
        state = engine.ensure_state(device)
        if engine._resolve_backend(device, "auto", True) != "hygon":
            return engine._generate_slow(out, distribution, mean=mean, stddev=stddev)
        # Identity also distinguishes +0.0 from -0.0 without computing bits
        # on each call. Equal newly constructed parameters build a new plan.
        plan = Plan(config, state, mean, stddev)
        engine._hygon_plan = plan
    f64 = dtype == torch.float64
    paired = distribution >= 2
    has_spare = False
    spare_index = 0
    spare = None
    if paired:
        if f64 not in engine._normal_cache:
            engine._normal_cache[f64] = torch.empty(2, dtype=dtype, device=device)
        spare = engine._normal_cache[f64]
        has_spare = engine._normal_has_spare.get(f64, False)
        spare_index = engine._normal_spare_index.get(f64, 0)
        pairs = (n - int(has_spare) + 1) // 2
    # Match HCU's pointer alignment and storage-range specialization.
    output_signature = (out.data_ptr() % 16 == 0, out.untyped_storage().size() <= (1 << 31) - 1)
    signature = (
        has_spare,
        integer_specialization_key(engine._cursor),
        spare_index,
        output_signature,
    )
    entry = plan.entries.get(signature)
    if entry is None:
        from .mrg32k3a import _launch_spec, _launcher

        grid, parameters, constants, options = _launch_spec(
            "hygon", n, engine.nstreams, distribution, f64, has_spare, mean, stddev, {}
        )
        entry = Entry(prepare_hygon(_launcher("hygon", paired), grid), parameters, constants, options)
        plan.entries[signature] = entry
    if paired:
        arguments = (state, out, spare, n, engine._cursor, spare_index) + entry.parameters
    else:
        arguments = (state, out, n, engine._cursor) + entry.parameters
    entry.bound.launch(arguments, entry.constants, entry.options, specialization_key=entry.key)
    if paired:
        has_new_spare = (n - int(has_spare)) % 2 != 0
        engine._normal_has_spare[f64] = has_new_spare
        if has_new_spare:
            engine._normal_spare_index[f64] = 1 - spare_index
        engine.offset += pairs * (4 if f64 else 2)
        engine._cursor = (engine._cursor + pairs) % engine.nstreams
    else:
        engine.offset += n * (2 if f64 and distribution else 1)
        engine._cursor = (engine._cursor + n) % engine.nstreams
    engine._state_offset = engine.offset
    return out
