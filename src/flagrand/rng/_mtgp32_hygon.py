"""Hygon MTGP32 plans for complete blocks and dynamic output tensors.

The existing recurrence/transform kernels keep the same 192-state sequence.
Plans retain allocations and compiled specializations, while pointers,
START_BLOCK and the current device stream remain launch-time arguments.
"""

from dataclasses import dataclass, field

import torch

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._hygon_launch import prepare_hygon
from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_sequence import ensure_working_state


@dataclass
class _Stage:
    bound: object
    constants: tuple
    options: tuple
    key: object = field(default_factory=object)


@dataclass
class _Plan:
    config: tuple
    state: object
    tensors: tuple
    words: int
    blocks: int
    mode: int
    parameters: tuple
    threshold: object
    aliases: object
    scratch: object
    launchers: tuple
    constants: tuple
    options: tuple
    grids: tuple
    entries: dict = field(default_factory=dict)


def _plan(engine, out, api, mean, stddev, lam, fused, warps, linear_table):
    config = (out.numel(), out.dtype, out.device, api, mean, stddev, lam, engine.seed, fused, warps, linear_table)
    plan = getattr(engine, "_hygon_mtgp_plan", None)
    if plan is not None and plan.config == config and plan.state is engine._working_state:
        return plan
    from flagrand.fused.poisson import _DEVICE_TABLES, _raw_poisson

    from ._mtgp32_fused import _fused_launcher
    from ._mtgp32_kernel import _contiguous_launcher
    from ._mtgp32_uniform_f32 import _uniform_f32_launcher
    from ._poisson_table import poisson_alias_table
    from .mtgp32 import _float_bits, _normal_launcher, _uniform_launcher

    n, f64 = out.numel(), out.dtype == torch.float64
    mode = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3, "poisson": 4}[api]
    words = (
        n
        if mode == 0
        else (2 * n if mode == 4 else ((n + 1) // 2 * (4 if f64 else 2) if mode >= 2 else n * (2 if f64 else 1)))
    )
    ensure_working_state(engine, engine.seed, str(out.device))
    state = engine._working_state
    tensors = (state,) + engine._ws_params
    blocks, states = words // 256, state.shape[0]
    base = (states, MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N, linear_table)
    actual_warps = min(warps, 2) if blocks <= 32 else warps
    grid = (min(blocks, states),)
    params = (_float_bits(mean, f64), _float_bits(stddev, f64))
    threshold = aliases = out
    first, size = 0, 1
    if mode == 4:
        table_key = (float(lam), 1 << 32, str(out.device))
        if table_key not in _DEVICE_TABLES:
            cutoff, alias, first = poisson_alias_table(float(lam), 1 << 32)
            signed = [x if x < 1 << 31 else x - (1 << 32) for x in cutoff]
            _DEVICE_TABLES[table_key] = (
                torch.tensor(signed, dtype=torch.int32, device=out.device),
                torch.tensor(alias, dtype=torch.int32, device=out.device),
                first,
            )
        threshold, aliases, first = _DEVICE_TABLES[table_key]
        size = threshold.numel()
    scratch = None
    if mode == 0 or (mode == 1 and not f64):
        launchers = (_contiguous_launcher if mode == 0 else _uniform_f32_launcher,)
        constants, options, grids = (base,), ((actual_warps,),), (grid,)
    elif fused:
        launchers = (_fused_launcher,)
        constants = ((states, mode, f64, first, size, 256, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N),)
        options, grids = ((actual_warps, mode not in (2, 3)),), (grid,)
    else:
        scratch = torch.empty(words, dtype=torch.int32, device=out.device)
        if mode == 1:
            second, constants2, options2, grid2 = _uniform_launcher, (f64, 256), (4,), ((n + 255) // 256,)
        elif mode == 4:
            second = CachedKernelLauncher(_raw_poisson, constexpr_names=("FIRST", "SIZE", "RAW_RANGE", "BLOCK"))
            constants2, options2, grid2 = (first, size, 1 << 32, 256), (4,), ((n + 255) // 256,)
        else:
            second = _normal_launcher(False)
            constants2, options2, grid2 = (f64, mode == 3, 256), (4, False), (((n + 1) // 2 + 255) // 256,)
        launchers = (_contiguous_launcher, second)
        constants, options, grids = (base, constants2), ((actual_warps,), options2), (grid, grid2)
    plan = _Plan(
        config,
        state,
        tensors,
        words,
        blocks,
        mode,
        params,
        threshold,
        aliases,
        scratch,
        launchers,
        constants,
        options,
        grids,
    )
    engine._hygon_mtgp_plan = plan
    return plan


def generate_aligned(
    engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0, fused=True, num_warps=4, linear_table=False
):
    plan = _plan(engine, out, api, mean, stddev, lam, fused, num_warps, linear_table)
    start = engine.offset // 256
    signature = (
        out.data_ptr() % 16 == 0,
        out.untyped_storage().size() <= (1 << 31) - 1,
        integer_specialization_key(start),
    )
    stages = plan.entries.get(signature)
    if stages is None:
        stages = tuple(
            _Stage(prepare_hygon(launcher, grid), constants, options)
            for launcher, grid, constants, options in zip(plan.launchers, plan.grids, plan.constants, plan.options)
        )
        plan.entries[signature] = stages
    if plan.scratch is not None:
        args = (plan.scratch,) + plan.tensors + (start, plan.blocks)
    elif plan.mode == 0 or (plan.mode == 1 and out.dtype == torch.float32):
        args = (out,) + plan.tensors + (start, plan.blocks)
    else:
        threshold = plan.threshold if plan.mode == 4 else out
        aliases = plan.aliases if plan.mode == 4 else out
        args = (out,) + plan.tensors + (start, plan.blocks, out.numel()) + plan.parameters + (threshold, aliases)
    stage = stages[0]
    stage.bound.launch(args, stage.constants, stage.options, specialization_key=stage.key)
    if plan.scratch is not None:
        if plan.mode == 1:
            args = (plan.scratch, out, out.numel())
        elif plan.mode == 4:
            args = (plan.scratch, out, plan.threshold, plan.aliases, out.numel())
        else:
            args = (plan.scratch, out, out.numel()) + plan.parameters
        stage = stages[1]
        stage.bound.launch(args, stage.constants, stage.options, specialization_key=stage.key)
    engine.offset += plan.words
    engine._ws_next_block_start += plan.words
    return out


def try_generate(engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0):
    n, f64 = out.numel(), out.dtype == torch.float64
    words = (
        n
        if api == "raw32"
        else (
            2 * n
            if api == "poisson"
            else ((n + 1) // 2) * (4 if f64 else 2) if api in ("normal", "lognormal") else n * (2 if f64 else 1)
        )
    )
    if engine.offset % 256 or words % 256 or getattr(engine, "_chunk_cache", None) is not None:
        return False
    generate_aligned(engine, out, api, mean=mean, stddev=stddev, lam=lam)
    return True
