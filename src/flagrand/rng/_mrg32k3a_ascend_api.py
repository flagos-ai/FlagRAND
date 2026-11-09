"""Ascend MRG launches with contiguous state tiles and rounded normal spares."""

import math
import struct
from dataclasses import dataclass
from functools import lru_cache

import torch
import triton

from flagrand.runtime.ascend_launcher import PreparedAscendLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._mrg32k3a_ascend_active import tile_count
from ._mrg32k3a_ascend_lookahead import BLOCK as LOOKAHEAD_BLOCK
from ._mrg32k3a_ascend_lookahead import MAX_OUTPUTS as LOOKAHEAD_MAX_OUTPUTS
from ._mrg32k3a_ascend_lookahead import MAX_STREAMS as LOOKAHEAD_MAX_STREAMS
from ._mrg32k3a_ascend_lookahead import PREFETCH_CALLS, Normal64Window, lookahead_launcher
from ._mrg32k3a_ascend_plan import NormalPlan, RawPlan

_JOINT_MIN_COUNT = 131072


@dataclass
class _Plan:
    config: tuple
    state: object
    launcher: object
    block: int
    output_block: int


@lru_cache(maxsize=2)
def _launcher(f64):
    entry_factory = None
    if f64:
        from ._mrg32k3a_ascend_soft64 import uniform64_kernel

        kernel = uniform64_kernel()
        names = ("S", "BLOCK", "CURSOR_ALIGNED")
    else:
        from ._mrg32k3a_ascend_active import generate32_kernel
        from ._mrg32k3a_ascend_launcher import create_entry_factory
        from ._mrg32k3a_ascend_stream import create_driver_view

        entry_factory = create_entry_factory()
        kernel = generate32_kernel
        names = (
            "S",
            "BLOCK",
            "DIST",
            "HAS_SPARE",
            "CURSOR_ALIGNED",
            "EMIT_SPARE",
            "FULL_ROUNDS",
            "OUTPUT_BLOCK",
            "JOINT",
        )
    launcher = PreparedAscendLauncher(
        kernel,
        constexpr_names=names,
        option_names=("num_warps", "enable_fp_fusion", "sync_solver"),
        entry_factory=entry_factory,
    )
    if not f64:
        launcher.driver = create_driver_view(launcher.driver)
    return launcher


def generate(engine, out, distribution, *, mean=0.0, stddev=1.0, **kwargs):
    paired = distribution >= 2
    if (
        kwargs
        or distribution not in (0, 1, 2, 3)
        or out.dtype == torch.float64
        or (not paired and (mean != 0.0 or stddev != 1.0))
        or (
            paired
            and (
                out.dtype != torch.float32
                or type(mean) is not float
                or type(stddev) is not float
                or mean != 0.0
                or stddev != 1.0
                or math.copysign(1.0, mean) < 0.0
            )
        )
    ):
        return _generate_without_plan_cache(engine, out, distribution, mean=mean, stddev=stddev, **kwargs)
    n = out.numel()
    if not n:
        return out
    config = (n, out.dtype, out.device, distribution, engine.nstreams)
    plan = engine._ascend_paired_cached_plan if paired else engine._ascend_raw_cached_plan
    if (
        plan is None
        or plan.config != config
        or plan.state is not engine._state
        or engine.seed != engine._state_seed
        or engine.offset != engine._state_offset
    ):
        plan_type = NormalPlan if paired else RawPlan
        plan = plan_type(engine, out, distribution, config, _launcher(False), _JOINT_MIN_COUNT)
        if paired:
            engine._ascend_paired_cached_plan = plan
        else:
            engine._ascend_raw_cached_plan = plan
    return plan.call(engine, out)


def _generate_without_plan_cache(engine, out, distribution, *, mean=0.0, stddev=1.0, **kwargs):
    n, f64 = out.numel(), out.dtype == torch.float64
    if not n:
        return out
    if f64 and distribution >= 2:
        if mean == 0.0 and stddev == 1.0:
            return _generate_normal64(engine, out, distribution, **kwargs)
        from ._mrg32k3a_ascend import _generate_fallback

        return _generate_fallback(engine, out, distribution, mean=mean, stddev=stddev, **kwargs)
    default_block = 256 if f64 or (distribution >= 2 and n <= 1024) else 1024
    block = kwargs.get("block_size", kwargs.get("block", default_block))
    config = (n, out.dtype, out.device, distribution, engine.nstreams, struct.pack("<dd", mean, stddev), block)
    plan = getattr(engine, "_ascend_plan", None)
    if (
        plan is None
        or plan.config != config
        or plan.state is not engine._state
        or engine.seed != engine._state_seed
        or engine.offset != engine._state_offset
    ):
        state = engine.ensure_state(out.device)
        output_block = 0 if f64 else triton.next_power_of_2(min(n, (2 if distribution >= 2 else 1) * block))
        plan = _Plan(config, state, _launcher(f64), block, output_block)
        engine._ascend_plan = plan
    paired = distribution >= 2
    has_spare, spare_index, spare = False, 0, plan.state
    if paired:
        if False not in engine._normal_cache:
            engine._normal_cache[False] = torch.empty(2, dtype=out.dtype, device=out.device)
        spare = engine._normal_cache[False]
        has_spare = engine._normal_has_spare.get(False, False)
        spare_index = engine._normal_spare_index.get(False, 0)
    count = (n - int(has_spare) + 1) // 2 if paired else n
    cursor = engine._cursor
    aligned = cursor % block == 0
    emit_spare = paired and (n - int(has_spare)) % 2 != 0
    grid = (tile_count(engine.nstreams, cursor, count, block),)
    target = out.view(torch.int32) if f64 else out
    if f64:
        args = (plan.state, target, n, cursor)
        constants = (engine.nstreams, block, aligned)
    else:
        args = (plan.state, target, spare, n, cursor, spare_index, float(mean), float(stddev))
        full_rounds = count >= engine.nstreams and count % engine.nstreams == 0
        joint = count >= max(engine.nstreams, _JOINT_MIN_COUNT)
        constants = (
            engine.nstreams,
            block,
            distribution,
            has_spare,
            aligned,
            emit_spare,
            full_rounds,
            plan.output_block,
            joint,
        )
    options = (kwargs.get("num_warps", 4), False, False)
    if kwargs.get("cached_launch", True):
        key = (config, target.data_ptr() % 16, integer_specialization_key(cursor), spare_index, constants)
        plan.launcher.launch(grid, args, constants, options, specialization_key=key)
    else:
        plan.launcher._regular(grid, args, constants, options)
    if paired:
        engine._normal_has_spare[False] = emit_spare
        if emit_spare:
            engine._normal_spare_index[False] = 1 - spare_index
    engine.offset += count * (2 if paired or f64 else 1)
    engine._state_offset = engine.offset
    engine._cursor = (cursor + count) % engine.nstreams
    return out


def _value_key(value):
    if isinstance(value, torch.Tensor):
        return value.device, value.dtype, value.data_ptr() % 16
    if isinstance(value, int):
        return integer_specialization_key(value)
    return value


class _StageLaunch:
    def __init__(self, kernel, runtime_count, names, n_index):
        self.runtime_count = runtime_count
        self.n_index = n_index
        self.launcher = PreparedAscendLauncher(
            kernel, constexpr_names=names, option_names=("num_warps", "enable_fp_fusion", "sync_solver")
        )

    def __call__(self, grid, values, options, cached):
        args, constants = values[: self.runtime_count], values[self.runtime_count :]
        if cached:
            key = (args[self.n_index], tuple(_value_key(v) for v in args), constants)
            return self.launcher.launch(grid, args, constants, options, specialization_key=key)
        return self.launcher._regular(grid, args, constants, options)


@lru_cache(maxsize=1)
def _normal64_launchers():
    from ._mrg32k3a_ascend_soft64 import normal64_kernels

    raw, radius, angle, output, exp = normal64_kernels()
    return (
        _StageLaunch(raw, 4, ("S", "B", "CURSOR_ALIGNED"), 2),
        _StageLaunch(radius, 3, ("B",), 2),
        _StageLaunch(angle, 3, ("B",), 2),
        _StageLaunch(output, 7, ("B", "HAS_SPARE", "EMIT_SPARE"), 4),
        _StageLaunch(exp, 3, ("B",), 2),
    )


def _generate_normal64(engine, out, distribution, **kwargs):
    n = out.numel()
    eligible = (
        0 < n <= LOOKAHEAD_MAX_OUTPUTS
        and n % 2 == 0
        and not kwargs
        and engine.nstreams % 16 == 0
        and engine.nstreams <= LOOKAHEAD_MAX_STREAMS
        and n // 2 <= engine.nstreams
        and not engine._normal_has_spare.get(True, False)
    )
    if not eligible:
        engine._ascend_normal64_window = None
        return _generate_normal64_staged(engine, out, distribution, **kwargs)
    launcher = lookahead_launcher()
    stream = launcher.driver.active.get_current_stream(out.device.index)
    window = engine._ascend_normal64_window
    if window is None or not window.matches(engine, out, distribution, stream):
        engine._ascend_normal64_window = None
        window = None
        window = Normal64Window(
            engine,
            out,
            distribution,
            PREFETCH_CALLS,
            _generate_normal64_staged,
            launcher,
            tile_count,
            LOOKAHEAD_BLOCK,
            stream,
        )
        engine._ascend_normal64_window = window
    return window.commit(engine, out)


def _generate_normal64_staged(engine, out, distribution, **kwargs):
    from ._mrg32k3a_ascend_soft64 import Scratch

    n = out.numel()
    state = engine.ensure_state(out.device)
    if True not in engine._normal_cache:
        engine._normal_cache[True] = torch.empty(2, dtype=torch.float64, device=out.device)
    spare = engine._normal_cache[True].view(torch.int32)
    has_spare = engine._normal_has_spare.get(True, False)
    spare_index = engine._normal_spare_index.get(True, 0)
    pairs = (n - int(has_spare) + 1) // 2
    emit_spare = (n - int(has_spare)) % 2 != 0
    key = (n, pairs, distribution == 3, out.device)
    plan = getattr(engine, "_ascend_staged64", None)
    if plan is None or plan.key != key:
        plan = Scratch(*key)
        engine._ascend_staged64 = plan
    plan.prepare_stream()
    raw, radius, angle, output, exp = _normal64_launchers()
    raw_block = 256
    block = kwargs.get("block_size", kwargs.get("block", 128))
    output_block, exp_block = 256, 128
    options = (kwargs.get("num_warps", 4), False, False)
    cached = kwargs.get("cached_launch", True)
    if pairs:
        raw(
            (tile_count(engine.nstreams, engine._cursor, pairs, raw_block),),
            (state, plan.raw, pairs, engine._cursor, engine.nstreams, raw_block, engine._cursor % raw_block == 0),
            options,
            cached,
        )
        radius((triton.cdiv(pairs, block),), (plan.raw, plan.radius, pairs, block), options, cached)
        angle((triton.cdiv(pairs, block),), (plan.raw, plan.angle, pairs, block), options, cached)
    standard = plan.standard if distribution == 3 else out.view(torch.int32)
    output(
        (triton.cdiv(max(1, pairs), output_block),),
        (plan.radius, plan.angle, standard, spare, n, pairs, spare_index, output_block, has_spare, emit_spare),
        options,
        cached,
    )
    if distribution == 3:
        exp((triton.cdiv(n, exp_block),), (standard, out.view(torch.int32), n, exp_block), options, cached)
    engine._normal_has_spare[True] = emit_spare
    if emit_spare:
        engine._normal_spare_index[True] = 1 - spare_index
    engine.offset += 4 * pairs
    engine._state_offset = engine.offset
    engine._cursor = (engine._cursor + pairs) % engine.nstreams
    return out


@lru_cache(maxsize=1)
def _poisson_launchers():
    from ._mrg32k3a_ascend_active import generate32_kernel
    from ._mrg32k3a_ascend_poisson import alias_kernel, even_pairs_kernel

    return (
        _StageLaunch(even_pairs_kernel, 6, ("S", "FIRST", "SIZE", "BLOCK", "CURSOR_ODD", "CURSOR_ALIGNED"), 4),
        _StageLaunch(
            generate32_kernel, 8, ("S", "BLOCK", "DIST", "HAS_SPARE", "CURSOR_ALIGNED", "EMIT_SPARE", "FULL_ROUNDS"), 3
        ),
        _StageLaunch(alias_kernel, 5, ("FIRST", "SIZE", "BLOCK"), 4),
    )


def launch_poisson(engine, out, threshold, aliases, first, size, block_size=512, num_warps=4):
    state = engine.ensure_state(out.device)
    n, streams, cursor = out.numel(), engine.nstreams, engine._cursor
    block, raw_block = block_size, 512
    options = (num_warps, False, False)
    fused, raw, convert = _poisson_launchers()
    if streams % 2:
        words = torch.empty(2 * n, dtype=torch.int32, device=out.device)
        raw(
            (tile_count(streams, cursor, 2 * n, raw_block),),
            (
                state,
                words,
                state,
                2 * n,
                cursor,
                0,
                0.0,
                1.0,
                streams,
                raw_block,
                0,
                False,
                cursor % raw_block == 0,
                False,
                2 * n >= streams and (2 * n) % streams == 0,
            ),
            options,
            True,
        )
        convert((triton.cdiv(n, block),), (words, out, threshold, aliases, n, first, size, block), options, True)
    else:
        pair_cursor = cursor // 2
        fused(
            (tile_count(streams // 2, pair_cursor, n, block),),
            (
                state,
                out,
                threshold,
                aliases,
                n,
                cursor,
                streams,
                first,
                size,
                block,
                cursor % 2 != 0,
                pair_cursor % block == 0,
            ),
            options,
            True,
        )
    return out
