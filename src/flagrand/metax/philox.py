from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from flagrand.metax.launcher import MetaXKernelLauncher
from flagrand.metax.transforms import (
    uint32_pair_to_uniform64,
    uint32_to_uniform,
    uniform_to_normal_f64,
    uniform_to_normal_portable_f32,
)
from flagrand.runtime import HAS_LIBTUNER, require_libtuner
from flagrand.runtime.libtuner import PHILOX_LIBTUNER_CONFIGS, libentry, libtuner


@triton.jit
def _philox_generate(seed, counter):
    c0 = tl.zeros_like(counter).to(tl.uint32)
    c1 = tl.zeros_like(counter).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _philox_window4(seed, counter, SHIFT: tl.constexpr):
    a0, a1, a2, a3 = _philox_generate(seed, counter)
    if SHIFT == 0:
        return a0, a1, a2, a3

    b0, b1, b2, b3 = _philox_generate(seed, counter + 1)
    if SHIFT == 1:
        return a1, a2, a3, b0
    if SHIFT == 2:
        return a2, a3, b0, b1
    return a3, b0, b1, b2


@triton.jit
def _raw_kernel(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter < n_counters
    r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)

    r02 = tl.join(r0, r2)
    r13 = tl.join(r1, r3)
    tile = tl.reshape(tl.join(r02, r13), (BLOCK, 4))
    offsets = (counter * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out + offsets, tile.to(tl.int32, bitcast=True), mask=mask[:, None])


@triton.jit(do_not_specialize=["base_counter"])
def _raw_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    base = group * 4
    tl.store(out + base + 0, r0.to(tl.int32, bitcast=True), mask=mask & (base + 0 < n))
    tl.store(out + base + 1, r1.to(tl.int32, bitcast=True), mask=mask & (base + 1 < n))
    tl.store(out + base + 2, r2.to(tl.int32, bitcast=True), mask=mask & (base + 2 < n))
    tl.store(out + base + 3, r3.to(tl.int32, bitcast=True), mask=mask & (base + 3 < n))


@triton.jit
def _uniform_f32_kernel(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter < n_counters
    r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)

    u02 = tl.join(u0, u2)
    u13 = tl.join(u1, u3)
    tile = tl.reshape(tl.join(u02, u13), (BLOCK, 4))
    offsets = (counter * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out + offsets, tile, mask=mask[:, None])


@triton.jit(do_not_specialize=["base_counter"])
def _uniform_f32_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    base = group * 4
    tl.store(out + base + 0, uint32_to_uniform(r0), mask=mask & (base + 0 < n))
    tl.store(out + base + 1, uint32_to_uniform(r1), mask=mask & (base + 1 < n))
    tl.store(out + base + 2, uint32_to_uniform(r2), mask=mask & (base + 2 < n))
    tl.store(out + base + 3, uint32_to_uniform(r3), mask=mask & (base + 3 < n))


@triton.jit
def _normal_f32_kernel(
    out,
    seed,
    base_counter,
    n,
    n_counters,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter < n_counters
    base = counter * 4
    r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
    z0, z1 = uniform_to_normal_portable_f32(uint32_to_uniform(r0), uint32_to_uniform(r1))
    z2, z3 = uniform_to_normal_portable_f32(uint32_to_uniform(r2), uint32_to_uniform(r3))

    if STANDARD:
        y0 = z0
        y1 = z1
        y2 = z2
        y3 = z3
    else:
        y0 = mean + stddev * z0
        y1 = mean + stddev * z1
        y2 = mean + stddev * z2
        y3 = mean + stddev * z3
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
        y2 = tl.exp(y2)
        y3 = tl.exp(y3)

    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _normal_f32_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    z0, z1 = uniform_to_normal_portable_f32(uint32_to_uniform(r0), uint32_to_uniform(r1))
    z2, z3 = uniform_to_normal_portable_f32(uint32_to_uniform(r2), uint32_to_uniform(r3))

    if STANDARD:
        y0 = z0
        y1 = z1
        y2 = z2
        y3 = z3
    else:
        y0 = mean + stddev * z0
        y1 = mean + stddev * z1
        y2 = mean + stddev * z2
        y3 = mean + stddev * z3
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
        y2 = tl.exp(y2)
        y3 = tl.exp(y3)

    base = group * 4
    tl.store(out + base + 0, y0, mask=mask & (base + 0 < n))
    tl.store(out + base + 1, y1, mask=mask & (base + 1 < n))
    tl.store(out + base + 2, y2, mask=mask & (base + 2 < n))
    tl.store(out + base + 3, y3, mask=mask & (base + 3 < n))


@triton.jit
def _uniform_f64_kernel(out, seed, base_counter, n, n_counters, BLOCK: tl.constexpr):
    counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter < n_counters
    base = counter * 2
    r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
    u0 = uint32_pair_to_uniform64(r0, r1)
    u1 = uint32_pair_to_uniform64(r2, r3)
    tile = tl.join(u0, u1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _uniform_f64_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    base = group * 2
    tl.store(out + base + 0, uint32_pair_to_uniform64(r0, r1), mask=mask & (base + 0 < n))
    tl.store(out + base + 1, uint32_pair_to_uniform64(r2, r3), mask=mask & (base + 1 < n))


@triton.jit
def _normal_f64_kernel(
    out,
    seed,
    base_counter,
    n,
    n_counters,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter < n_counters
    base = counter * 2
    r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
    z0, z1 = uniform_to_normal_f64(
        uint32_pair_to_uniform64(r0, r1),
        uint32_pair_to_uniform64(r2, r3),
    )

    if STANDARD:
        y0 = z0
        y1 = z1
    else:
        y0 = mean + stddev * z0
        y1 = mean + stddev * z1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)

    tile = tl.join(y0, y1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _normal_f64_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    z0, z1 = uniform_to_normal_f64(
        uint32_pair_to_uniform64(r0, r1),
        uint32_pair_to_uniform64(r2, r3),
    )

    if STANDARD:
        y0 = z0
        y1 = z1
    else:
        y0 = mean + stddev * z0
        y1 = mean + stddev * z1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)

    base = group * 2
    tl.store(out + base + 0, y0, mask=mask & (base + 0 < n))
    tl.store(out + base + 1, y1, mask=mask & (base + 1 < n))


if HAS_LIBTUNER:

    @libentry()
    @libtuner(
        configs=PHILOX_LIBTUNER_CONFIGS,
        key=["n_counters"],
        strategy=["log"],
        warmup=10,
        rep=30,
    )
    @triton.jit
    def _raw_kernel_tuned(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
        counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = counter < n_counters
        r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
        r02 = tl.join(r0, r2)
        r13 = tl.join(r1, r3)
        tile = tl.reshape(tl.join(r02, r13), (BLOCK, 4))
        offsets = (counter * 4)[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out + offsets, tile.to(tl.int32, bitcast=True), mask=mask[:, None])

    @libentry()
    @libtuner(
        configs=PHILOX_LIBTUNER_CONFIGS,
        key=["n_counters"],
        strategy=["log"],
        warmup=10,
        rep=30,
    )
    @triton.jit
    def _uniform_f32_kernel_tuned(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
        counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = counter < n_counters
        r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
        u0 = uint32_to_uniform(r0)
        u1 = uint32_to_uniform(r1)
        u2 = uint32_to_uniform(r2)
        u3 = uint32_to_uniform(r3)
        u02 = tl.join(u0, u2)
        u13 = tl.join(u1, u3)
        tile = tl.reshape(tl.join(u02, u13), (BLOCK, 4))
        offsets = (counter * 4)[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out + offsets, tile, mask=mask[:, None])

    @libentry()
    @libtuner(
        configs=PHILOX_LIBTUNER_CONFIGS,
        key=["n_counters"],
        strategy=["log"],
        warmup=10,
        rep=30,
    )
    @triton.jit
    def _normal_f32_kernel_tuned(
        out,
        seed,
        base_counter,
        n,
        n_counters,
        mean,
        stddev,
        STANDARD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = counter < n_counters
        base = counter * 4
        r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
        z0, z1 = uniform_to_normal_portable_f32(uint32_to_uniform(r0), uint32_to_uniform(r1))
        z2, z3 = uniform_to_normal_portable_f32(uint32_to_uniform(r2), uint32_to_uniform(r3))
        if STANDARD:
            y0 = z0
            y1 = z1
            y2 = z2
            y3 = z3
        else:
            y0 = mean + stddev * z0
            y1 = mean + stddev * z1
            y2 = mean + stddev * z2
            y3 = mean + stddev * z3
        y02 = tl.join(y0, y2)
        y13 = tl.join(y1, y3)
        tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
        offsets = base[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out + offsets, tile, mask=mask[:, None] & (offsets < n))

    @libentry()
    @libtuner(
        configs=PHILOX_LIBTUNER_CONFIGS,
        key=["n_counters"],
        strategy=["log"],
        warmup=10,
        rep=30,
    )
    @triton.jit
    def _normal_f64_kernel_tuned(
        out,
        seed,
        base_counter,
        n,
        n_counters,
        mean,
        stddev,
        STANDARD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        counter = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = counter < n_counters
        base = counter * 2
        r0, r1, r2, r3 = _philox_generate(seed, base_counter + counter)
        z0, z1 = uniform_to_normal_f64(
            uint32_pair_to_uniform64(r0, r1),
            uint32_pair_to_uniform64(r2, r3),
        )
        if STANDARD:
            y0 = z0
            y1 = z1
        else:
            y0 = mean + stddev * z0
            y1 = mean + stddev * z1
        tile = tl.join(y0, y1)
        offsets = base[:, None] + tl.arange(0, 2)[None, :]
        tl.store(out + offsets, tile, mask=mask[:, None] & (offsets < n))

else:
    _raw_kernel_tuned = None
    _uniform_f32_kernel_tuned = None
    _normal_f32_kernel_tuned = None
    _normal_f64_kernel_tuned = None


_RAW_LAUNCHER = MetaXKernelLauncher(_raw_kernel, constexpr_names=("BLOCK",))
_RAW_GENERIC_LAUNCHER = MetaXKernelLauncher(
    _raw_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_UNIFORM_F32_LAUNCHER = MetaXKernelLauncher(
    _uniform_f32_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_F32_GENERIC_LAUNCHER = MetaXKernelLauncher(
    _uniform_f32_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_NORMAL_F32_LAUNCHER = MetaXKernelLauncher(
    _normal_f32_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "BLOCK"),
)
_NORMAL_F32_GENERIC_LAUNCHER = MetaXKernelLauncher(
    _normal_f32_generic_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "SHIFT", "BLOCK"),
)
_UNIFORM_F64_LAUNCHER = MetaXKernelLauncher(
    _uniform_f64_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_F64_GENERIC_LAUNCHER = MetaXKernelLauncher(
    _uniform_f64_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_NORMAL_F64_LAUNCHER = MetaXKernelLauncher(
    _normal_f64_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "BLOCK"),
)
_NORMAL_F64_GENERIC_LAUNCHER = MetaXKernelLauncher(
    _normal_f64_generic_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "SHIFT", "BLOCK"),
)


def generate_metax_raw_u32(
    out: torch.Tensor,
    generator: Any,
    *,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool = False,
) -> torch.Tensor:
    seed, offset, n = _launch_state(out, generator, "MetaX Philox raw")
    if n:
        base_counter = offset // 4
        if offset % 4 == 0 and n % 4 == 0:
            n_counters = n // 4
            if use_libtuner_kernel:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _raw_kernel_tuned[grid](out, seed, base_counter, n_counters)
            else:
                _RAW_LAUNCHER.launch(
                    (n_counters + block_size - 1) // block_size,
                    (out, seed, base_counter, n_counters),
                    (block_size,),
                    (num_warps,),
                )
        else:
            n_groups = (n + 3) // 4
            _RAW_GENERIC_LAUNCHER.launch(
                (n_groups + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_groups),
                (offset % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset + n
    return out


def generate_metax_uniform_f32(
    out: torch.Tensor,
    generator: Any,
    *,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool = False,
) -> torch.Tensor:
    seed, offset, n = _launch_state(out, generator, "MetaX Philox uniform_f32")
    if n:
        base_counter = offset // 4
        if offset % 4 == 0 and n % 4 == 0:
            n_counters = n // 4
            if use_libtuner_kernel:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _uniform_f32_kernel_tuned[grid](out, seed, base_counter, n_counters)
            else:
                _UNIFORM_F32_LAUNCHER.launch(
                    (n_counters + block_size - 1) // block_size,
                    (out, seed, base_counter, n_counters),
                    (block_size,),
                    (num_warps,),
                )
        else:
            n_groups = (n + 3) // 4
            _UNIFORM_F32_GENERIC_LAUNCHER.launch(
                (n_groups + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_groups),
                (offset % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset + n
    return out


def generate_metax_normal_f32(
    out: torch.Tensor,
    generator: Any,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    lognormal: bool,
    use_libtuner_kernel: bool = False,
) -> torch.Tensor:
    seed, offset, n = _launch_state(
        out,
        generator,
        "MetaX Philox lognormal_f32" if lognormal else "MetaX Philox normal_f32",
        require_even=True,
    )
    if n:
        base_counter = offset // 4
        standard = mean == 0.0 and stddev == 1.0
        if offset % 4 == 0 and n % 4 == 0:
            n_counters = n // 4
            if use_libtuner_kernel and not lognormal:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _normal_f32_kernel_tuned[grid](
                    out,
                    seed,
                    base_counter,
                    n,
                    n_counters,
                    mean,
                    stddev,
                    STANDARD=standard,
                )
            else:
                _NORMAL_F32_LAUNCHER.launch(
                    (n_counters + block_size - 1) // block_size,
                    (out, seed, base_counter, n, n_counters, mean, stddev),
                    (lognormal, standard, block_size),
                    (num_warps,),
                )
        else:
            n_groups = (n + 3) // 4
            _NORMAL_F32_GENERIC_LAUNCHER.launch(
                (n_groups + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_groups, mean, stddev),
                (lognormal, standard, offset % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset + n
    return out


def generate_metax_uniform_f64(
    out: torch.Tensor,
    generator: Any,
    *,
    block_size: int,
    num_warps: int,
) -> torch.Tensor:
    seed, offset, n = _launch_state(out, generator, "MetaX Philox uniform_f64")
    if n:
        base_counter = offset // 4
        if offset % 4 == 0 and n % 2 == 0:
            n_counters = n // 2
            _UNIFORM_F64_LAUNCHER.launch(
                (n_counters + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_counters),
                (block_size,),
                (num_warps,),
            )
        else:
            n_groups = (n + 1) // 2
            _UNIFORM_F64_GENERIC_LAUNCHER.launch(
                (n_groups + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_groups),
                (offset % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset + 2 * n
    return out


def generate_metax_normal_f64(
    out: torch.Tensor,
    generator: Any,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    lognormal: bool,
    use_libtuner_kernel: bool = False,
) -> torch.Tensor:
    seed, offset, n = _launch_state(
        out,
        generator,
        "MetaX Philox lognormal_f64" if lognormal else "MetaX Philox normal_f64",
        require_even=True,
    )
    if n:
        base_counter = offset // 4
        n_counters = n // 2
        standard = mean == 0.0 and stddev == 1.0
        if offset % 4 == 0:
            if use_libtuner_kernel and not lognormal:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _normal_f64_kernel_tuned[grid](
                    out,
                    seed,
                    base_counter,
                    n,
                    n_counters,
                    mean,
                    stddev,
                    STANDARD=standard,
                )
            else:
                _NORMAL_F64_LAUNCHER.launch(
                    (n_counters + block_size - 1) // block_size,
                    (out, seed, base_counter, n, n_counters, mean, stddev),
                    (lognormal, standard, block_size),
                    (num_warps,),
                )
        else:
            _NORMAL_F64_GENERIC_LAUNCHER.launch(
                (n_counters + block_size - 1) // block_size,
                (out, seed, base_counter, n, n_counters, mean, stddev),
                (lognormal, standard, offset % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset + 2 * n
    return out


def _launch_state(
    out: torch.Tensor,
    generator: Any,
    op_name: str,
    *,
    require_even: bool = False,
) -> tuple[int, int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if require_even and n % 2:
        raise ValueError(f"{op_name}: requires an even element count, got {n}.")
    offset = int(generator.offset)
    if offset < 0:
        raise ValueError(f"{op_name}: offset must be non-negative, got {offset}.")
    return int(generator.seed), offset, n


__all__ = [
    "generate_metax_normal_f32",
    "generate_metax_normal_f64",
    "generate_metax_raw_u32",
    "generate_metax_uniform_f32",
    "generate_metax_uniform_f64",
]
