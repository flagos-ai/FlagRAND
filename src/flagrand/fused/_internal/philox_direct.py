from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.runtime import CachedKernelLauncher, uses_portable_f32_math
from flagrand.fused._internal.transforms import (
    uint32_to_uniform,
    uniform_to_normal_fast_f32,
    uniform_to_normal_portable_f32,
)


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _philox_uniform_f32_kernel(out_ptr, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)

    u01 = tl.join(u0, u1)
    u23 = tl.join(u2, u3)
    tile = tl.reshape(tl.join(u01, u23), (BLOCK, 4))
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + base, tile, mask=mask[:, None])


@triton.jit
def _philox_normal_f32_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_counters,
    mean,
    stddev,
    BLOCK: tl.constexpr,
    PORTABLE_MATH: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 4

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)
    if PORTABLE_MATH:
        n0, n1 = uniform_to_normal_portable_f32(u0, u1)
        n2, n3 = uniform_to_normal_portable_f32(u2, u3)
    else:
        n0, n1 = uniform_to_normal_fast_f32(u0, u1)
        n2, n3 = uniform_to_normal_fast_f32(u2, u3)

    y0 = mean + stddev * n0
    y1 = mean + stddev * n1
    y2 = mean + stddev * n2
    y3 = mean + stddev * n3

    y01 = tl.join(y0, y1)
    y23 = tl.join(y2, y3)
    tile = tl.reshape(tl.join(y01, y23), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_lognormal_f32_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_counters,
    mean,
    stddev,
    BLOCK: tl.constexpr,
    PORTABLE_MATH: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 4

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)
    if PORTABLE_MATH:
        n0, n1 = uniform_to_normal_portable_f32(u0, u1)
        n2, n3 = uniform_to_normal_portable_f32(u2, u3)
    else:
        n0, n1 = uniform_to_normal_fast_f32(u0, u1)
        n2, n3 = uniform_to_normal_fast_f32(u2, u3)

    y0 = tl.exp(mean + stddev * n0)
    y1 = tl.exp(mean + stddev * n1)
    y2 = tl.exp(mean + stddev * n2)
    y3 = tl.exp(mean + stddev * n3)

    y01 = tl.join(y0, y1)
    y23 = tl.join(y2, y3)
    tile = tl.reshape(tl.join(y01, y23), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


_PHILOX_UNIFORM_F32_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f32_kernel,
    constexpr_names=("BLOCK",),
)
_PHILOX_NORMAL_F32_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f32_kernel,
    constexpr_names=("BLOCK", "PORTABLE_MATH"),
)
_PHILOX_LOGNORMAL_F32_LAUNCHER = CachedKernelLauncher(
    _philox_lognormal_f32_kernel,
    constexpr_names=("BLOCK", "PORTABLE_MATH"),
)
def generate_philox_uniform_f32(
    out: torch.Tensor,
    generator,
    *,
    block_size: int,
    num_warps: int,
    op_name: str = "generate_uniform",
) -> None:
    seed_val, offset_val, n_counters = _philox_launch_args(out, generator, op_name)
    grid = triton.cdiv(n_counters, block_size)
    _PHILOX_UNIFORM_F32_LAUNCHER.launch(
        grid,
        (out, seed_val, offset_val // 4, n_counters),
        (block_size,),
        (num_warps,),
    )
    generator.offset = offset_val + out.numel()


def generate_philox_normal_f32(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    _portable_math: bool | None = None,
) -> None:
    seed_val, offset_val, n_counters = _philox_launch_args(out, generator, "generate_normal")
    grid = triton.cdiv(n_counters, block_size)
    _PHILOX_NORMAL_F32_LAUNCHER.launch(
        grid,
        (out, seed_val, offset_val // 4, out.numel(), n_counters, mean, stddev),
        (
            block_size,
            uses_portable_f32_math() if _portable_math is None else _portable_math,
        ),
        (num_warps,),
    )
    generator.offset = offset_val + out.numel()


def generate_philox_lognormal_f32(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    _portable_math: bool | None = None,
) -> None:
    seed_val, offset_val, n_counters = _philox_launch_args(out, generator, "generate_lognormal")
    grid = triton.cdiv(n_counters, block_size)
    _PHILOX_LOGNORMAL_F32_LAUNCHER.launch(
        grid,
        (out, seed_val, offset_val // 4, out.numel(), n_counters, mean, stddev),
        (
            block_size,
            uses_portable_f32_math() if _portable_math is None else _portable_math,
        ),
        (num_warps,),
    )
    generator.offset = offset_val + out.numel()


def _philox_launch_args(out: torch.Tensor, generator, op_name: str) -> tuple[int, int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if n % 4 != 0:
        raise ValueError(
            f"{op_name}: Philox requires element count to be a multiple of 4, got {n}."
        )
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: Philox offset must be >= 0, got {offset_val}.")
    if offset_val % 4 != 0:
        raise ValueError(
            f"{op_name}: Philox offset is measured in uint32 outputs and must be "
            f"a multiple of 4, got {offset_val}."
        )
    return int(getattr(generator, "seed", 0)), offset_val, n // 4
