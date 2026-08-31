from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.transforms import uint32_pair_to_uniform64_curand_compat, uniform_to_normal_fast_f64
from flagrand.runtime import CachedKernelLauncher
from flagrand.runtime.specialization import make_specialization_key


@triton.jit
def _philox_at(seed, subsequence, COUNTER_OFFSET: tl.constexpr):
    subsequence = subsequence.to(tl.uint64)
    c0 = (tl.zeros_like(subsequence) + COUNTER_OFFSET).to(tl.uint32)
    c1 = (tl.zeros_like(subsequence)).to(tl.uint32)
    c2 = subsequence.to(tl.uint32)
    c3 = (subsequence >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _uniform4(seed, subsequence):
    a0, a1, a2, a3 = _philox_at(seed, subsequence, 0)
    b0, b1, b2, b3 = _philox_at(seed, subsequence, 1)
    return (
        uint32_pair_to_uniform64_curand_compat(a0, a1),
        uint32_pair_to_uniform64_curand_compat(a2, a3),
        uint32_pair_to_uniform64_curand_compat(b0, b1),
        uint32_pair_to_uniform64_curand_compat(b2, b3),
    )


@triton.jit
def _window4(a0, a1, a2, a3, b0, b1, b2, SHIFT: tl.constexpr):
    if SHIFT == 0:
        return a0, a1, a2, a3
    if SHIFT == 1:
        return a1, a2, a3, b0
    if SHIFT == 2:
        return a2, a3, b0, b1
    return a3, b0, b1, b2


@triton.jit
def _store4(out_ptr, base, n, mask, y0, y1, y2, y3, BLOCK: tl.constexpr):
    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _uniform_f64_kernel(out_ptr, seed, base_subsequence, n, n_groups, BLOCK: tl.constexpr):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    y0, y1, y2, y3 = _uniform4(seed, base_subsequence + group)
    _store4(out_ptr, group * 4, n, mask, y0, y1, y2, y3, BLOCK)


@triton.jit
def _uniform_f64_generic_kernel(
    out_ptr,
    seed,
    base_subsequence,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    a0, a1, a2, a3 = _uniform4(seed, base_subsequence + group)
    b0, b1, b2, _ = _uniform4(seed, base_subsequence + group + 1)
    y0, y1, y2, y3 = _window4(a0, a1, a2, a3, b0, b1, b2, SHIFT)
    _store4(out_ptr, group * 4, n, mask, y0, y1, y2, y3, BLOCK)


@triton.jit
def _normal4(seed, subsequence, mean, stddev, LOGNORMAL: tl.constexpr, STANDARD: tl.constexpr):
    u0, u1, u2, u3 = _uniform4(seed, subsequence)
    n0, n1 = uniform_to_normal_fast_f64(u0, u1)
    n2, n3 = uniform_to_normal_fast_f64(u2, u3)
    if STANDARD:
        y0, y1, y2, y3 = n0, n1, n2, n3
    else:
        y0 = mean + stddev * n0
        y1 = mean + stddev * n1
        y2 = mean + stddev * n2
        y3 = mean + stddev * n3
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
        y2 = tl.exp(y2)
        y3 = tl.exp(y3)
    return y0, y1, y2, y3


@triton.jit
def _normal_f64_kernel(
    out_ptr,
    seed,
    base_subsequence,
    n,
    n_groups,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    y0, y1, y2, y3 = _normal4(
        seed,
        base_subsequence + group,
        mean,
        stddev,
        LOGNORMAL,
        STANDARD,
    )
    _store4(out_ptr, group * 4, n, mask, y0, y1, y2, y3, BLOCK)


@triton.jit
def _normal_f64_generic_kernel(
    out_ptr,
    seed,
    base_subsequence,
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
    a0, a1, a2, a3 = _normal4(
        seed,
        base_subsequence + group,
        mean,
        stddev,
        LOGNORMAL,
        STANDARD,
    )
    b0, b1, b2, _ = _normal4(
        seed,
        base_subsequence + group + 1,
        mean,
        stddev,
        LOGNORMAL,
        STANDARD,
    )
    y0, y1, y2, y3 = _window4(a0, a1, a2, a3, b0, b1, b2, SHIFT)
    _store4(out_ptr, group * 4, n, mask, y0, y1, y2, y3, BLOCK)


_UNIFORM_F64_LAUNCHER = CachedKernelLauncher(
    _uniform_f64_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_F64_GENERIC_LAUNCHER = CachedKernelLauncher(
    _uniform_f64_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_NORMAL_F64_LAUNCHER = CachedKernelLauncher(
    _normal_f64_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "BLOCK"),
)
_NORMAL_F64_GENERIC_LAUNCHER = CachedKernelLauncher(
    _normal_f64_generic_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "SHIFT", "BLOCK"),
)


def generate_nvidia_philox_uniform_f64(
    out: torch.Tensor,
    generator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    seed, offset = _launch_args(out, generator, "generate_uniform")
    _launch_uniform(out, seed, offset, block_size, num_warps)
    generator.offset = offset + out.numel()


def generate_nvidia_philox_normal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_normal_like(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=False,
        op_name="generate_normal",
    )


def generate_nvidia_philox_lognormal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_normal_like(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=True,
        op_name="generate_lognormal",
    )


def _launch_uniform(out: torch.Tensor, seed: int, offset: int, block_size: int, num_warps: int) -> None:
    n = out.numel()
    if not n:
        return
    n_groups = triton.cdiv(n, 4)
    grid = triton.cdiv(n_groups, block_size)
    base_subsequence = offset // 4
    shift = offset % 4
    if shift == 0:
        arguments = (out, seed, base_subsequence, n, n_groups)
        constexpr_values = (block_size,)
        option_values = (num_warps,)
        _UNIFORM_F64_LAUNCHER.launch(
            grid,
            arguments,
            constexpr_values,
            option_values,
            specialization_key=make_specialization_key(arguments, constexpr_values, option_values),
        )
    else:
        arguments = (out, seed, base_subsequence, n, n_groups)
        constexpr_values = (shift, block_size)
        option_values = (num_warps,)
        _UNIFORM_F64_GENERIC_LAUNCHER.launch(
            grid,
            arguments,
            constexpr_values,
            option_values,
            specialization_key=make_specialization_key(arguments, constexpr_values, option_values),
        )


def _generate_normal_like(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    lognormal: bool,
    op_name: str,
) -> None:
    seed, offset = _launch_args(out, generator, op_name, require_even=True)
    n = out.numel()
    if n:
        n_groups = triton.cdiv(n, 4)
        grid = triton.cdiv(n_groups, block_size)
        base_subsequence = offset // 4
        shift = offset % 4
        standard = mean == 0.0 and stddev == 1.0
        if shift == 0:
            arguments = (out, seed, base_subsequence, n, n_groups, mean, stddev)
            constexpr_values = (lognormal, standard, block_size)
            option_values = (num_warps,)
            _NORMAL_F64_LAUNCHER.launch(
                grid,
                arguments,
                constexpr_values,
                option_values,
                specialization_key=make_specialization_key(arguments, constexpr_values, option_values),
            )
        else:
            arguments = (out, seed, base_subsequence, n, n_groups, mean, stddev)
            constexpr_values = (lognormal, standard, shift, block_size)
            option_values = (num_warps,)
            _NORMAL_F64_GENERIC_LAUNCHER.launch(
                grid,
                arguments,
                constexpr_values,
                option_values,
                specialization_key=make_specialization_key(arguments, constexpr_values, option_values),
            )
    generator.offset = offset + n


def _launch_args(
    out: torch.Tensor,
    generator,
    op_name: str,
    *,
    require_even: bool = False,
) -> tuple[int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if require_even and n % 2:
        raise ValueError(f"{op_name}: Philox float64 path requires an even element count, got {n}.")
    offset = int(generator.offset)
    if offset < 0:
        raise ValueError(f"{op_name}: Philox offset must be >= 0, got {offset}.")
    return int(generator.seed), offset
