from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.runtime import CachedKernelLauncher, HAS_LIBTUNER, require_libtuner
from flagrand.runtime.libtuner import PHILOX_LIBTUNER_CONFIGS, libentry, libtuner
from flagrand.fused._internal.transforms import (
    uint32_pair_to_uniform64_curand_compat,
    uniform_to_normal_fast_f64,
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
def _philox_window4(seed, counter, SHIFT: tl.constexpr):
    a0, a1, a2, a3 = _philox_generate(seed, counter)
    b0, b1, b2, b3 = _philox_generate(seed, counter + 1)
    if SHIFT == 0:
        w0, w1, w2, w3 = a0, a1, a2, a3
    elif SHIFT == 1:
        w0, w1, w2, w3 = a1, a2, a3, b0
    elif SHIFT == 2:
        w0, w1, w2, w3 = a2, a3, b0, b1
    else:
        w0, w1, w2, w3 = a3, b0, b1, b2
    return w0, w1, w2, w3


@triton.jit
def _philox_uniform_f64_kernel(out_ptr, seed, base_counter, n, n_counters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 2

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)

    tile = tl.join(u0, u1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_uniform_f64_generic_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid * BLOCK + tl.arange(0, BLOCK)
    group_mask = group < n_groups

    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)

    base = group * 2
    tl.store(out_ptr + base + 0, u0, mask=group_mask & ((base + 0) < n))
    tl.store(out_ptr + base + 1, u1, mask=group_mask & ((base + 1) < n))


@triton.jit
def _philox_normal_f64_kernel(
    out_ptr,
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
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters
    base = offs * 2

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)
    n0, n1 = uniform_to_normal_fast_f64(u0, u1)

    if STANDARD:
        y0 = n0
        y1 = n1
    else:
        y0 = mean + stddev * n0
        y1 = mean + stddev * n1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)

    tile = tl.join(y0, y1)
    offsets = base[:, None] + tl.arange(0, 2)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_normal_f64_generic_kernel(
    out_ptr,
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
    pid = tl.program_id(0)
    group = pid * BLOCK + tl.arange(0, BLOCK)
    group_mask = group < n_groups

    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)
    n0, n1 = uniform_to_normal_fast_f64(u0, u1)

    if STANDARD:
        y0 = n0
        y1 = n1
    else:
        y0 = mean + stddev * n0
        y1 = mean + stddev * n1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)

    base = group * 2
    tl.store(out_ptr + base + 0, y0, mask=group_mask & ((base + 0) < n))
    tl.store(out_ptr + base + 1, y1, mask=group_mask & ((base + 1) < n))


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
    def _philox_normal_f64_kernel_tuned(
        out_ptr,
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
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_counters
        base = offs * 2

        r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
        u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
        u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)
        n0, n1 = uniform_to_normal_fast_f64(u0, u1)

        if STANDARD:
            y0 = n0
            y1 = n1
        else:
            y0 = mean + stddev * n0
            y1 = mean + stddev * n1
        if LOGNORMAL:
            y0 = tl.exp(y0)
            y1 = tl.exp(y1)

        tile = tl.join(y0, y1)
        offsets = base[:, None] + tl.arange(0, 2)[None, :]
        tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))

else:
    _philox_normal_f64_kernel_tuned = None


_PHILOX_UNIFORM_F64_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f64_kernel,
    constexpr_names=("BLOCK",),
)
_PHILOX_UNIFORM_F64_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f64_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_PHILOX_NORMAL_F64_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f64_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "BLOCK"),
)
_PHILOX_NORMAL_F64_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f64_generic_kernel,
    constexpr_names=("LOGNORMAL", "STANDARD", "SHIFT", "BLOCK"),
)


def generate_philox_uniform_f64(
    out: torch.Tensor,
    generator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    seed_val, offset_val = _philox_launch_args_f64(out, generator, "generate_uniform")
    n = out.numel()
    if n:
        base_counter = offset_val // 4
        if offset_val % 4 == 0 and n % 2 == 0:
            n_counters = n // 2
            grid = triton.cdiv(n_counters, block_size)
            _PHILOX_UNIFORM_F64_LAUNCHER.launch(
                grid,
                (out, seed_val, base_counter, n, n_counters),
                (block_size,),
                (num_warps,),
            )
        else:
            n_groups = triton.cdiv(n, 2)
            grid = triton.cdiv(n_groups, block_size)
            _PHILOX_UNIFORM_F64_GENERIC_LAUNCHER.launch(
                grid,
                (out, seed_val, base_counter, n, n_groups),
                (offset_val % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset_val + 2 * n


def generate_philox_normal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool = False,
) -> None:
    _generate_philox_normal_like_f64(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        use_libtuner_kernel=use_libtuner_kernel,
        lognormal=False,
        op_name="generate_normal",
    )


def generate_philox_lognormal_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool = False,
) -> None:
    _generate_philox_normal_like_f64(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        use_libtuner_kernel=use_libtuner_kernel,
        lognormal=True,
        op_name="generate_lognormal",
    )


def _generate_philox_normal_like_f64(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool,
    lognormal: bool,
    op_name: str,
) -> None:
    seed_val, offset_val = _philox_launch_args_f64(
        out,
        generator,
        op_name,
        require_even=True,
    )
    n = out.numel()
    if n:
        base_counter = offset_val // 4
        standard = mean == 0.0 and stddev == 1.0
        if offset_val % 4 == 0:
            n_counters = n // 2
            if use_libtuner_kernel and not lognormal:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _philox_normal_f64_kernel_tuned[grid](
                    out,
                    seed_val,
                    base_counter,
                    n,
                    n_counters,
                    mean,
                    stddev,
                    LOGNORMAL=lognormal,
                    STANDARD=standard,
                )
            else:
                grid = triton.cdiv(n_counters, block_size)
                _PHILOX_NORMAL_F64_LAUNCHER.launch(
                    grid,
                    (out, seed_val, base_counter, n, n_counters, mean, stddev),
                    (lognormal, standard, block_size),
                    (num_warps,),
                )
        else:
            n_groups = n // 2
            grid = triton.cdiv(n_groups, block_size)
            _PHILOX_NORMAL_F64_GENERIC_LAUNCHER.launch(
                grid,
                (out, seed_val, base_counter, n, n_groups, mean, stddev),
                (lognormal, standard, offset_val % 4, block_size),
                (num_warps,),
            )
    generator.offset = offset_val + 2 * n


def _philox_launch_args_f64(
    out: torch.Tensor,
    generator,
    op_name: str,
    *,
    require_even: bool = False,
) -> tuple[int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if require_even and n % 2 != 0:
        raise ValueError(f"{op_name}: Philox float64 path requires an even element count, got {n}.")
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: Philox offset must be >= 0, got {offset_val}.")
    return int(getattr(generator, "seed", 0)), offset_val
