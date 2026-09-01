from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.fused._internal.transforms import (
    uint32_to_uniform,
    uniform_to_normal_fast_f32,
    uniform_to_normal_portable_f32,
)
from flagrand.runtime import HAS_LIBTUNER, CachedKernelLauncher, require_libtuner, uses_portable_f32_math
from flagrand.runtime.libtuner import PHILOX_LIBTUNER_CONFIGS, libentry, libtuner
from flagrand.runtime.specialization import integer_specialization_key


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
def _philox_uniform_f32_kernel(out_ptr, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_counters

    r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)

    u02 = tl.join(u0, u2)
    u13 = tl.join(u1, u3)
    tile = tl.reshape(tl.join(u02, u13), (BLOCK, 4))
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + base, tile, mask=mask[:, None])


@triton.jit
def _philox_uniform_f32_generic_kernel(
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
    u0 = uint32_to_uniform(r0)
    u1 = uint32_to_uniform(r1)
    u2 = uint32_to_uniform(r2)
    u3 = uint32_to_uniform(r3)

    base = group * 4
    tl.store(out_ptr + base + 0, u0, mask=group_mask & (base + 0 < n))
    tl.store(out_ptr + base + 1, u1, mask=group_mask & (base + 1 < n))
    tl.store(out_ptr + base + 2, u2, mask=group_mask & (base + 2 < n))
    tl.store(out_ptr + base + 3, u3, mask=group_mask & (base + 3 < n))


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

    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_normal_f32_generic_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_groups,
    mean,
    stddev,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    PORTABLE_MATH: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid * BLOCK + tl.arange(0, BLOCK)
    group_mask = group < n_groups

    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
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

    base = group * 4
    tl.store(out_ptr + base + 0, y0, mask=group_mask & (base + 0 < n))
    tl.store(out_ptr + base + 1, y1, mask=group_mask & (base + 1 < n))
    tl.store(out_ptr + base + 2, y2, mask=group_mask & (base + 2 < n))
    tl.store(out_ptr + base + 3, y3, mask=group_mask & (base + 3 < n))


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

    y02 = tl.join(y0, y2)
    y13 = tl.join(y1, y3)
    tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
    offsets = base[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))


@triton.jit
def _philox_lognormal_f32_generic_kernel(
    out_ptr,
    seed,
    base_counter,
    n,
    n_groups,
    mean,
    stddev,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    PORTABLE_MATH: tl.constexpr,
):
    pid = tl.program_id(0)
    group = pid * BLOCK + tl.arange(0, BLOCK)
    group_mask = group < n_groups

    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group, SHIFT)
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

    base = group * 4
    tl.store(out_ptr + base + 0, y0, mask=group_mask & (base + 0 < n))
    tl.store(out_ptr + base + 1, y1, mask=group_mask & (base + 1 < n))
    tl.store(out_ptr + base + 2, y2, mask=group_mask & (base + 2 < n))
    tl.store(out_ptr + base + 3, y3, mask=group_mask & (base + 3 < n))


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
    def _philox_uniform_f32_kernel_tuned(
        out_ptr,
        seed,
        base_counter,
        n_counters,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_counters

        r0, r1, r2, r3 = _philox_generate(seed, base_counter + offs)
        u0 = uint32_to_uniform(r0)
        u1 = uint32_to_uniform(r1)
        u2 = uint32_to_uniform(r2)
        u3 = uint32_to_uniform(r3)

        u02 = tl.join(u0, u2)
        u13 = tl.join(u1, u3)
        tile = tl.reshape(tl.join(u02, u13), (BLOCK, 4))
        base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out_ptr + base, tile, mask=mask[:, None])

    @libentry()
    @libtuner(
        configs=PHILOX_LIBTUNER_CONFIGS,
        key=["n_counters"],
        strategy=["log"],
        warmup=10,
        rep=30,
    )
    @triton.jit
    def _philox_normal_f32_kernel_tuned(
        out_ptr,
        seed,
        base_counter,
        n,
        n_counters,
        mean,
        stddev,
        PORTABLE_MATH: tl.constexpr,
        BLOCK: tl.constexpr,
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

        y02 = tl.join(y0, y2)
        y13 = tl.join(y1, y3)
        tile = tl.reshape(tl.join(y02, y13), (BLOCK, 4))
        offsets = base[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out_ptr + offsets, tile, mask=mask[:, None] & (offsets < n))

else:
    _philox_uniform_f32_kernel_tuned = None
    _philox_normal_f32_kernel_tuned = None


_PHILOX_UNIFORM_F32_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f32_kernel,
    constexpr_names=("BLOCK",),
)
_PHILOX_UNIFORM_F32_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_uniform_f32_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_PHILOX_NORMAL_F32_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f32_kernel,
    constexpr_names=("BLOCK", "PORTABLE_MATH"),
)
_PHILOX_NORMAL_F32_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_normal_f32_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK", "PORTABLE_MATH"),
)
_PHILOX_LOGNORMAL_F32_LAUNCHER = CachedKernelLauncher(
    _philox_lognormal_f32_kernel,
    constexpr_names=("BLOCK", "PORTABLE_MATH"),
)
_PHILOX_LOGNORMAL_F32_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_lognormal_f32_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK", "PORTABLE_MATH"),
)


def generate_philox_uniform_f32(
    out: torch.Tensor,
    generator,
    *,
    block_size: int,
    num_warps: int,
    op_name: str = "generate_uniform",
    use_libtuner_kernel: bool = False,
) -> None:
    seed_val, offset_val, n = _philox_launch_args(out, generator, op_name)
    _launch_philox_uniform_f32(
        out,
        seed_val,
        offset_val,
        n,
        block_size,
        num_warps,
        use_libtuner_kernel,
    )
    generator.offset = offset_val + n


def _launch_philox_uniform_f32(
    out: torch.Tensor,
    seed_val: int,
    offset_val: int,
    n: int,
    block_size: int,
    num_warps: int,
    use_libtuner_kernel: bool = False,
) -> None:
    if n == 0:
        return

    base_counter = offset_val // 4
    if offset_val % 4 == 0 and n % 4 == 0:
        n_counters = n // 4
        if use_libtuner_kernel:
            require_libtuner()
            grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
            _philox_uniform_f32_kernel_tuned[grid](
                out,
                seed_val,
                base_counter,
                n_counters,
            )
        else:
            grid = (n_counters + block_size - 1) // block_size
            _PHILOX_UNIFORM_F32_LAUNCHER.launch(
                grid,
                (out, seed_val, base_counter, n_counters),
                (block_size,),
                (num_warps,),
                specialization_key=(
                    out.device,
                    out.dtype,
                    out.data_ptr() % 16 == 0,
                    seed_val,
                    integer_specialization_key(base_counter),
                    n_counters,
                    block_size,
                    num_warps,
                ),
            )
    else:
        shift = offset_val % 4
        n_groups = (n + 3) // 4
        grid = (n_groups + block_size - 1) // block_size
        arguments = (out, seed_val, base_counter, n, n_groups)
        constexpr_values = (shift, block_size)
        option_values = (num_warps,)
        _PHILOX_UNIFORM_F32_GENERIC_LAUNCHER.launch(
            grid,
            arguments,
            constexpr_values,
            option_values,
            specialization_key=(
                out.device,
                out.dtype,
                out.data_ptr() % 16 == 0,
                seed_val,
                integer_specialization_key(base_counter),
                n,
                n_groups,
                shift,
                block_size,
                num_warps,
            ),
        )


def generate_philox_normal_f32(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    _portable_math: bool | None = None,
    use_libtuner_kernel: bool = False,
) -> None:
    seed_val, offset_val, n = _philox_launch_args(
        out,
        generator,
        "generate_normal",
        require_even=True,
    )
    _launch_normal_like_f32(
        out,
        n,
        seed_val,
        offset_val,
        mean,
        stddev,
        block_size,
        num_warps,
        uses_portable_f32_math() if _portable_math is None else _portable_math,
        lognormal=False,
        use_libtuner_kernel=use_libtuner_kernel,
    )
    generator.offset = offset_val + n


def generate_philox_lognormal_f32(
    out: torch.Tensor,
    generator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    _portable_math: bool | None = None,
    use_libtuner_kernel: bool = False,
) -> None:
    seed_val, offset_val, n = _philox_launch_args(
        out,
        generator,
        "generate_lognormal",
        require_even=True,
    )
    _launch_normal_like_f32(
        out,
        n,
        seed_val,
        offset_val,
        mean,
        stddev,
        block_size,
        num_warps,
        uses_portable_f32_math() if _portable_math is None else _portable_math,
        lognormal=True,
        use_libtuner_kernel=use_libtuner_kernel,
    )
    generator.offset = offset_val + n


def _launch_normal_like_f32(
    out: torch.Tensor,
    n: int,
    seed_val: int,
    offset_val: int,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    portable_math: bool,
    *,
    lognormal: bool,
    use_libtuner_kernel: bool,
) -> None:
    if n == 0:
        return

    base_counter = offset_val // 4
    if offset_val % 4 == 0 and n % 4 == 0:
        n_counters = n // 4
        if use_libtuner_kernel and not lognormal:
            require_libtuner()
            grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
            _philox_normal_f32_kernel_tuned[grid](
                out,
                seed_val,
                base_counter,
                n,
                n_counters,
                mean,
                stddev,
                PORTABLE_MATH=portable_math,
            )
        else:
            grid = (n_counters + block_size - 1) // block_size
            launcher = _PHILOX_LOGNORMAL_F32_LAUNCHER if lognormal else _PHILOX_NORMAL_F32_LAUNCHER
            launcher.launch(
                grid,
                (out, seed_val, base_counter, n, n_counters, mean, stddev),
                (block_size, portable_math),
                (num_warps,),
                specialization_key=(
                    out.device,
                    out.dtype,
                    out.data_ptr() % 16 == 0,
                    seed_val,
                    integer_specialization_key(base_counter),
                    n,
                    n_counters,
                    mean,
                    stddev,
                    block_size,
                    portable_math,
                    num_warps,
                ),
            )
    else:
        shift = offset_val % 4
        n_groups = (n + 3) // 4
        grid = (n_groups + block_size - 1) // block_size
        launcher = _PHILOX_LOGNORMAL_F32_GENERIC_LAUNCHER if lognormal else _PHILOX_NORMAL_F32_GENERIC_LAUNCHER
        arguments = (out, seed_val, base_counter, n, n_groups, mean, stddev)
        constexpr_values = (shift, block_size, portable_math)
        option_values = (num_warps,)
        launcher.launch(
            grid,
            arguments,
            constexpr_values,
            option_values,
            specialization_key=(
                out.device,
                out.dtype,
                out.data_ptr() % 16 == 0,
                seed_val,
                integer_specialization_key(base_counter),
                n,
                n_groups,
                mean,
                stddev,
                shift,
                block_size,
                portable_math,
                num_warps,
            ),
        )


def _philox_launch_args(
    out: torch.Tensor,
    generator,
    op_name: str,
    *,
    require_even: bool = False,
) -> tuple[int, int, int]:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError(f"{op_name}: output must be contiguous.")
    if require_even and n % 2 != 0:
        raise ValueError(f"{op_name}: Philox Box-Muller path requires an even element count, got {n}.")
    offset_val = int(getattr(generator, "offset", 0))
    if offset_val < 0:
        raise ValueError(f"{op_name}: Philox offset must be >= 0, got {offset_val}.")
    return int(getattr(generator, "seed", 0)), offset_val, n
