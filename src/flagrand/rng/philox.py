from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from flagrand.runtime import (
    CachedKernelLauncher,
    HAS_LIBTUNER,
    require_libtuner,
    use_libtuner,
)
from flagrand.runtime.libtuner import PHILOX_LIBTUNER_CONFIGS, libentry, libtuner


@triton.jit
def _philox_generate(seed, counter):
    c0 = (tl.zeros_like(counter)).to(tl.uint32)
    c1 = (tl.zeros_like(counter)).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed, c0, c1, c2, c3)
    return r0, r1, r2, r3


@triton.jit
def _philox_window4(seed, counter, SHIFT: tl.constexpr):
    a0, a1, a2, a3 = _philox_generate(seed, counter)
    if SHIFT == 0:
        return a0, a2, a1, a3

    b0, b1, b2, b3 = _philox_generate(seed, counter + 1)
    if SHIFT == 1:
        return a2, a1, a3, b0
    if SHIFT == 2:
        return a1, a3, b0, b2
    return a3, b0, b2, b1


@triton.jit
def _philox_kernel(out_ptr, seed, base_counter, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    counter = base_counter + offs
    r0, r1, r2, r3 = _philox_generate(seed, counter)
    # Build a (BLOCK, 4) tile and store with one 2D op so the compiler can
    # emit 16-byte vector stores (4 contiguous uint32 per counter).
    r01 = tl.join(r0, r1)            # (BLOCK, 2)
    r23 = tl.join(r2, r3)            # (BLOCK, 2)
    tile = tl.join(r01, r23)         # (BLOCK, 2, 2)
    tile = tl.reshape(tile, (BLOCK, 4))
    base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
    tl.store(out_ptr + base, tile.to(tl.int32, bitcast=True), mask=mask[:, None])


@triton.jit
def _philox_generic_kernel(
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
    base = group * 4
    tl.store(out_ptr + base + 0, r0.to(tl.int32, bitcast=True), mask=group_mask & (base + 0 < n))
    tl.store(out_ptr + base + 1, r1.to(tl.int32, bitcast=True), mask=group_mask & (base + 1 < n))
    tl.store(out_ptr + base + 2, r2.to(tl.int32, bitcast=True), mask=group_mask & (base + 2 < n))
    tl.store(out_ptr + base + 3, r3.to(tl.int32, bitcast=True), mask=group_mask & (base + 3 < n))


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
    def _philox_kernel_tuned(
        out_ptr,
        seed,
        base_counter,
        n_counters,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_counters
        counter = base_counter + offs
        r0, r1, r2, r3 = _philox_generate(seed, counter)
        r01 = tl.join(r0, r1)
        r23 = tl.join(r2, r3)
        tile = tl.join(r01, r23)
        tile = tl.reshape(tile, (BLOCK, 4))
        base = (offs * 4)[:, None] + tl.arange(0, 4)[None, :]
        tl.store(out_ptr + base, tile.to(tl.int32, bitcast=True), mask=mask[:, None])

else:
    _philox_kernel_tuned = None


_PHILOX_RAW_LAUNCHER = CachedKernelLauncher(
    _philox_kernel,
    constexpr_names=("BLOCK",),
)
_PHILOX_RAW_GENERIC_LAUNCHER = CachedKernelLauncher(
    _philox_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)


@dataclass
class PhiloxGenerator:
    seed: int = 0
    offset: int = 0

    @property
    def dimensions(self) -> None:
        return None

    def generate(
        self,
        out: torch.Tensor,
        *,
        seed: int | None = None,
        offset: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        block_size = kwargs.get("block_size", 512)
        num_warps = kwargs.get("num_warps", 4)
        manual_config = "block_size" in kwargs or "num_warps" in kwargs

        return generate_philox_raw_u32(
            out,
            self,
            seed=seed,
            offset=offset,
            block_size=block_size,
            num_warps=num_warps,
            use_libtuner_kernel=use_libtuner() and not manual_config,
        )


def generate_philox_raw_u32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    seed: int | None = None,
    offset: int | None = None,
    block_size: int = 512,
    num_warps: int = 4,
    use_libtuner_kernel: bool = False,
) -> torch.Tensor:
    n = out.numel()
    if not out.is_contiguous():
        raise ValueError("Philox: output must be contiguous.")
    if block_size <= 0:
        raise ValueError(f"Philox: block_size must be > 0, got {block_size}.")
    if num_warps <= 0:
        raise ValueError(f"Philox: num_warps must be > 0, got {num_warps}.")

    seed_val = generator.seed if seed is None else int(seed)
    offset_val = generator.offset if offset is None else int(offset)
    if offset_val < 0:
        raise ValueError(f"Philox: offset must be >= 0, got {offset_val}.")

    if n:
        base_counter = offset_val // 4
        if offset_val % 4 == 0 and n % 4 == 0:
            n_counters = n // 4
            if use_libtuner_kernel:
                require_libtuner()
                grid = lambda meta: (triton.cdiv(n_counters, meta["BLOCK"]),)
                _philox_kernel_tuned[grid](out, seed_val, base_counter, n_counters)
            else:
                grid = triton.cdiv(n_counters, block_size)
                _PHILOX_RAW_LAUNCHER.launch(
                    grid,
                    (out, seed_val, base_counter, n_counters),
                    (block_size,),
                    (num_warps,),
                )
        else:
            n_groups = triton.cdiv(n, 4)
            grid = triton.cdiv(n_groups, block_size)
            _PHILOX_RAW_GENERIC_LAUNCHER.launch(
                grid,
                (out, seed_val, base_counter, n, n_groups),
                (offset_val % 4, block_size),
                (num_warps,),
            )
    if seed is None and offset is None:
        generator.offset = offset_val + n
    return out
