from __future__ import annotations

import torch

from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CurandGenerator,
)
from flagrand.fused._internal.philox_direct import (
    generate_philox_lognormal_f32,
    generate_philox_normal_f32,
    generate_philox_uniform_f32,
)
from flagrand.fused._internal.philox_direct_f64 import (
    generate_philox_lognormal_f64,
    generate_philox_normal_f64,
    generate_philox_uniform_f64,
)
from flagrand.rng.philox import PhiloxGenerator, generate_philox_raw_u32


_PHILOX_F64_NORMAL_CONFIG = (128, 4)


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps = prepared
    if out.numel():
        generate_philox_raw_u32(
            out,
            engine,
            block_size=block_size,
            num_warps=num_warps,
        )
    return True


def try_generate_uniform(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps = prepared
    if out.numel() == 0:
        return True
    operation = (
        generate_philox_uniform_f64
        if out.dtype == torch.float64
        else generate_philox_uniform_f32
    )
    operation(
        out,
        engine,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


def try_generate_normal(
    generator: object,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    kwargs: dict[str, object],
    lognormal: bool,
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps = prepared
    if not kwargs and out.dtype == torch.float64:
        block_size, num_warps = _PHILOX_F64_NORMAL_CONFIG
    if out.numel() == 0:
        return True
    if out.dtype == torch.float64:
        operation = (
            generate_philox_lognormal_f64
            if lognormal
            else generate_philox_normal_f64
        )
    else:
        operation = (
            generate_philox_lognormal_f32
            if lognormal
            else generate_philox_normal_f32
        )
    operation(
        out,
        engine,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


def _prepare(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> tuple[PhiloxGenerator, int, int] | None:
    if (
        not isinstance(generator, CurandGenerator)
        or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10
    ):
        return None
    config = _launch_config(kwargs)
    if config is None or out.device.type != "cuda" or not out.is_contiguous():
        return None
    return generator.engine, config[0], config[1]


def _launch_config(kwargs: dict[str, object]) -> tuple[int, int] | None:
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    block_size = kwargs.get("block_size", 512)
    num_warps = kwargs.get("num_warps", 4)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return block_size, num_warps
