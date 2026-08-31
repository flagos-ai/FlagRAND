from __future__ import annotations

import torch

from flagrand._curand_philox_fast_path import _prepare
from flagrand.fused._internal.philox_direct import (
    generate_philox_lognormal_f32,
    generate_philox_normal_f32,
    generate_philox_uniform_f32,
)
from flagrand.nvidia import (
    generate_nvidia_philox_lognormal_f64,
    generate_nvidia_philox_normal_f64,
    generate_nvidia_philox_uniform_f64,
)
from flagrand.rng.philox import generate_philox_raw_u32

_F64_NORMAL_CONFIG = (128, 4)


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, _ = prepared
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
    engine, block_size, num_warps, _ = prepared
    if out.numel() == 0:
        return True
    operation = generate_nvidia_philox_uniform_f64 if out.dtype == torch.float64 else generate_philox_uniform_f32
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
    engine, block_size, num_warps, _ = prepared
    if out.numel() == 0:
        return True
    if out.dtype == torch.float32:
        operation = generate_philox_lognormal_f32 if lognormal else generate_philox_normal_f32
        operation(
            out,
            engine,
            mean=mean,
            stddev=stddev,
            block_size=block_size,
            num_warps=num_warps,
            _portable_math=False,
        )
        return True
    if not kwargs:
        block_size, num_warps = _F64_NORMAL_CONFIG
    operation = generate_nvidia_philox_lognormal_f64 if lognormal else generate_nvidia_philox_normal_f64
    operation(
        out,
        engine,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
    )
    return True


__all__ = ["try_generate_normal", "try_generate_raw", "try_generate_uniform"]
