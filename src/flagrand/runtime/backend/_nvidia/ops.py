from __future__ import annotations

import torch

from flagrand._curand_philox_fast_path import _prepare
from flagrand._curand_philox_fast_path import try_generate_normal as generic_try_generate_normal
from flagrand._curand_philox_fast_path import try_generate_raw as generic_try_generate_raw
from flagrand._curand_philox_fast_path import try_generate_uniform as generic_try_generate_uniform
from flagrand.nvidia import (
    generate_nvidia_philox_lognormal_f64,
    generate_nvidia_philox_normal_f64,
    generate_nvidia_philox_uniform_f64,
)

_F64_NORMAL_CONFIG = (128, 4)


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    return generic_try_generate_raw(generator, out, kwargs)


def try_generate_uniform(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    if out.dtype != torch.float64:
        return generic_try_generate_uniform(generator, out, kwargs)
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, _ = prepared
    generate_nvidia_philox_uniform_f64(
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
    if out.dtype != torch.float64:
        return generic_try_generate_normal(
            generator,
            out,
            mean=mean,
            stddev=stddev,
            kwargs=kwargs,
            lognormal=lognormal,
        )
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, _ = prepared
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
