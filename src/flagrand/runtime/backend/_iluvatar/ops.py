from __future__ import annotations

import torch

from flagrand._curand_philox_fast_path import _prepare
from flagrand._curand_philox_fast_path import try_generate_normal as generic_try_generate_normal
from flagrand.corex import (
    generate_corex_lognormal_f32,
    generate_corex_normal_f32,
    generate_corex_raw_u32,
    generate_corex_uniform_f32,
    generate_corex_uniform_f64,
)


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
        generate_corex_raw_u32(
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
    operation = generate_corex_uniform_f64 if out.dtype == torch.float64 else generate_corex_uniform_f32
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
    if out.dtype != torch.float32:
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
    if out.numel():
        operation = generate_corex_lognormal_f32 if lognormal else generate_corex_normal_f32
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
