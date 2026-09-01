from __future__ import annotations

import torch

from flagrand._curand_philox_fast_path import _prepare
from flagrand.ascend.huawei910c import (
    generate_huawei910c_native_f64,
    generate_huawei910c_normal_f32_fast,
    generate_huawei910c_raw_uniform_f32_fast,
)

_DEFAULT_BLOCK_SIZE = 64


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs, default_block_size=_DEFAULT_BLOCK_SIZE)
    if prepared is None:
        return False
    engine, _, num_warps, _ = prepared
    if out.numel():
        generate_huawei910c_raw_uniform_f32_fast(
            out,
            engine,
            uniform=False,
            num_warps=num_warps,
        )
    return True


def try_generate_uniform(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs, default_block_size=_DEFAULT_BLOCK_SIZE)
    if prepared is None:
        return False
    engine, _, num_warps, _ = prepared
    if out.numel() == 0:
        return True
    if out.dtype == torch.float64:
        generate_huawei910c_native_f64(
            out,
            engine,
            distribution="uniform_f64",
            mean=0.0,
            stddev=1.0,
        )
    else:
        generate_huawei910c_raw_uniform_f32_fast(
            out,
            engine,
            uniform=True,
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
    prepared = _prepare(generator, out, kwargs, default_block_size=_DEFAULT_BLOCK_SIZE)
    if prepared is None:
        return False
    engine, _, num_warps, _ = prepared
    if out.numel() == 0:
        return True
    if out.dtype == torch.float32:
        generate_huawei910c_normal_f32_fast(
            out,
            engine,
            mean=mean,
            stddev=stddev,
            lognormal=lognormal,
            num_warps=num_warps,
        )
    else:
        generate_huawei910c_native_f64(
            out,
            engine,
            distribution=f"{'lognormal' if lognormal else 'normal'}_f64",
            mean=mean,
            stddev=stddev,
        )
    return True


__all__ = ["try_generate_normal", "try_generate_raw", "try_generate_uniform"]
