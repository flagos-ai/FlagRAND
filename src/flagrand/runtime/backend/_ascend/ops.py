from __future__ import annotations

import torch

from flagrand._curand_handle import CURAND_RNG_PSEUDO_PHILOX4_32_10, CurandGenerator
from flagrand._curand_philox_fast_path import _prepare
from flagrand.ascend.huawei910c import (
    _generate_huawei910c_normal_f32_prepared,
    _generate_huawei910c_raw_uniform_f32_prepared,
    generate_huawei910c_native_f64,
    generate_huawei910c_normal_f32_fast,
    generate_huawei910c_raw_uniform_f32_fast,
)

_DEFAULT_BLOCK_SIZE = 64


def generate(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.int32:
        raise TypeError(f"generate: expected output dtype {torch.int32}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate")
    if count:
        _generate_huawei910c_raw_uniform_f32_prepared(
            out,
            engine,
            count,
            uniform=False,
            num_warps=num_warps,
        )
    return out


def generate_uniform(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_uniform: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_uniform")
    if count:
        _generate_huawei910c_raw_uniform_f32_prepared(
            out,
            engine,
            count,
            uniform=True,
            num_warps=num_warps,
        )
    return out


def generate_uniform_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_uniform_double: expected output dtype {torch.float64}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_uniform_double")
    if count:
        generate_huawei910c_native_f64(
            out,
            engine,
            distribution="uniform_f64",
            mean=0.0,
            stddev=1.0,
        )
    return out


def generate_normal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_normal: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_normal")
    if count:
        _generate_huawei910c_normal_f32_prepared(
            out,
            engine,
            count,
            mean=mean,
            stddev=stddev,
            lognormal=False,
            num_warps=num_warps,
        )
    return out


def generate_normal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_normal_double: expected output dtype {torch.float64}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_normal_double")
    if count:
        generate_huawei910c_native_f64(
            out,
            engine,
            distribution="normal_f64",
            mean=mean,
            stddev=stddev,
        )
    return out


def generate_lognormal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_lognormal: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_lognormal")
    if count:
        _generate_huawei910c_normal_f32_prepared(
            out,
            engine,
            count,
            mean=mean,
            stddev=stddev,
            lognormal=True,
            num_warps=num_warps,
        )
    return out


def generate_lognormal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    num_warps: int = 4,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_lognormal_double: expected output dtype {torch.float64}, got {out.dtype}.")
    engine, count = _prepare_ascend(generator, out, block_size, num_warps, kwargs, "generate_lognormal_double")
    if count:
        generate_huawei910c_native_f64(
            out,
            engine,
            distribution="lognormal_f64",
            mean=mean,
            stddev=stddev,
        )
    return out


def _prepare_ascend(
    generator: object,
    out: torch.Tensor,
    block_size: object,
    num_warps: object,
    kwargs: dict[str, object],
    op_name: str,
) -> tuple[object, int]:
    if not isinstance(generator, CurandGenerator) or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        _unsupported(op_name)
    if kwargs:
        _unsupported(op_name)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    if out.device.type != "npu" or not out.is_contiguous():
        _unsupported(op_name)
    return generator.engine, out.numel()


def _unsupported(op_name: str) -> None:
    raise ValueError(
        f"{op_name}: requires a Philox CurandGenerator, a "
        "contiguous CUDA/NPU output, and only block_size/num_warps options."
    )


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


__all__ = [
    "generate",
    "generate_lognormal",
    "generate_lognormal_double",
    "generate_normal",
    "generate_normal_double",
    "generate_uniform",
    "generate_uniform_double",
    "try_generate_normal",
    "try_generate_raw",
    "try_generate_uniform",
]
