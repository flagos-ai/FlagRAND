from __future__ import annotations

import torch

from flagrand._curand_handle import CURAND_RNG_PSEUDO_PHILOX4_32_10, CurandGenerator
from flagrand._curand_philox_fast_path import _launch_config
from flagrand._curand_philox_fast_path import _prepare as _generic_prepare
from flagrand.fused._internal.philox_direct import (
    _launch_normal_like_f32,
    _launch_philox_uniform_f32,
    generate_philox_lognormal_f32,
    generate_philox_normal_f32,
    generate_philox_uniform_f32,
)
from flagrand.nvidia import (
    generate_nvidia_philox_lognormal_f64,
    generate_nvidia_philox_normal_f64,
    generate_nvidia_philox_uniform_f64,
)
from flagrand.nvidia.philox import _launch_normal_like as _launch_normal_like_f64
from flagrand.nvidia.philox import _launch_uniform as _launch_uniform_f64
from flagrand.rng.philox import _launch_philox_raw_u32, generate_philox_raw_u32

_F64_NORMAL_CONFIG = (128, 4)
_DEFAULT_CONFIG = (512, 4)


def generate(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.int32:
        raise TypeError(f"generate: expected output dtype {torch.int32}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_nvidia(generator, out, kwargs, "generate")
    seed = engine.seed
    offset = engine.offset
    _launch_philox_raw_u32(out, seed, offset, n, block_size, num_warps)
    engine.offset = offset + n
    return out


def generate_uniform(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_uniform: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_nvidia(generator, out, kwargs, "generate_uniform")
    seed = engine.seed
    offset = engine.offset
    _launch_philox_uniform_f32(out, seed, offset, n, block_size, num_warps)
    engine.offset = offset + n
    return out


def generate_uniform_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_uniform_double: expected output dtype {torch.float64}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_nvidia(generator, out, kwargs, "generate_uniform_double")
    seed = engine.seed
    offset = engine.offset
    _launch_uniform_f64(out, seed, offset, n, block_size, num_warps)
    engine.offset = offset + n
    return out


def generate_normal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_normal: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_nvidia(generator, out, kwargs, "generate_normal")
    if n % 2:
        raise ValueError(f"generate_normal: Philox Box-Muller path requires an even element count, got {n}.")
    seed = engine.seed
    offset = engine.offset
    _launch_normal_like_f32(
        out,
        n,
        seed,
        offset,
        mean,
        stddev,
        block_size,
        num_warps,
        False,
        lognormal=False,
        use_libtuner_kernel=False,
    )
    engine.offset = offset + n
    return out


def generate_normal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_normal_double: expected output dtype {torch.float64}, got {out.dtype}.")
    default_config = _F64_NORMAL_CONFIG if not kwargs else _DEFAULT_CONFIG
    engine, n, block_size, num_warps = _prepare_nvidia(
        generator,
        out,
        kwargs,
        "generate_normal_double",
        default_config,
    )
    if n % 2:
        raise ValueError(f"generate_normal_double: Philox float64 path requires an even element count, got {n}.")
    seed = engine.seed
    offset = engine.offset
    _launch_normal_like_f64(out, seed, offset, n, mean, stddev, block_size, num_warps, False)
    engine.offset = offset + n
    return out


def generate_lognormal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float32:
        raise TypeError(f"generate_lognormal: expected output dtype {torch.float32}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_nvidia(generator, out, kwargs, "generate_lognormal")
    if n % 2:
        raise ValueError(f"generate_lognormal: Philox Box-Muller path requires an even element count, got {n}.")
    seed = engine.seed
    offset = engine.offset
    _launch_normal_like_f32(
        out,
        n,
        seed,
        offset,
        mean,
        stddev,
        block_size,
        num_warps,
        False,
        lognormal=True,
        use_libtuner_kernel=False,
    )
    engine.offset = offset + n
    return out


def generate_lognormal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.float64:
        raise TypeError(f"generate_lognormal_double: expected output dtype {torch.float64}, got {out.dtype}.")
    default_config = _F64_NORMAL_CONFIG if not kwargs else _DEFAULT_CONFIG
    engine, n, block_size, num_warps = _prepare_nvidia(
        generator,
        out,
        kwargs,
        "generate_lognormal_double",
        default_config,
    )
    if n % 2:
        raise ValueError(f"generate_lognormal_double: Philox float64 path requires an even element count, got {n}.")
    seed = engine.seed
    offset = engine.offset
    _launch_normal_like_f64(out, seed, offset, n, mean, stddev, block_size, num_warps, True)
    engine.offset = offset + n
    return out


def _prepare_nvidia(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
    op_name: str,
    default_config: tuple[int, int] = _DEFAULT_CONFIG,
):
    if not isinstance(generator, CurandGenerator) or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        _unsupported(op_name)
    if kwargs:
        config = _launch_config(kwargs, default_block_size=default_config[0])
        if config is None:
            _unsupported(op_name)
    else:
        config = default_config
    if out.device.type != "cuda" or not out.is_contiguous():
        _unsupported(op_name)
    return generator.engine, out.numel(), config[0], config[1]


def _unsupported(op_name: str) -> None:
    raise ValueError(
        f"{op_name}: this preview requires a Philox CurandGenerator, a "
        "contiguous CUDA/NPU output, and only block_size/num_warps options."
    )


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _generic_prepare(generator, out, kwargs)
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
    prepared = _generic_prepare(generator, out, kwargs)
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
    prepared = _generic_prepare(generator, out, kwargs)
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


__all__ = [
    "generate",
    "generate_uniform",
    "generate_uniform_double",
    "generate_normal",
    "generate_normal_double",
    "generate_lognormal",
    "generate_lognormal_double",
    "try_generate_normal",
    "try_generate_raw",
    "try_generate_uniform",
]
