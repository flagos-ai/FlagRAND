from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

from flagrand._curand_handle import CURAND_RNG_PSEUDO_PHILOX4_32_10, CurandGenerator
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
from flagrand.runtime import current_backend, use_libtuner

_PHILOX_F64_NORMAL_CONFIG = (128, 4)
_ASCEND_DEFAULT_BLOCK_SIZE = 64
_CUDA_DEFAULT_BLOCK_SIZE = 512

AscendGenerator = Callable[..., object]


@lru_cache(maxsize=1)
def _ascend_generators() -> tuple[AscendGenerator, AscendGenerator]:
    """Load Ascend kernels only after an NPU tensor selects that backend."""

    from flagrand.ascend.huawei910c import generate_huawei910c, generate_huawei910c_native_f64

    return generate_huawei910c, generate_huawei910c_native_f64


@lru_cache(maxsize=1)
def _ascend_normal_f32_generator() -> AscendGenerator:
    from flagrand.ascend.huawei910c import generate_huawei910c_normal_f32_fast

    return generate_huawei910c_normal_f32_fast


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, manual_config = prepared
    if out.numel():
        if out.device.type == "npu":
            generate_huawei910c, _ = _ascend_generators()
            generate_huawei910c(
                out,
                engine,
                distribution="raw32",
                num_warps=num_warps,
            )
        else:
            generate_philox_raw_u32(
                out,
                engine,
                block_size=block_size,
                num_warps=num_warps,
                use_libtuner_kernel=use_libtuner() and not manual_config,
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

    engine, block_size, num_warps, manual_config = prepared
    if out.numel() == 0:
        return True

    if out.device.type == "npu":
        generate_huawei910c, generate_huawei910c_native_f64 = _ascend_generators()
        if out.dtype == torch.float64:
            generate_huawei910c_native_f64(
                out,
                engine,
                distribution="uniform_f64",
                mean=0.0,
                stddev=1.0,
            )
        else:
            generate_huawei910c(
                out,
                engine,
                distribution="uniform_f32",
                num_warps=num_warps,
            )
        return True

    operation = generate_philox_uniform_f64 if out.dtype == torch.float64 else generate_philox_uniform_f32
    operation(
        out,
        engine,
        block_size=block_size,
        num_warps=num_warps,
        **(
            {"use_libtuner_kernel": True} if out.dtype == torch.float32 and use_libtuner() and not manual_config else {}
        ),
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
    if out.device.type == "npu" and out.dtype == torch.float32:
        if (
            not isinstance(generator, CurandGenerator)
            or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10
            or not out.is_contiguous()
            or any(name not in {"block_size", "num_warps"} for name in kwargs)
        ):
            return False

        block_size = kwargs.get("block_size", _ASCEND_DEFAULT_BLOCK_SIZE)
        num_warps = kwargs.get("num_warps", 4)
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
        if not isinstance(num_warps, int) or num_warps <= 0:
            raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")

        if out.numel():
            generate_huawei910c_normal_f32_fast = _ascend_normal_f32_generator()
            generate_huawei910c_normal_f32_fast(
                out,
                generator.engine,
                mean=mean,
                stddev=stddev,
                lognormal=lognormal,
                num_warps=num_warps,
            )
        return True

    prepared = _prepare(generator, out, kwargs)
    if prepared is None:
        return False

    engine, block_size, num_warps, manual_config = prepared
    if not kwargs and out.dtype == torch.float64:
        block_size, num_warps = _PHILOX_F64_NORMAL_CONFIG

    if out.numel() == 0:
        return True

    if out.device.type == "npu":
        generate_huawei910c, generate_huawei910c_native_f64 = _ascend_generators()
        suffix = "f64" if out.dtype == torch.float64 else "f32"
        distribution = f"{'lognormal' if lognormal else 'normal'}_{suffix}"

        if out.dtype == torch.float64:
            generate_huawei910c_native_f64(
                out,
                engine,
                distribution=distribution,
                mean=mean,
                stddev=stddev,
            )
        else:
            generate_huawei910c(
                out,
                engine,
                distribution=distribution,
                mean=mean,
                stddev=stddev,
                num_warps=num_warps,
            )
        return True

    if out.dtype == torch.float64:
        operation = generate_philox_lognormal_f64 if lognormal else generate_philox_normal_f64
    else:
        operation = generate_philox_lognormal_f32 if lognormal else generate_philox_normal_f32

    operation(
        out,
        engine,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        **(
            {"use_libtuner_kernel": True}
            if (use_libtuner() and not manual_config and not lognormal and out.dtype in {torch.float32, torch.float64})
            else {}
        ),
    )
    return True


def _prepare(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> tuple[PhiloxGenerator, int, int, bool] | None:
    if not isinstance(generator, CurandGenerator) or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return None
    if out.device.type == "cuda":
        current_backend()
    default_block_size = _ASCEND_DEFAULT_BLOCK_SIZE if out.device.type == "npu" else _CUDA_DEFAULT_BLOCK_SIZE
    config = _launch_config(kwargs, default_block_size=default_block_size)
    if config is None or out.device.type not in {"cuda", "npu"} or not out.is_contiguous():
        return None
    return generator.engine, config[0], config[1], bool(kwargs)


def _launch_config(
    kwargs: dict[str, object],
    *,
    default_block_size: int = _CUDA_DEFAULT_BLOCK_SIZE,
) -> tuple[int, int] | None:
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    block_size = kwargs.get("block_size", default_block_size)
    num_warps = kwargs.get("num_warps", 4)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return block_size, num_warps
