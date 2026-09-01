from __future__ import annotations

import torch

from flagrand._curand_handle import CURAND_RNG_PSEUDO_PHILOX4_32_10, CurandGenerator
from flagrand.metax import (
    generate_metax_normal_f32,
    generate_metax_normal_f64,
    generate_metax_raw_u32,
    generate_metax_uniform_f32,
    generate_metax_uniform_f64,
)
from flagrand.runtime import use_libtuner

_DEFAULT_BLOCK_SIZE = 512
_F64_NORMAL_CONFIG = (128, 4)


def generate(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.int32, "generate")
    _require_handled(try_generate_raw(generator, out, kwargs), "generate")
    return out


def generate_uniform(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_uniform")
    _require_handled(try_generate_uniform(generator, out, kwargs), "generate_uniform")
    return out


def generate_uniform_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_uniform_double")
    _require_handled(try_generate_uniform(generator, out, kwargs), "generate_uniform_double")
    return out


def generate_normal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_normal")
    _require_handled(
        try_generate_normal(
            generator,
            out,
            mean=mean,
            stddev=stddev,
            kwargs=kwargs,
            lognormal=False,
        ),
        "generate_normal",
    )
    return out


def generate_normal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_normal_double")
    _require_handled(
        try_generate_normal(
            generator,
            out,
            mean=mean,
            stddev=stddev,
            kwargs=kwargs,
            lognormal=False,
        ),
        "generate_normal_double",
    )
    return out


def generate_lognormal(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float32, "generate_lognormal")
    _require_handled(
        try_generate_normal(
            generator,
            out,
            mean=mean,
            stddev=stddev,
            kwargs=kwargs,
            lognormal=True,
        ),
        "generate_lognormal",
    )
    return out


def generate_lognormal_double(
    generator: CurandGenerator,
    out: torch.Tensor,
    *,
    mean: float,
    stddev: float,
    **kwargs: object,
) -> torch.Tensor:
    _require_dtype(out, torch.float64, "generate_lognormal_double")
    _require_handled(
        try_generate_normal(
            generator,
            out,
            mean=mean,
            stddev=stddev,
            kwargs=kwargs,
            lognormal=True,
        ),
        "generate_lognormal_double",
    )
    return out


def try_generate_raw(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> bool:
    prepared = _prepare_metax(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, manual_config = prepared
    if out.numel():
        generate_metax_raw_u32(
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
    prepared = _prepare_metax(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, manual_config = prepared
    if out.numel() == 0:
        return True
    if out.dtype == torch.float64:
        generate_metax_uniform_f64(
            out,
            engine,
            block_size=block_size,
            num_warps=num_warps,
        )
    else:
        generate_metax_uniform_f32(
            out,
            engine,
            block_size=block_size,
            num_warps=num_warps,
            use_libtuner_kernel=use_libtuner() and not manual_config,
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
    prepared = _prepare_metax(generator, out, kwargs)
    if prepared is None:
        return False
    engine, block_size, num_warps, manual_config = prepared
    if out.numel() == 0:
        return True
    if out.dtype == torch.float32:
        generate_metax_normal_f32(
            out,
            engine,
            mean=mean,
            stddev=stddev,
            block_size=block_size,
            num_warps=num_warps,
            lognormal=lognormal,
            use_libtuner_kernel=use_libtuner() and not manual_config and not lognormal,
        )
        return True

    if not kwargs:
        block_size, num_warps = _F64_NORMAL_CONFIG
    generate_metax_normal_f64(
        out,
        engine,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=lognormal,
        use_libtuner_kernel=use_libtuner() and not manual_config and not lognormal,
    )
    return True


def _prepare_metax(
    generator: object,
    out: torch.Tensor,
    kwargs: dict[str, object],
) -> tuple[object, int, int, bool] | None:
    if not isinstance(generator, CurandGenerator) or generator.rng_type != CURAND_RNG_PSEUDO_PHILOX4_32_10:
        return None
    config = _launch_config(kwargs)
    if config is None or out.device.type != "cuda" or not out.is_contiguous():
        return None
    return generator.engine, config[0], config[1], bool(kwargs)


def _launch_config(kwargs: dict[str, object]) -> tuple[int, int] | None:
    if any(name not in {"block_size", "num_warps"} for name in kwargs):
        return None
    block_size = kwargs.get("block_size", _DEFAULT_BLOCK_SIZE)
    num_warps = kwargs.get("num_warps", 4)
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}.")
    if not isinstance(num_warps, int) or num_warps <= 0:
        raise ValueError(f"num_warps must be a positive integer, got {num_warps!r}.")
    return block_size, num_warps


def _require_dtype(out: torch.Tensor, dtype: torch.dtype, op_name: str) -> None:
    if out.dtype != dtype:
        raise TypeError(f"{op_name}: expected output dtype {dtype}, got {out.dtype}.")


def _require_handled(handled: bool, op_name: str) -> None:
    if not handled:
        raise ValueError(
            f"{op_name}: requires a Philox CurandGenerator, a "
            "contiguous CUDA/NPU output, and only block_size/num_warps options."
        )


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
