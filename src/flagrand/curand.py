from __future__ import annotations

import torch

from flagrand._curand_handle import (
    CURAND_RNG_PSEUDO_PHILOX4_32_10,
    CurandGenerator,
    create_generator,
    set_generator_offset,
    set_pseudo_random_generator_seed,
)
from flagrand._curand_philox_fast_path import try_generate_normal, try_generate_raw, try_generate_uniform


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
    _require_handled(
        try_generate_uniform(generator, out, kwargs),
        "generate_uniform_double",
    )
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
    if not try_generate_normal(
        generator,
        out,
        mean=mean,
        stddev=stddev,
        kwargs=kwargs,
        lognormal=False,
    ):
        _require_handled(False, "generate_normal")
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
    if out.dtype != torch.float32:
        raise TypeError(f"generate_lognormal: expected output dtype {torch.float32}, got {out.dtype}.")
    if not try_generate_normal(
        generator,
        out,
        mean=mean,
        stddev=stddev,
        kwargs=kwargs,
        lognormal=True,
    ):
        _require_handled(False, "generate_lognormal")
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


def _require_dtype(out: torch.Tensor, dtype: torch.dtype, op_name: str) -> None:
    if out.dtype != dtype:
        raise TypeError(f"{op_name}: expected output dtype {dtype}, got {out.dtype}.")


def _require_handled(handled: bool, op_name: str) -> None:
    if not handled:
        raise ValueError(
            f"{op_name}: this preview requires a Philox CurandGenerator, a "
            "contiguous CUDA/NPU output, and only block_size/num_warps options."
        )


__all__ = [
    "CURAND_RNG_PSEUDO_PHILOX4_32_10",
    "CurandGenerator",
    "create_generator",
    "set_pseudo_random_generator_seed",
    "set_generator_offset",
    "generate",
    "generate_uniform",
    "generate_uniform_double",
    "generate_normal",
    "generate_normal_double",
    "generate_lognormal",
    "generate_lognormal_double",
]
