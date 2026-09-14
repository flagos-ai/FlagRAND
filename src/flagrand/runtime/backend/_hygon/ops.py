from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

from flagrand._curand_handle import CURAND_RNG_PSEUDO_PHILOX4_32_10, CurandGenerator
from flagrand._curand_philox_fast_path import _launch_config, _prepare
from flagrand._curand_philox_fast_path import try_generate_normal as _portable_normal
from flagrand.fused._internal.philox_direct import _launch_normal_like_f32, _launch_philox_uniform_f32
from flagrand.fused._internal.philox_direct_f64 import (
    _PHILOX_UNIFORM_F64_GENERIC_LAUNCHER,
    _PHILOX_UNIFORM_F64_LAUNCHER,
    _philox_launch_args_f64,
)
from flagrand.fused._internal.transforms import _sincos_poly_f64, uint32_pair_to_uniform64_curand_compat
from flagrand.rng.philox import _launch_philox_raw_u32, _philox_window4
from flagrand.runtime import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

_DEFAULT_CONFIG = (256, 4)
_F32_NORMAL_CONFIG = (128, 2)
_F64_NORMAL_CONFIG = (256, 4)


@triton.jit
def _normal_f64(
    OUT,
    seed,
    base_counter,
    n,
    mean_bits,
    stddev_bits,
    SHIFT: tl.constexpr,
    LOGNORMAL: tl.constexpr,
    STANDARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r0, r1, r2, r3 = _philox_window4(seed, base_counter + group.to(tl.uint64), SHIFT)
    u0 = uint32_pair_to_uniform64_curand_compat(r0, r1)
    u1 = uint32_pair_to_uniform64_curand_compat(r2, r3)
    radius = libdevice.sqrt(-2.0 * libdevice.log(tl.maximum(u0, 1.1102230246251565e-16)))
    sine, cosine = _sincos_poly_f64(u1)
    # Python float launch arguments default to f32 in Triton. Transport their
    # IEEE-754 bits as runtime integers so varying mean/stddev stays binary64.
    if STANDARD:
        y0, y1 = radius * cosine, radius * sine
    else:
        mean = tl.cast(mean_bits, tl.uint64).to(tl.float64, bitcast=True)
        stddev = tl.cast(stddev_bits, tl.uint64).to(tl.float64, bitcast=True)
        y0 = mean + stddev * radius * cosine
        y1 = mean + stddev * radius * sine
    if LOGNORMAL:
        y0 = libdevice.exp(y0)
        y1 = libdevice.exp(y1)
    offsets = group[:, None] * 2 + tl.arange(0, 2)[None, :]
    tl.store(OUT + offsets, tl.join(y0, y1), offsets < n)


_NORMAL = CachedKernelLauncher(_normal_f64, constexpr_names=("SHIFT", "LOGNORMAL", "STANDARD", "BLOCK"))


def try_generate_normal(generator, out, *, mean, stddev, kwargs, lognormal):
    if out.dtype != torch.float64:
        return _portable_normal(generator, out, mean=mean, stddev=stddev, kwargs=kwargs, lognormal=lognormal)
    prepared = _prepare(generator, out, kwargs, default_block_size=128)
    if prepared is None:
        return False
    engine, block, warps, _ = prepared
    seed, offset = _philox_launch_args_f64(out, engine, "Hygon Philox float64", require_even=True)
    n = out.numel()
    _launch_normal_like_f64(out, seed, offset, n, mean, stddev, block, warps, lognormal)
    engine.offset = offset + 2 * n
    return True


def _launch_normal_like_f64(out, seed, offset, n, mean, stddev, block, warps, lognormal):
    if n:
        standard = mean == 0.0 and stddev == 1.0
        mean_bits = 0 if standard else struct.unpack("<Q", struct.pack("<d", mean))[0]
        stddev_bits = 0 if standard else struct.unpack("<Q", struct.pack("<d", stddev))[0]
        counter = offset // 4
        _NORMAL.launch(
            (n // 2 + block - 1) // block,
            (out, seed, counter, n, mean_bits, stddev_bits),
            (offset % 4, lognormal, standard, block),
            (warps,),
            specialization_key=(
                out.device,
                out.dtype,
                out.data_ptr() % 16 == 0,
                seed,
                integer_specialization_key(counter),
                n,
                mean_bits,
                stddev_bits,
                offset % 4,
                lognormal,
                standard,
                block,
                warps,
            ),
        )


def _launch_uniform_f64(out, seed, offset, n, block, warps):
    if not n:
        return
    counter = offset // 4
    groups = (n + 1) // 2
    key = (
        out.device,
        out.dtype,
        out.data_ptr() % 16 == 0,
        seed,
        integer_specialization_key(counter),
        n,
        offset % 4,
        block,
        warps,
    )
    if offset % 4 == 0 and n % 2 == 0:
        _PHILOX_UNIFORM_F64_LAUNCHER.launch(
            (groups + block - 1) // block, (out, seed, counter, n, groups), (block,), (warps,), specialization_key=key
        )
    else:
        _PHILOX_UNIFORM_F64_GENERIC_LAUNCHER.launch(
            (groups + block - 1) // block,
            (out, seed, counter, n, groups),
            (offset % 4, block),
            (warps,),
            specialization_key=key,
        )


def generate(
    generator: CurandGenerator,
    out: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    if out.dtype != torch.int32:
        raise TypeError(f"generate: expected output dtype {torch.int32}, got {out.dtype}.")
    engine, n, block_size, num_warps = _prepare_hygon(generator, out, kwargs, "generate")
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
    engine, n, block_size, num_warps = _prepare_hygon(generator, out, kwargs, "generate_uniform")
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
    engine, n, block_size, num_warps = _prepare_hygon(generator, out, kwargs, "generate_uniform_double")
    seed = engine.seed
    offset = engine.offset
    _launch_uniform_f64(out, seed, offset, n, block_size, num_warps)
    engine.offset = offset + 2 * n
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
    engine, n, block_size, num_warps = _prepare_hygon(generator, out, kwargs, "generate_normal", _F32_NORMAL_CONFIG)
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
        True,
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
    engine, n, block_size, num_warps = _prepare_hygon(
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
    engine.offset = offset + 2 * n
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
    engine, n, block_size, num_warps = _prepare_hygon(generator, out, kwargs, "generate_lognormal", _F32_NORMAL_CONFIG)
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
        True,
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
    engine, n, block_size, num_warps = _prepare_hygon(
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
    engine.offset = offset + 2 * n
    return out


def _prepare_hygon(
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
        f"{op_name}: requires a Philox CurandGenerator, a "
        "contiguous CUDA/NPU output, and only block_size/num_warps options."
    )
