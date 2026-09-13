from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice

from flagrand._curand_philox_fast_path import _prepare
from flagrand._curand_philox_fast_path import try_generate_normal as _portable_normal
from flagrand.fused._internal.philox_direct_f64 import _philox_launch_args_f64
from flagrand.fused._internal.transforms import _sincos_poly_f64, uint32_pair_to_uniform64_curand_compat
from flagrand.rng.philox import _philox_window4
from flagrand.runtime import CachedKernelLauncher


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
    mean = tl.cast(mean_bits, tl.uint64).to(tl.float64, bitcast=True)
    stddev = tl.cast(stddev_bits, tl.uint64).to(tl.float64, bitcast=True)
    y0 = mean + stddev * radius * cosine
    y1 = mean + stddev * radius * sine
    if LOGNORMAL:
        y0 = libdevice.exp(y0)
        y1 = libdevice.exp(y1)
    offsets = group[:, None] * 2 + tl.arange(0, 2)[None, :]
    tl.store(OUT + offsets, tl.join(y0, y1), offsets < n)


_NORMAL = CachedKernelLauncher(_normal_f64, constexpr_names=("SHIFT", "LOGNORMAL", "BLOCK"))


def try_generate_normal(generator, out, *, mean, stddev, kwargs, lognormal):
    if out.dtype != torch.float64:
        return _portable_normal(generator, out, mean=mean, stddev=stddev, kwargs=kwargs, lognormal=lognormal)
    prepared = _prepare(generator, out, kwargs, default_block_size=128)
    if prepared is None:
        return False
    engine, block, warps, _ = prepared
    seed, offset = _philox_launch_args_f64(out, engine, "Hygon Philox float64", require_even=True)
    n = out.numel()
    if n:
        mean_bits = struct.unpack("<Q", struct.pack("<d", mean))[0]
        stddev_bits = struct.unpack("<Q", struct.pack("<d", stddev))[0]
        _NORMAL.launch(
            triton.cdiv(n // 2, block),
            (out, seed, offset // 4, n, mean_bits, stddev_bits),
            (offset % 4, lognormal, block),
            (warps,),
        )
    engine.offset = offset + 2 * n
    return True
