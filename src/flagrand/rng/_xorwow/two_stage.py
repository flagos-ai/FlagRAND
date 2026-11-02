"""Separate XORWOW recurrence and binary64 transforms for large complete rows."""

from functools import lru_cache

import torch
import triton.language as tl

from ...runtime.compiled_launcher import CachedKernelLauncher
from ...runtime.specialization import launch_specialization_key
from ._binding import bind_math_kernel
from .api import _float_bits, _math_module, generate
from .kernels import _uniform64
from .metax_math import sqrt_radial64
from .native_math import sincos_turns
from .range_math64 import exp64


def supports_two_stage(backend, n, nstreams, distribution, dtype, config):
    """Select complete Box--Muller rows on the measured backend math paths."""
    if backend == "thead":
        return (
            distribution in ("normal", "lognormal")
            and dtype == torch.float64
            and 2**24 <= n < 2**30
            and nstreams == 32768
            and n % (2 * nstreams) == 0
            and config.get("math_mode") == "reduced"
        )
    math_mode = "range_exp_rsqrt64" if distribution == "lognormal" else "reduced_rsqrt64"
    return (
        backend == "metax"
        and distribution in ("normal", "lognormal")
        and dtype == torch.float64
        and 2**20 <= n < 2**30
        and nstreams % 128 == 0
        and n % (2 * nstreams) == 0
        and config.get("math_mode") == math_mode
    )


def _build_transform(device_math):
    def transform_kernel(
        RAW,
        OUT,
        N,
        MU_BITS,
        SIGMA_BITS,
        NSTREAMS: tl.constexpr,
        BLOCK: tl.constexpr,
        LOGNORMAL: tl.constexpr,
        PPU: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        stream = pair % NSTREAMS
        row = pair // NSTREAMS
        raw_base = row * (4 * NSTREAMS) + stream
        a = tl.load(RAW + raw_base).to(tl.uint32, bitcast=True)
        b = tl.load(RAW + raw_base + NSTREAMS).to(tl.uint32, bitcast=True)
        c = tl.load(RAW + raw_base + 2 * NSTREAMS).to(tl.uint32, bitcast=True)
        e = tl.load(RAW + raw_base + 3 * NSTREAMS).to(tl.uint32, bitcast=True)
        u1 = _uniform64(a, b, True)
        u2 = _uniform64(c, e, False)
        if PPU:
            radius = device_math.sqrt(-2.0 * device_math.log(u1))
        else:
            radius = sqrt_radial64(-2.0 * device_math.log(u1))
        sine, cosine = sincos_turns(u2, True)
        mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
        sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
        value0 = mu + sigma * (radius * cosine)
        value1 = mu + sigma * (radius * sine)
        if LOGNORMAL:
            if PPU:
                value0 = device_math.exp(value0)
                value1 = device_math.exp(value1)
            else:
                value0 = exp64(value0)
                value1 = exp64(value1)
        output_base = row * (2 * NSTREAMS) + stream
        tl.store(OUT + output_base, value0)
        tl.store(OUT + output_base + NSTREAMS, value1)

    return bind_math_kernel(transform_kernel, device_math)


@lru_cache(maxsize=2)
def _transform_launcher(backend):
    kernel = _build_transform(_math_module(backend))
    launcher = CachedKernelLauncher(
        kernel,
        constexpr_names=("NSTREAMS", "BLOCK", "LOGNORMAL", "PPU"),
        option_names=("num_warps", "num_stages", "enable_fp_fusion"),
    )
    return kernel, launcher


def generate_two_stage(state, out, workspace, distribution, mu, sigma, *, backend):
    """Preserve draw-major streams while exposing each output pair independently."""
    n = out.numel()
    generate(
        state,
        2 * n,
        distribution="raw32",
        out=workspace,
        backend=backend,
        block=64 if backend == "thead" else 128,
        num_warps=2,
        rounds=8,
        num_stages=1,
        aligned=True,
        index32=True,
        math_mode="native",
    )
    kernel, launcher = _transform_launcher(backend)
    arguments = (workspace, out, n, _float_bits(mu, True), _float_bits(sigma, True))
    constants = (state.shape[1], 128, distribution == "lognormal", backend == "thead")
    options = (4 if backend == "thead" else 2, 1, False)
    key = launch_specialization_key(arguments, 2, constants, options, kernel.debug)
    launcher.launch((n // 256,), arguments, constants, options, specialization_key=key)
    return out
