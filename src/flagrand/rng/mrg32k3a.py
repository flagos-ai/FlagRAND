"""Persistent MRG32k3a streams with fused continuous distributions.

Raw outputs are residues in [1, 4294967087], not a full uint32 lattice.
The recurrence follows L'Ecuyer (1999); seed mapping is FlagRand-specific.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

from ..runtime.compiled_launcher import CachedKernelLauncher, PreparedKernelLaunch
from ..runtime.specialization import integer_specialization_key, launch_specialization_key
from ._mrg32k3a_iluvatar_host import capture, generate_hot
from ._mrg32k3a_iluvatar_math import sincos_bounded
from ._mrg32k3a_math import M1, initial_states
from ._xorwow._binding import bind_kernel
from ._xorwow.api import _math_module, resolve_backend
from ._xorwow.native_math import sincos_turns


@lru_cache(maxsize=8)
def _seed_tensor(seed, nstreams, offset):
    return torch.tensor(initial_states(seed, nstreams, offset), dtype=torch.int32, device="cpu")


@triton.jit
def _mrg32k3a_step(x0, x1, x2, y0, y1, y2):
    # Both products and their difference fit signed int64. Use a positive
    # dividend so backend integer-remainder conventions cannot change mod.
    p1 = 1403580 * x1 + 810728 * (4294967087 - x0)
    p2 = 527612 * y2 + 1370589 * (4294944443 - y0)
    # M1=2**32-209 and M2=2**32-22853. Fold the high word using
    # 2**32 == c (mod M). For these recurrence bounds one/two folds leave
    # values below 2*M1/2*M2, so one conditional subtraction is sufficient.
    x = (p1 & 0xFFFFFFFF) + 209 * (p1 >> 32)
    x = tl.where(x >= 4294967087, x - 4294967087, x)
    y = (p2 & 0xFFFFFFFF) + 22853 * (p2 >> 32)
    y = (y & 0xFFFFFFFF) + 22853 * (y >> 32)
    y = tl.where(y >= 4294944443, y - 4294944443, y)
    difference = x - y
    z = tl.where(difference <= 0, difference + 4294967087, difference)
    return z, x1, x2, x, y1, y2, y


@triton.jit
def _mrg32k3a_step_u32(x0, x1, x2, y0, y1, y2):
    # Canonical state and M-state fit uint32. Widen before multiplication;
    # products and high-word folds still need int64.
    cx = tl.full((), 4294967087, tl.uint32) - x0
    cy = tl.full((), 4294944443, tl.uint32) - y0
    p1 = 1403580 * x1.to(tl.int64) + 810728 * cx.to(tl.int64)
    p2 = 527612 * y2.to(tl.int64) + 1370589 * cy.to(tl.int64)
    x = (p1 & 0xFFFFFFFF) + 209 * (p1 >> 32)
    x = tl.where(x >= 4294967087, x - 4294967087, x).to(tl.uint32)
    y = (p2 & 0xFFFFFFFF) + 22853 * (p2 >> 32)
    y = (y & 0xFFFFFFFF) + 22853 * (y >> 32)
    y = tl.where(y >= 4294944443, y - 4294944443, y).to(tl.uint32)
    difference = x.to(tl.int64) - y.to(tl.int64)
    z = tl.where(difference <= 0, difference + 4294967087, difference)
    return z, x1, x2, x, y1, y2, y


@triton.jit
def _mrg32k3a_step_typed(x0, x1, x2, y0, y1, y2, STATE_U32: tl.constexpr):
    if STATE_U32:
        return _mrg32k3a_step_u32(x0, x1, x2, y0, y1, y2)
    else:
        return _mrg32k3a_step(x0, x1, x2, y0, y1, y2)


@triton.jit
def _mrg32k3a_uniform64_digits(a, b, RECIPROCAL: tl.constexpr):
    if RECIPROCAL:
        inverse = tl.full((), 1.0 / 4294967087.0, tl.float64)
        return ((a - 1).to(tl.float64) + b.to(tl.float64) * inverse) * inverse
    modulus = tl.full((), 4294967087.0, tl.float64)
    return ((a - 1).to(tl.float64) + b.to(tl.float64) / modulus) / modulus


@triton.jit
def _mrg32k3a_residue_u32(x, y):
    # Canonical components satisfy 0 <= x < M1 and 0 <= y < M2 < M1.
    # uint32 wrap gives x-y+M1 when x <= y, including M1 when x == y.
    difference = x - y
    modulus = tl.full((), 4294967087, tl.uint32)
    return tl.where(x > y, difference, difference + modulus)


@triton.jit
def _mrg32k3a_uniform64_u32_residues(x0, x1, x2, y0, y1, y2, RECIPROCAL: tl.constexpr):
    _, a, b, c, d, e, f = _mrg32k3a_step_u32(x0, x1, x2, y0, y1, y2)
    first = _mrg32k3a_residue_u32(c, f)
    _, a, b, c, d, e, f = _mrg32k3a_step_u32(a, b, c, d, e, f)
    second = _mrg32k3a_residue_u32(c, f)
    # Keep both base-M1 digits and the original binary64 evaluation order.
    value = _mrg32k3a_uniform64_digits(first, second, RECIPROCAL)
    return value, a, b, c, d, e, f


@triton.jit
def _mrg32k3a_uniform(
    x0,
    x1,
    x2,
    y0,
    y1,
    y2,
    F64: tl.constexpr,
    RECIPROCAL: tl.constexpr = False,
    STATE_U32: tl.constexpr = False,
):
    a, x0, x1, x2, y0, y1, y2 = _mrg32k3a_step_typed(x0, x1, x2, y0, y1, y2, STATE_U32)
    if F64:
        b, x0, x1, x2, y0, y1, y2 = _mrg32k3a_step_typed(x0, x1, x2, y0, y1, y2, STATE_U32)
        # Two base-M1 digits. Values occupy (0, 1], with binary64 rounding;
        # casting a single raw word to double would leave ~32 random bits.
        u = _mrg32k3a_uniform64_digits(a, b, RECIPROCAL)
    else:
        u = a.to(tl.float32) * (1.0 / 4294967087.0)
    return u, x0, x1, x2, y0, y1, y2


def _build_kernel(device_math):
    def generate_kernel(
        STATE,
        OUT,
        N,
        CURSOR,
        MEAN_BITS,
        STD_BITS,
        NSTREAMS: tl.constexpr,
        BLOCK: tl.constexpr,
        DISTRIBUTION: tl.constexpr,
        F64: tl.constexpr,
        REDUCED_TRIG: tl.constexpr,
        COMPACT: tl.constexpr = False,
        RECIPROCAL: tl.constexpr = False,
        STATE_U32: tl.constexpr = False,
    ):
        slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if COMPACT:
            stream = (CURSOR + slot) % NSTREAMS
            present = (slot < NSTREAMS) & (slot < N)
            first = slot
        else:
            stream = slot
            present = stream < NSTREAMS
            first = (stream + NSTREAMS - CURSOR) % NSTREAMS
        x0 = tl.load(STATE + stream, present, other=1).to(tl.uint32)
        x1 = tl.load(STATE + NSTREAMS + stream, present, other=1).to(tl.uint32)
        x2 = tl.load(STATE + 2 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y0 = tl.load(STATE + 3 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y1 = tl.load(STATE + 4 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y2 = tl.load(STATE + 5 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        if not STATE_U32:
            x0, x1, x2 = x0.to(tl.int64), x1.to(tl.int64), x2.to(tl.int64)
            y0, y1, y2 = y0.to(tl.int64), y1.to(tl.int64), y2.to(tl.int64)
        if DISTRIBUTION >= 2:
            if F64:
                mean = MEAN_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                std = STD_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
            else:
                mean = MEAN_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                std = STD_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
        for batch in range(tl.cdiv(N, NSTREAMS)):
            index = first + batch * NSTREAMS
            active = present & (index < N)
            if DISTRIBUTION == 0:
                value, a, b, c, d, e, f = _mrg32k3a_step_typed(x0, x1, x2, y0, y1, y2, STATE_U32)
                value = value.to(tl.uint32)
            else:
                if (STATE_U32 and F64) and DISTRIBUTION == 1:
                    u1, a, b, c, d, e, f = _mrg32k3a_uniform64_u32_residues(x0, x1, x2, y0, y1, y2, RECIPROCAL)
                else:
                    u1, a, b, c, d, e, f = _mrg32k3a_uniform(x0, x1, x2, y0, y1, y2, F64, RECIPROCAL, STATE_U32)
                value = u1
                if DISTRIBUTION >= 2:
                    u2, a, b, c, d, e, f = _mrg32k3a_uniform(a, b, c, d, e, f, F64, RECIPROCAL, STATE_U32)
                    radius = device_math.sqrt(-2.0 * device_math.log(u1))
                    if F64 and REDUCED_TRIG:
                        # u2 is measured in turns and lies in (0, 1].
                        # This preserves raw draws/state, but floating results
                        # may differ from cos of an already-rounded 2*pi*u2.
                        _, cosine = sincos_turns(u2, True)
                    else:
                        if F64:
                            angle = u2 * tl.full((), 6.2831853071795864769, tl.float64)
                        else:
                            angle = u2 * 6.2831853071795864769
                        cosine = device_math.cos(angle)
                    normal = radius * cosine
                    value = mean + std * normal
                    if DISTRIBUTION == 3:
                        value = device_math.exp(value)
            tl.store(OUT + index, value, active)
            x0 = tl.where(active, a, x0)
            x1 = tl.where(active, b, x1)
            x2 = tl.where(active, c, x2)
            y0 = tl.where(active, d, y0)
            y1 = tl.where(active, e, y1)
            y2 = tl.where(active, f, y2)
        tl.store(STATE + stream, x0.to(tl.uint32), present)
        tl.store(STATE + NSTREAMS + stream, x1.to(tl.uint32), present)
        tl.store(STATE + 2 * NSTREAMS + stream, x2.to(tl.uint32), present)
        tl.store(STATE + 3 * NSTREAMS + stream, y0.to(tl.uint32), present)
        tl.store(STATE + 4 * NSTREAMS + stream, y1.to(tl.uint32), present)
        tl.store(STATE + 5 * NSTREAMS + stream, y2.to(tl.uint32), present)

    return bind_kernel(generate_kernel, do_not_specialize=["N", "CURSOR", "MEAN_BITS", "STD_BITS"])


@lru_cache(None)
def _kernel(backend, transformed):
    return _build_kernel(_math_module(backend) if transformed else tl)


def _build_paired_kernel(device_math):
    def paired_kernel(
        STATE,
        OUT,
        SPARE,
        N,
        CURSOR,
        SPARE_INDEX,
        MEAN_BITS,
        STD_BITS,
        NSTREAMS: tl.constexpr,
        BLOCK: tl.constexpr,
        F64: tl.constexpr,
        LOGNORMAL: tl.constexpr,
        HAS_SPARE: tl.constexpr,
        COMPACT: tl.constexpr = False,
        RECIPROCAL: tl.constexpr = False,
        STATE_U32: tl.constexpr = False,
        JOINT_SINCOS: tl.constexpr = False,
    ):
        slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        pairs = tl.cdiv(N - HAS_SPARE, 2)
        if COMPACT:
            stream = (CURSOR + slot) % NSTREAMS
            present = (slot < NSTREAMS) & (slot < pairs)
            first = slot
        else:
            stream = slot
            present = stream < NSTREAMS
            first = (stream + NSTREAMS - CURSOR) % NSTREAMS
        cache_lane = tl.zeros((BLOCK,), tl.int32)
        if F64:
            mean = MEAN_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
            std = STD_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
        else:
            mean = MEAN_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
            std = STD_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
        if HAS_SPARE:
            z = tl.load(SPARE + SPARE_INDEX + cache_lane, slot == 0, other=0.0)
            value = tl.fma(std, z, mean)
            if LOGNORMAL:
                value = device_math.exp(value)
            tl.store(OUT + cache_lane, value, slot == 0)

        x0 = tl.load(STATE + stream, present, other=1).to(tl.uint32)
        x1 = tl.load(STATE + NSTREAMS + stream, present, other=1).to(tl.uint32)
        x2 = tl.load(STATE + 2 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y0 = tl.load(STATE + 3 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y1 = tl.load(STATE + 4 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        y2 = tl.load(STATE + 5 * NSTREAMS + stream, present, other=1).to(tl.uint32)
        if not STATE_U32:
            x0, x1, x2 = x0.to(tl.int64), x1.to(tl.int64), x2.to(tl.int64)
            y0, y1, y2 = y0.to(tl.int64), y1.to(tl.int64), y2.to(tl.int64)
        for batch in range(tl.cdiv(pairs, NSTREAMS)):
            pair = first + batch * NSTREAMS
            active = present & (pair < pairs)
            u1, a, b, c, d, e, f = _mrg32k3a_uniform(x0, x1, x2, y0, y1, y2, F64, RECIPROCAL, STATE_U32)
            u2, a, b, c, d, e, f = _mrg32k3a_uniform(a, b, c, d, e, f, F64, RECIPROCAL, STATE_U32)
            radius = device_math.sqrt(-2.0 * device_math.log(u1))
            if F64:
                sine, cosine = sincos_turns(u2, True)
            else:
                angle = u2 * 6.2831853071795864769
                if JOINT_SINCOS:
                    sine, cosine = sincos_bounded(angle)
                else:
                    sine = device_math.sin(angle)
                    cosine = device_math.cos(angle)
            # Round the standard values before applying parameters, so a
            # cached tail and an immediately consumed value use the same bits.
            z0 = tl.fma(radius, cosine, 0.0)
            z1 = tl.fma(radius, sine, 0.0)
            index = 2 * pair + HAS_SPARE
            v0 = tl.fma(std, z0, mean)
            v1 = tl.fma(std, z1, mean)
            if LOGNORMAL:
                v0 = device_math.exp(v0)
                v1 = device_math.exp(v1)
            tl.store(OUT + index, v0, active)
            tl.store(OUT + index + 1, v1, active & (index + 1 < N))
            # Two physical slots avoid a read/write race between the CTA
            # consuming the old spare and the CTA producing the new spare.
            tl.store(SPARE + (1 - SPARE_INDEX) + cache_lane, z1, active & (index + 1 == N))
            x0 = tl.where(active, a, x0)
            x1 = tl.where(active, b, x1)
            x2 = tl.where(active, c, x2)
            y0 = tl.where(active, d, y0)
            y1 = tl.where(active, e, y1)
            y2 = tl.where(active, f, y2)
        tl.store(STATE + stream, x0.to(tl.uint32), present)
        tl.store(STATE + NSTREAMS + stream, x1.to(tl.uint32), present)
        tl.store(STATE + 2 * NSTREAMS + stream, x2.to(tl.uint32), present)
        tl.store(STATE + 3 * NSTREAMS + stream, y0.to(tl.uint32), present)
        tl.store(STATE + 4 * NSTREAMS + stream, y1.to(tl.uint32), present)
        tl.store(STATE + 5 * NSTREAMS + stream, y2.to(tl.uint32), present)

    return bind_kernel(
        paired_kernel,
        do_not_specialize=[
            "N",
            "CURSOR",
            "SPARE_INDEX",
            "MEAN_BITS",
            "STD_BITS",
        ],
    )


@lru_cache(None)
def _paired_kernel(backend):
    return _build_paired_kernel(_math_module(backend))


@lru_cache(None)
def _launcher(backend, paired):
    if paired:
        return CachedKernelLauncher(
            _paired_kernel(backend),
            constexpr_names=(
                "NSTREAMS",
                "BLOCK",
                "F64",
                "LOGNORMAL",
                "HAS_SPARE",
                "COMPACT",
                "RECIPROCAL",
                "STATE_U32",
                "JOINT_SINCOS",
            ),
            option_names=("num_warps",),
        )
    return CachedKernelLauncher(
        _kernel(backend, False),
        constexpr_names=(
            "NSTREAMS",
            "BLOCK",
            "DISTRIBUTION",
            "F64",
            "REDUCED_TRIG",
            "COMPACT",
            "RECIPROCAL",
            "STATE_U32",
        ),
        option_names=("num_warps",),
    )


def _float_bits(value, f64):
    return struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", value))[0]


def _launch_spec(backend, n, nstreams, distribution, f64, has_spare, mean, stddev, options):
    """Construct the shared shape, parameter and constexpr portion of a call."""
    block = int(options.get("block_size", 128))
    reciprocal = bool(options.get("uniform_reciprocal", True))
    corex_f32 = backend == "iluvatar" and not f64
    corex_small = False
    if corex_f32:
        active_streams = (n - int(has_spare) + 1) // 2 if distribution >= 2 else n
        corex_small = active_streams < nstreams
    state_u32 = backend == "thead"
    if backend == "iluvatar":
        state_u32 = bool(options.get("state_u32", corex_f32 and not corex_small))
    default_warps = 2 if backend in ("metax", "hygon") else 4
    if corex_f32:
        default_warps = 4 if corex_small else 2
    warps = int(options.get("num_warps", default_warps))
    if distribution >= 2:
        pairs = (n - int(has_spare) + 1) // 2
        compact = bool(
            options.get("compact_streams", ((f64 or backend == "thead") and pairs < nstreams) or corex_small)
        )
        work = min(pairs, nstreams) if compact else nstreams
        grid = (max(1, triton.cdiv(work, block)),)
        constants = (nstreams, block, f64, distribution == 3, has_spare, compact, reciprocal, state_u32, corex_f32)
    else:
        compact = bool(options.get("compact_streams", (backend == "thead" and n < nstreams) or corex_small))
        work = min(n, nstreams) if compact else nstreams
        grid = (triton.cdiv(work, block),)
        constants = (nstreams, block, distribution, f64, True, compact, reciprocal, state_u32)
    return grid, (_float_bits(mean, f64), _float_bits(stddev, f64)), constants, (warps,)


@dataclass
class _PreparedMrgLaunch:
    launcher: CachedKernelLauncher
    grid: tuple
    n: int
    parameters: tuple
    constants: tuple
    options: tuple
    paired: bool
    bound_launcher: PreparedKernelLaunch | None = None
    _keys: dict = field(default_factory=dict)
    _bindings: dict = field(default_factory=dict)

    def launch(self, state, out, cursor, spare=None, spare_index=0):
        if self.paired:
            tensors = (state, out, spare)
            scalars = (self.n, cursor, spare_index) + self.parameters
            dynamic = (integer_specialization_key(cursor), integer_specialization_key(spare_index))
        else:
            tensors = (state, out)
            scalars = (self.n, cursor) + self.parameters
            dynamic = (integer_specialization_key(cursor),)
        arguments = tensors + scalars
        signatures = ((state.dtype, state.data_ptr() % 16 == 0), (out.dtype, out.data_ptr() % 16 == 0))
        if self.paired:
            signatures += ((spare.dtype, spare.data_ptr() % 16 == 0),)
        debug = self.launcher._kernel.debug
        signature = (driver.active.get_current_device(), signatures, dynamic, debug)
        key = self._keys.get(signature)
        if key is None:
            key = launch_specialization_key(
                arguments, len(tensors), self.constants, self.options, debug, tensor_signatures=signatures
            )
            self._keys[signature] = key
        # Cursor, spare index and pointers remain runtime arguments. The
        # existing launcher resolves each call's stream and runtime modes.
        if self.bound_launcher is not None:
            # Cursor and spare-index specializations recur across calls. Keep
            # their prepared entries so switching signatures does not rebind.
            binding = self._bindings.get(signature)
            if binding is None:
                binding = self.launcher.prepare(self.grid)
                self._bindings[signature] = binding
            return binding.launch(arguments, self.constants, self.options, specialization_key=key)
        return self.launcher.launch(self.grid, arguments, self.constants, self.options, specialization_key=key)


@lru_cache(maxsize=128)
def _prepared_launch(backend, device, n, nstreams, distribution, f64, has_spare, mean, mean_sign, stddev, stddev_sign):
    # Device and signed-zero tokens belong to the cache key. Parameters are
    # packed only on a miss; generator state is never captured by the plan.
    grid, parameters, constants, options = _launch_spec(
        backend, n, nstreams, distribution, f64, has_spare, mean, stddev, {}
    )
    paired = distribution >= 2
    launcher = _launcher(backend, paired)
    return _PreparedMrgLaunch(
        launcher,
        grid,
        n,
        parameters,
        constants,
        options,
        paired,
        bound_launcher=launcher.prepare(grid) if backend == "thead" else None,
    )


@dataclass
class Mrg32k3aGenerator:
    """131072 fixed independent streams separated by 2**76 raw steps.

    Stream count is part of the output-sequence configuration: the default
    131072-stream interleaving differs from the earlier 4096/32768/65536 layouts.
    It stays fixed across output shapes and calls. The six-word state occupies
    3 MiB; explicit ``nstreams=65536`` retains the earlier 1.5 MiB layout.

    FP64 uniforms use two base-M1 digits and a binary64 reciprocal constant.
    ``uniform_reciprocal=False`` retains the division reference path; it can
    differ in the last bits while consuming identical integer state.

    Initial/reset offset skips raw outputs in the round-robin interleaving.
    Raw/uniform advance cursor once per output. Normal/lognormal generate a
    cosine/sine pair on each stream and advance cursor once per new pair.
    Same-distribution split calls preserve this order, including odd sizes.

    ``offset`` counts consumed raw words after reset: raw/f32 uniform use 1,
    f64 uniform uses 2, each f32 normal pair uses 2, and each f64 pair uses 4.
    Each precision has one pending standard normal after an odd tail. It
    survives raw/uniform/Poisson and other-precision calls; the next normal
    or lognormal of that precision applies its own mean/stddev/exp to it
    without further raw consumption. Reset/state reinitialization clears it.
    After transformed/mixed calls, seed+offset alone is not a state snapshot;
    continuation also needs cursor, pending values and their validity flags.
    Graph replay does not update Python cursor/cache flags. Raw/uniform sizes
    divisible by nstreams, or normal sizes divisible by 2*nstreams with no
    pending value, preserve their interleaving during replay.
    """

    raw_range: ClassVar[int] = M1
    seed: int = 0
    offset: int = 0
    nstreams: int = 131072
    _state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _corex_recent_plan: object | None = field(default=None, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _state_seed: int | None = field(default=None, init=False, repr=False)
    _state_offset: int | None = field(default=None, init=False, repr=False)
    _normal_cache: dict = field(default_factory=dict, init=False, repr=False)
    _normal_has_spare: dict = field(default_factory=dict, init=False, repr=False)
    _normal_spare_index: dict = field(default_factory=dict, init=False, repr=False)
    _backend_device: torch.device | None = field(default=None, init=False, repr=False)
    _auto_backend: str | None = field(default=None, init=False, repr=False)

    @property
    def dimensions(self):
        return None

    @property
    def state(self):
        return self._state

    def reset(self, seed=None, offset=0):
        if seed is not None:
            self.seed = int(seed)
        self.offset = int(offset)
        self._state = None
        self._state_seed = None
        self._state_offset = None
        self._cursor = self.offset % self.nstreams
        self._clear_normal_cache()

    def _clear_normal_cache(self):
        self._corex_recent_plan = None
        self._normal_cache.clear()
        self._normal_has_spare.clear()
        self._normal_spare_index.clear()

    def ensure_state(self, device, **kwargs):
        if self._state is None or self.seed != self._state_seed or self.offset != self._state_offset:
            self._state = _seed_tensor(self.seed, self.nstreams, self.offset).to(device=device, copy=True)
            self._cursor = self.offset % self.nstreams
            self._state_seed = self.seed
            self._state_offset = self.offset
            self._clear_normal_cache()
        elif self._state.device != torch.device(device):
            self._state = self._state.to(device=device)
            self._normal_cache = {key: value.to(device=device) for key, value in self._normal_cache.items()}
        return self._state

    def generate(self, out, **kwargs):
        return self._generate(out, 0, **kwargs)

    def generate_uniform(self, out, **kwargs):
        return self._generate(out, 1, **kwargs)

    def generate_normal(self, out, mean=0.0, stddev=1.0, lognormal=False, **kwargs):
        return self._generate(out, 3 if lognormal else 2, mean=mean, stddev=stddev, **kwargs)

    def _resolve_backend(self, device, backend, cache):
        if backend != "auto" or not cache:
            return resolve_backend(backend, device)
        if self._backend_device != device:
            self._auto_backend = resolve_backend(backend, device)
            self._backend_device = device
        return self._auto_backend

    _generate = generate_hot

    def _generate_slow(self, out, distribution, *, seed=None, offset=None, mean=0.0, stddev=1.0, **kwargs):
        if seed is not None or offset is not None:
            temporary = Mrg32k3aGenerator(
                self.seed if seed is None else int(seed), self.offset if offset is None else int(offset), self.nstreams
            )
            return temporary._generate(out, distribution, mean=mean, stddev=stddev, **kwargs)
        n = out.numel()
        if n == 0:
            return out
        device = out.device
        state = self.ensure_state(device)
        backend = self._resolve_backend(device, str(kwargs.get("backend", "auto")), kwargs.get("cache_backend", True))
        f64 = out.dtype == torch.float64
        paired = distribution >= 2
        has_spare = False
        spare_index = 0
        spare = None
        if paired:
            if f64 not in self._normal_cache:
                self._normal_cache[f64] = torch.empty(2, dtype=out.dtype, device=device)
            spare = self._normal_cache[f64]
            has_spare = self._normal_has_spare.get(f64, False)
            spare_index = self._normal_spare_index.get(f64, 0)
            pairs = (n - int(has_spare) + 1) // 2

        if (backend == "thead" or (backend == "iluvatar" and not f64)) and not kwargs:
            prepared = _prepared_launch(
                backend,
                device,
                n,
                self.nstreams,
                distribution,
                f64,
                has_spare,
                mean,
                math.copysign(1.0, mean),
                stddev,
                math.copysign(1.0, stddev),
            )
            compiled = prepared.launch(state, out, self._cursor, spare, spare_index)
            capture(self, out, distribution, mean, stddev, backend, prepared, compiled, has_spare)
        else:
            grid, parameters, constants, options = _launch_spec(
                backend, n, self.nstreams, distribution, f64, has_spare, mean, stddev, kwargs
            )
            if paired:
                arguments = (state, out, spare, n, self._cursor, spare_index) + parameters
            else:
                arguments = (state, out, n, self._cursor) + parameters
            launch = _launcher(backend, paired)
            if kwargs.get("cached_launch", True):
                key = (
                    launch_specialization_key(arguments, 3 if paired else 2, constants, options, launch._kernel.debug)
                    if kwargs.get("keyed_launch", True)
                    else None
                )
                launch.launch(grid, arguments, constants, options, specialization_key=key)
            else:
                constexpr = dict(zip(launch._constexpr_names, constants))
                launch._kernel[grid](*arguments, **constexpr, num_warps=options[0])
        if paired:
            has_new_spare = (n - int(has_spare)) % 2 != 0
            self._normal_has_spare[f64] = has_new_spare
            if has_new_spare:
                self._normal_spare_index[f64] = 1 - spare_index
            self.offset += pairs * (4 if f64 else 2)
            self._state_offset = self.offset
            self._cursor = (self._cursor + pairs) % self.nstreams
            return out
        consumed = (2 if f64 else 1) * (2 if distribution >= 2 else 1) if distribution else 1
        self.offset += n * consumed
        self._state_offset = self.offset
        self._cursor = (self._cursor + n) % self.nstreams
        return out
