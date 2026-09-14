from __future__ import annotations

import torch
import triton
import triton.language as tl

from flagrand.rng.philox import PhiloxGenerator, generate_philox_raw_u32
from flagrand.runtime import CachedKernelLauncher


@triton.jit
def _counter_at(base, index):
    # Widen before adding: an i32 base near 2**31 must carry into the high word.
    return tl.cast(base, tl.uint64) + tl.cast(index, tl.uint64)


@triton.jit
def _philox_generate(seed, counter):
    c0 = tl.zeros_like(counter).to(tl.uint32)
    c1 = tl.zeros_like(counter).to(tl.uint32)
    c = counter.to(tl.uint64)
    c2 = c.to(tl.uint32)
    c3 = (c >> 32).to(tl.uint32)
    return tl.philox(seed, c0, c1, c2, c3)


@triton.jit
def _logical_words(seed, counter):
    r0, r1, r2, r3 = _philox_generate(seed, counter)
    return r0, r2, r1, r3


@triton.jit
def _logical_window4(seed, counter, SHIFT: tl.constexpr):
    a0, a1, a2, a3 = _logical_words(seed, counter)
    if SHIFT == 0:
        return a0, a1, a2, a3
    b0, b1, b2, b3 = _logical_words(seed, counter + 1)
    if SHIFT == 1:
        return a1, a2, a3, b0
    if SHIFT == 2:
        return a2, a3, b0, b1
    return a3, b0, b1, b2


@triton.jit
def _uniform_f32(raw):
    signed = raw.to(tl.int32, bitcast=True)
    value = signed.to(tl.float32)
    unsigned = tl.where(signed < 0, value + 4294967296.0, value)
    uniform = 1.0 - unsigned * 2.3283064365386963e-10
    return tl.maximum(tl.minimum(uniform, 1.0), 1.1920928955078125e-7)


@triton.jit
def _dyadic53_to_f64(value):
    is_one = value == (1 << 53)
    work = tl.where(is_one, (1 << 52), value)
    normalized = work
    exponent = tl.zeros_like(work).to(tl.int32)

    high = normalized >> 32
    take = high != 0
    normalized = tl.where(take, high, normalized)
    exponent += tl.where(take, 32, 0)
    high = normalized >> 16
    take = high != 0
    normalized = tl.where(take, high, normalized)
    exponent += tl.where(take, 16, 0)
    high = normalized >> 8
    take = high != 0
    normalized = tl.where(take, high, normalized)
    exponent += tl.where(take, 8, 0)
    high = normalized >> 4
    take = high != 0
    normalized = tl.where(take, high, normalized)
    exponent += tl.where(take, 4, 0)
    high = normalized >> 2
    take = high != 0
    normalized = tl.where(take, high, normalized)
    exponent += tl.where(take, 2, 0)
    high = normalized >> 1
    take = high != 0
    exponent += tl.where(take, 1, 0)

    mantissa = (work << (52 - exponent)) & ((1 << 52) - 1)
    ieee_exponent = (exponent + 970).to(tl.uint64)
    bits = (ieee_exponent << 52) | mantissa
    bits = tl.where(is_one, 0x3FF0000000000000, bits)
    return bits.to(tl.float64, bitcast=True)


@triton.jit
def _uniform_f64(raw0, raw1):
    hi = raw0.to(tl.uint32, bitcast=True).to(tl.uint64) >> 5
    lo = raw1.to(tl.uint32, bitcast=True).to(tl.uint64) >> 6
    mantissa = (hi << 26) | lo
    return _dyadic53_to_f64((1 << 53) - mantissa)


@triton.jit
def _box_muller_f32(raw0, raw1):
    u0 = _uniform_f32(raw0)
    u1 = _uniform_f32(raw1)
    radius = tl.sqrt(-2.0 * tl.log(u0))
    angle = 6.283185307179586 * u1
    return radius * tl.cos(angle), radius * tl.sin(angle)


@triton.jit
def _uniform_aligned_kernel(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    counter_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter_offset < n_counters
    w0, w1, w2, w3 = _logical_words(seed, _counter_at(base_counter, counter_offset))
    base = counter_offset * 4
    tl.store(out + base + 0, _uniform_f32(w0), mask=mask)
    tl.store(out + base + 1, _uniform_f32(w1), mask=mask)
    tl.store(out + base + 2, _uniform_f32(w2), mask=mask)
    tl.store(out + base + 3, _uniform_f32(w3), mask=mask)


@triton.jit
def _uniform_generic_kernel(out, seed, offset, n, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < n
    word_index = _counter_at(offset, index)
    counter = word_index >> 2
    lane = word_index & 3
    w0, w1, w2, w3 = _logical_words(seed, counter)
    raw = tl.where(lane == 0, w0, tl.where(lane == 1, w1, tl.where(lane == 2, w2, w3)))
    tl.store(out + index, _uniform_f32(raw), mask=mask)


@triton.jit
def _uniform_f64_aligned_kernel(out, seed, base_counter, n_counters, BLOCK: tl.constexpr):
    counter_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter_offset < n_counters
    w0, w1, w2, w3 = _logical_words(seed, _counter_at(base_counter, counter_offset))
    base = counter_offset * 2
    tl.store(out + base + 0, _uniform_f64(w0, w1), mask=mask)
    tl.store(out + base + 1, _uniform_f64(w2, w3), mask=mask)


@triton.jit
def _uniform_f64_generic_kernel(
    out,
    seed,
    base_counter,
    n,
    n_groups,
    SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = group < n_groups
    w0, w1, w2, w3 = _logical_window4(seed, _counter_at(base_counter, group), SHIFT)
    base = group * 2
    tl.store(out + base + 0, _uniform_f64(w0, w1), mask=mask & (base + 0 < n))
    tl.store(out + base + 1, _uniform_f64(w2, w3), mask=mask & (base + 1 < n))


@triton.jit
def _normal_aligned_kernel(
    out,
    seed,
    base_counter,
    n_counters,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    counter_offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = counter_offset < n_counters
    w0, w1, w2, w3 = _logical_words(seed, _counter_at(base_counter, counter_offset))
    z0, z1 = _box_muller_f32(w0, w1)
    z2, z3 = _box_muller_f32(w2, w3)
    y0 = mean + stddev * z0
    y1 = mean + stddev * z1
    y2 = mean + stddev * z2
    y3 = mean + stddev * z3
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
        y2 = tl.exp(y2)
        y3 = tl.exp(y3)
    base = counter_offset * 4
    tl.store(out + base + 0, y0, mask=mask)
    tl.store(out + base + 1, y1, mask=mask)
    tl.store(out + base + 2, y2, mask=mask)
    tl.store(out + base + 3, y3, mask=mask)


@triton.jit
def _normal_pair_kernel(
    out,
    seed,
    offset,
    n_pairs,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair < n_pairs
    word_index = _counter_at(offset, pair * 2)
    counter = word_index >> 2
    lane = word_index & 3
    w0, w1, w2, w3 = _logical_words(seed, counter)
    raw0 = tl.where(lane == 0, w0, w2)
    raw1 = tl.where(lane == 0, w1, w3)
    z0, z1 = _box_muller_f32(raw0, raw1)
    y0 = mean + stddev * z0
    y1 = mean + stddev * z1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
    base = pair * 2
    tl.store(out + base + 0, y0, mask=mask)
    tl.store(out + base + 1, y1, mask=mask)


@triton.jit
def _normal_from_raw_kernel(
    out,
    raw,
    n_pairs,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pair < n_pairs
    base = pair * 2
    raw0 = tl.load(raw + base + 0, mask=mask)
    raw1 = tl.load(raw + base + 1, mask=mask)
    z0, z1 = _box_muller_f32(raw0, raw1)
    y0 = mean + stddev * z0
    y1 = mean + stddev * z1
    if LOGNORMAL:
        y0 = tl.exp(y0)
        y1 = tl.exp(y1)
    tl.store(out + base + 0, y0, mask=mask)
    tl.store(out + base + 1, y1, mask=mask)


_UNIFORM_ALIGNED_LAUNCHER = CachedKernelLauncher(
    _uniform_aligned_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_GENERIC_LAUNCHER = CachedKernelLauncher(
    _uniform_generic_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_F64_ALIGNED_LAUNCHER = CachedKernelLauncher(
    _uniform_f64_aligned_kernel,
    constexpr_names=("BLOCK",),
)
_UNIFORM_F64_GENERIC_LAUNCHER = CachedKernelLauncher(
    _uniform_f64_generic_kernel,
    constexpr_names=("SHIFT", "BLOCK"),
)
_NORMAL_ALIGNED_LAUNCHER = CachedKernelLauncher(
    _normal_aligned_kernel,
    constexpr_names=("LOGNORMAL", "BLOCK"),
)
_NORMAL_PAIR_LAUNCHER = CachedKernelLauncher(
    _normal_pair_kernel,
    constexpr_names=("LOGNORMAL", "BLOCK"),
)
_NORMAL_FROM_RAW_LAUNCHER = CachedKernelLauncher(
    _normal_from_raw_kernel,
    constexpr_names=("LOGNORMAL", "BLOCK"),
)


def generate_corex_raw_u32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    generate_philox_raw_u32(
        out,
        generator,
        block_size=block_size,
        num_warps=num_warps,
        scalar_alignment=4,
    )


def generate_corex_uniform_f32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    n = out.numel()
    seed = int(generator.seed)
    offset = int(generator.offset)
    if offset % 4 == 0 and n % 4 == 0:
        n_counters = n // 4
        _UNIFORM_ALIGNED_LAUNCHER.launch(
            triton.cdiv(n_counters, block_size),
            (out, seed, offset // 4, n_counters),
            (block_size,),
            (num_warps,),
        )
    else:
        _UNIFORM_GENERIC_LAUNCHER.launch(
            triton.cdiv(n, block_size),
            (out, seed, offset, n),
            (block_size,),
            (num_warps,),
        )
    generator.offset = offset + n


def generate_corex_uniform_f64(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    block_size: int,
    num_warps: int,
) -> None:
    n = out.numel()
    seed = int(generator.seed)
    offset = int(generator.offset)
    if offset % 4 == 0 and n % 2 == 0:
        n_counters = n // 2
        _UNIFORM_F64_ALIGNED_LAUNCHER.launch(
            triton.cdiv(n_counters, block_size),
            (out, seed, offset // 4, n_counters),
            (block_size,),
            (num_warps,),
        )
    else:
        n_groups = triton.cdiv(n, 2)
        _UNIFORM_F64_GENERIC_LAUNCHER.launch(
            triton.cdiv(n_groups, block_size),
            (out, seed, offset // 4, n, n_groups),
            (offset % 4, block_size),
            (num_warps,),
        )
    generator.offset = offset + 2 * n


def generate_corex_normal_f32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_corex_normal_like_f32(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=False,
    )


def generate_corex_lognormal_f32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
) -> None:
    _generate_corex_normal_like_f32(
        out,
        generator,
        mean=mean,
        stddev=stddev,
        block_size=block_size,
        num_warps=num_warps,
        lognormal=True,
    )


def _generate_corex_normal_like_f32(
    out: torch.Tensor,
    generator: PhiloxGenerator,
    *,
    mean: float,
    stddev: float,
    block_size: int,
    num_warps: int,
    lognormal: bool,
) -> None:
    n = out.numel()
    if n % 2 != 0:
        raise ValueError(f"CoreX Philox Box-Muller path requires an even element count, got {n}.")
    seed = int(generator.seed)
    offset = int(generator.offset)
    if offset % 4 == 0 and n % 4 == 0:
        n_counters = n // 4
        _NORMAL_ALIGNED_LAUNCHER.launch(
            triton.cdiv(n_counters, block_size),
            (out, seed, offset // 4, n_counters, float(mean), float(stddev)),
            (lognormal, block_size),
            (num_warps,),
        )
    elif offset % 2 == 0:
        n_pairs = n // 2
        _NORMAL_PAIR_LAUNCHER.launch(
            triton.cdiv(n_pairs, block_size),
            (out, seed, offset, n_pairs, float(mean), float(stddev)),
            (lognormal, block_size),
            (num_warps,),
        )
    else:
        raw = torch.empty(n, device=out.device, dtype=torch.int32)
        raw_generator = PhiloxGenerator(seed=seed, offset=offset)
        generate_philox_raw_u32(
            raw,
            raw_generator,
            block_size=block_size,
            num_warps=num_warps,
        )
        n_pairs = n // 2
        _NORMAL_FROM_RAW_LAUNCHER.launch(
            triton.cdiv(n_pairs, block_size),
            (out, raw, n_pairs, float(mean), float(stddev)),
            (lognormal, block_size),
            (num_warps,),
        )
    generator.offset = offset + n
