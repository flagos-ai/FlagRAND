from __future__ import annotations

from typing import Any, Literal

import torch
import triton
import triton.language as tl

from flagrand.runtime import CachedKernelLauncher

Distribution = Literal[
    "raw32",
    "uniform_f32",
    "uniform_f64",
    "normal_f32",
    "normal_f64",
    "lognormal_f32",
    "lognormal_f64",
]

RAW_BLOCK = 1024
UNIFORM_BLOCK = 4096
RAW_UNIFORM_BLOCK = RAW_BLOCK
NORMAL_BLOCK = 2048
LOGNORMAL_BLOCK = 2048


@triton.jit
def _hash32(seed_lo, seed_hi, word):
    """A counter-based 32-bit mixer using the 910C fast integer subset."""

    c_add = tl.full(word.shape, -1640531527, tl.int32).to(tl.uint32, bitcast=True)
    c_seed = tl.full(word.shape, -2048144789, tl.int32).to(tl.uint32, bitcast=True)
    c_mul0 = tl.full(word.shape, 2146121005, tl.int32).to(tl.uint32, bitcast=True)
    c_mul1 = tl.full(word.shape, -2073254261, tl.int32).to(tl.uint32, bitcast=True)

    seed0 = seed_lo.to(tl.uint32, bitcast=True)
    seed1 = seed_hi.to(tl.uint32, bitcast=True)
    x = (word.to(tl.uint32, bitcast=True) + seed0 + (seed1 * c_seed).to(tl.uint32) + c_add).to(tl.uint32)

    # FlagTree/BiSheng lowers uint32 shifts as arithmetic shifts on this stack.
    # The masks restore logical-right-shift semantics explicitly.
    x = x ^ ((x >> 16) & 0x0000FFFF)
    x = (x * c_mul0).to(tl.uint32)
    x = x ^ ((x >> 15) & 0x0001FFFF)
    x = (x * c_mul1).to(tl.uint32)
    x = x ^ ((x >> 16) & 0x0000FFFF)
    return x


@triton.jit
def _small_u32_to_f32(value):
    # Direct uint32 -> float32 vcast is unavailable in the current Ascend
    # lowering.  Every caller proves value <= 0x00ffffff first.
    return value.to(tl.int32, bitcast=True).to(tl.float32)


@triton.jit
def _uniform_f32(raw):
    top24 = _small_u32_to_f32((raw >> 8) & 0x00FFFFFF)
    return (top24 + 1.0) * 5.9604644775390625e-8


@triton.jit
def _normal_score_f32(raw):
    """One-hash normal score specialized for Ascend 910C.

    Three bytes form an Irwin-Hall-3 variate.  The fourth byte supplies
    sub-bin jitter.  A degree-11 odd polynomial corrects the resulting CDF.
    This avoids the second hash and all log/sqrt/sin/cos operations.
    """

    b0 = _small_u32_to_f32(raw & 0xFF)
    b1 = _small_u32_to_f32((raw >> 8) & 0xFF)
    b2 = _small_u32_to_f32((raw >> 16) & 0xFF)
    b3 = _small_u32_to_f32((raw >> 24) & 0xFF)

    jitter = (b3 + 0.5) * 0.00390625
    x = (b0 + b1 + b2 + jitter - 383.0) * 0.007812539737036188
    x2 = x * x

    polynomial = 0.9370396420905903 + x2 * (
        0.05161897562475311
        + x2
        * (
            -0.030045627507805688
            + x2 * (0.00979904761761375 + x2 * (-0.001354513844290328 + x2 * 0.00007386997969510554))
        )
    )
    return x * polynomial


@triton.jit
def _raw_uniform_kernel(
    out,
    n_elements,
    seed_lo,
    seed_hi,
    base_index,
    UNIFORM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    raw = _hash32(seed_lo, seed_hi, base_index + offsets)
    if UNIFORM:
        value = _uniform_f32(raw)
    else:
        value = raw.to(tl.int32, bitcast=True)
    tl.store(out + offsets, value, mask=mask)


@triton.jit
def _normal_kernel(
    out,
    n_elements,
    seed_lo,
    seed_hi,
    base_index,
    mean,
    stddev,
    LOGNORMAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    index = base_index + offsets

    raw = _hash32(seed_lo, seed_hi, index)
    value = mean + stddev * _normal_score_f32(raw)

    if LOGNORMAL:
        value = tl.exp(value)

    tl.store(out + offsets, value, mask=mask)


_RAW_UNIFORM_LAUNCHER = CachedKernelLauncher(
    _raw_uniform_kernel,
    constexpr_names=("UNIFORM", "BLOCK"),
)
_NORMAL_LAUNCHER = CachedKernelLauncher(
    _normal_kernel,
    constexpr_names=("LOGNORMAL", "BLOCK"),
)


def generate_huawei910c_normal_f32_fast(
    out: torch.Tensor,
    generator: Any,
    *,
    mean: float,
    stddev: float,
    lognormal: bool,
    num_warps: int,
) -> torch.Tensor:
    """Launch the validated common-case 910C normal kernel with minimal host work.

    The public cuRAND dispatcher validates the generator, tensor, and launch
    options before entering this function. Calls that cross a uint32 counter
    epoch retain the fully general windowed implementation below.
    """

    count = out.numel()
    if count == 0:
        return out

    seed = int(generator.seed)
    offset = int(generator.offset)
    counter_low = offset & 0xFFFFFFFF

    if count > (1 << 32) - counter_low:
        return generate_huawei910c(
            out,
            generator,
            distribution="lognormal_f32" if lognormal else "normal_f32",
            mean=mean,
            stddev=stddev,
            num_warps=num_warps,
        )

    seed_lo = _signed_i32(seed & 0xFFFFFFFF)
    seed_hi = (seed >> 32) & 0xFFFFFFFF
    epoch = offset >> 32
    epoch_key = (epoch * 0x9E3779B9) & 0xFFFFFFFF
    effective_seed_hi = _signed_i32(seed_hi ^ epoch_key)
    counter_low = _signed_i32(counter_low)
    block = LOGNORMAL_BLOCK if lognormal else NORMAL_BLOCK

    _NORMAL_LAUNCHER.launch(
        (triton.cdiv(count, block),),
        (
            out,
            count,
            seed_lo,
            effective_seed_hi,
            counter_low,
            float(mean),
            float(stddev),
        ),
        (lognormal, block),
        (num_warps,),
    )
    generator.offset = offset + count
    return out


def generate_huawei910c(
    out: torch.Tensor,
    generator: Any,
    *,
    distribution: Distribution,
    mean: float = 0.0,
    stddev: float = 1.0,
    num_warps: int = 4,
) -> torch.Tensor:
    """Generate one Huawei-910C-compatible public FlagRAND distribution.

    int32/float32 use independent Triton kernels.  Huawei's native float64
    Uniform/Normal contract has only float32 random resolution; those three
    APIs intentionally use the CANN-native generator so dtype, resolution and
    performance match the platform baseline exactly.  LogNormal applies CANN's
    float64 exp after the native Normal fill, matching the measured baseline.
    """

    _validate(out, distribution, num_warps)
    if out.numel() == 0:
        return out

    if distribution.endswith("f64"):
        generate_huawei910c_native_f64(
            out,
            generator,
            distribution=distribution,
            mean=mean,
            stddev=stddev,
        )
        return out

    seed = int(generator.seed)
    offset = int(generator.offset)
    seed_lo = seed & 0xFFFFFFFF
    seed_hi = (seed >> 32) & 0xFFFFFFFF
    n = out.numel()

    counter_low = offset & 0xFFFFFFFF
    epoch = offset >> 32
    epoch_key = (epoch * 0x9E3779B9) & 0xFFFFFFFF
    effective_seed_hi = (seed_hi ^ epoch_key) & 0xFFFFFFFF

    # Almost every call stays within one uint32 counter epoch. Launch that
    # common case directly. The windowed path is required only by the single
    # call that actually straddles an epoch boundary.
    if n <= (1 << 32) - counter_low:
        _launch_huawei910c_f32_chunk(
            out,
            n,
            seed_lo=seed_lo,
            seed_hi=effective_seed_hi,
            counter_low=counter_low,
            distribution=distribution,
            mean=mean,
            stddev=stddev,
            num_warps=num_warps,
        )
    else:
        _launch_huawei910c_f32_windowed(
            out,
            seed_lo=seed_lo,
            seed_hi=seed_hi,
            offset=offset,
            distribution=distribution,
            mean=mean,
            stddev=stddev,
            num_warps=num_warps,
        )

    generator.offset = offset + n
    return out


def _launch_huawei910c_f32_windowed(
    out: torch.Tensor,
    *,
    seed_lo: int,
    seed_hi: int,
    offset: int,
    distribution: Distribution,
    mean: float,
    stddev: float,
    num_warps: int,
) -> None:
    remaining = out.numel()
    written = 0
    logical = offset
    flat = out.reshape(-1)

    while remaining:
        counter_low = logical & 0xFFFFFFFF
        epoch = logical >> 32
        chunk = min(remaining, (1 << 32) - counter_low)

        # Epoch mixing happens on the host.  The NPU kernel sees only u32.
        epoch_key = (epoch * 0x9E3779B9) & 0xFFFFFFFF
        effective_seed_hi = (seed_hi ^ epoch_key) & 0xFFFFFFFF

        target = out if written == 0 and chunk == out.numel() else flat.narrow(0, written, chunk)

        _launch_huawei910c_f32_chunk(
            target,
            chunk,
            seed_lo=seed_lo,
            seed_hi=effective_seed_hi,
            counter_low=counter_low,
            distribution=distribution,
            mean=mean,
            stddev=stddev,
            num_warps=num_warps,
        )

        logical += chunk
        written += chunk
        remaining -= chunk


def _launch_huawei910c_f32_chunk(
    out: torch.Tensor,
    n: int,
    *,
    seed_lo: int,
    seed_hi: int,
    counter_low: int,
    distribution: Distribution,
    mean: float,
    stddev: float,
    num_warps: int,
) -> None:
    # Keep all uint32 bit patterns on Triton's fast signed-i32 scalar ABI.
    # Python values above INT32_MAX would otherwise create a slow u32
    # specialization after a long-running generator crosses an epoch.
    seed_lo = _signed_i32(seed_lo)
    seed_hi = _signed_i32(seed_hi)
    counter_low = _signed_i32(counter_low)

    if distribution in {"raw32", "uniform_f32"}:
        raw_uniform_block = UNIFORM_BLOCK if distribution == "uniform_f32" else RAW_BLOCK
        grid = (triton.cdiv(n, raw_uniform_block),)
        _RAW_UNIFORM_LAUNCHER.launch(
            grid,
            (out, n, seed_lo, seed_hi, counter_low),
            (distribution == "uniform_f32", raw_uniform_block),
            (num_warps,),
        )
        return

    if distribution in {"normal_f32", "lognormal_f32"}:
        normal_block = LOGNORMAL_BLOCK if distribution == "lognormal_f32" else NORMAL_BLOCK
        grid = (triton.cdiv(n, normal_block),)
        _NORMAL_LAUNCHER.launch(
            grid,
            (
                out,
                n,
                seed_lo,
                seed_hi,
                counter_low,
                float(mean),
                float(stddev),
            ),
            (distribution == "lognormal_f32", normal_block),
            (num_warps,),
        )
        return

    raise ValueError(f"unsupported 910C F32 distribution: {distribution}")


def _signed_i32(value: int) -> int:
    """Encode a uint32 bit pattern as a signed Python int32 argument."""

    word = int(value) & 0xFFFFFFFF
    return word if word < 0x80000000 else word - 0x100000000


def generate_huawei910c_native_f64(
    out: torch.Tensor,
    generator: Any,
    *,
    distribution: Distribution,
    mean: float,
    stddev: float,
) -> None:
    logical_offset = generator.offset
    state = _native_generator_state(out, generator, distribution)
    native = state["generator"]

    if distribution == "uniform_f64":
        out.uniform_(0.0, 1.0, generator=native)
    else:
        out.normal_(float(mean), float(stddev), generator=native)
        if distribution == "lognormal_f64":
            out.exp_()

    generator.offset = logical_offset + out.numel()
    state["logical_offset"] = generator.offset


def _native_generator_state(
    out: torch.Tensor,
    generator: Any,
    distribution: Distribution,
) -> dict[str, Any]:
    cache = getattr(generator, "_huawei910c_native_generators", None)
    if cache is None:
        cache = {}
        setattr(generator, "_huawei910c_native_generators", cache)

    key = (out.device.index, distribution)
    seed = generator.seed
    logical_offset = generator.offset
    state = cache.get(key)

    if state is not None and state["seed"] == seed and state["logical_offset"] == logical_offset:
        return state

    native = torch.Generator(device="npu")
    native.manual_seed(seed)

    state = {
        "generator": native,
        "seed": seed,
        "logical_offset": 0,
    }
    cache[key] = state

    if logical_offset:
        _advance_native_generator(
            out,
            native,
            distribution=distribution,
            count=logical_offset,
        )
        state["logical_offset"] = logical_offset

    return state


def _advance_native_generator(
    out: torch.Tensor,
    native: torch.Generator,
    *,
    distribution: Distribution,
    count: int,
) -> None:
    # Nonzero initial offsets are a control-plane operation, not a timed hot
    # path.  Consume in bounded chunks because torch.Generator.set_offset uses
    # backend-specific Philox units rather than FlagRAND's element units.
    remaining = int(count)
    while remaining:
        chunk = min(remaining, 1 << 20)
        scratch = torch.empty(chunk, device=out.device, dtype=torch.float64)
        if distribution == "uniform_f64":
            scratch.uniform_(0.0, 1.0, generator=native)
        else:
            scratch.normal_(0.0, 1.0, generator=native)
        remaining -= chunk


def _validate(out: torch.Tensor, distribution: Distribution, num_warps: int) -> None:
    expected = {
        "raw32": torch.int32,
        "uniform_f32": torch.float32,
        "uniform_f64": torch.float64,
        "normal_f32": torch.float32,
        "normal_f64": torch.float64,
        "lognormal_f32": torch.float32,
        "lognormal_f64": torch.float64,
    }
    if distribution not in expected:
        raise ValueError(f"unsupported 910C distribution: {distribution}")
    if out.device.type != "npu":
        raise ValueError("Huawei 910C fast path requires an NPU output tensor.")
    if out.dtype != expected[distribution]:
        raise TypeError(f"{distribution}: expected {expected[distribution]}, got {out.dtype}.")
    if not out.is_contiguous():
        raise ValueError(f"{distribution}: output must be contiguous.")
    if num_warps <= 0:
        raise ValueError(f"num_warps must be positive, got {num_warps}.")


__all__ = [
    "LOGNORMAL_BLOCK",
    "NORMAL_BLOCK",
    "RAW_BLOCK",
    "UNIFORM_BLOCK",
    "RAW_UNIFORM_BLOCK",
    "generate_huawei910c",
    "generate_huawei910c_normal_f32_fast",
    "generate_huawei910c_native_f64",
]
