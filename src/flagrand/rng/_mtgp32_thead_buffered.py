"""Bounded MTGP32 raw and distribution caches for the PPU backend.

The existing 192-state recurrence advances the physical frontier; ``offset``
counts consumed raw32 words. The unconsumed raw cache remains part of the
generator's continuation across distribution and dtype changes.

The default prefetch is 32 calls, capped at 1048576 raw words. A request
larger than this cap still needs its own raw words, plus at most 255 words
for block alignment. Derived values are retained for N <= 65536, or N <= 524288 for
FP32 normal and lognormal. The raw prefetch bound remains unchanged.
"""

from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

from flagrand.fused.poisson import _raw_poisson
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.ppu_tensor_copy import load_ppu_tensor_copy
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_kernel import launch_mtgp32_contiguous
from ._mtgp32_sequence import clear_chunk_cache, ensure_working_state
from ._mtgp32_transforms import _uniform_kernel, build_normal_kernel

PREFETCH_CALLS = 32
MAX_PREFETCH_WORDS = 1048576
MAX_BUFFERED_OUTPUTS = 65536
MAX_DERIVED_OUTPUTS = 65536
_portable_normal = build_normal_kernel(False, False, True, ppu_native_log64=True, ppu_native_exp64=True)


class _BufferState:
    def __init__(self, prefetch_calls=PREFETCH_CALLS, max_prefetch_words=MAX_PREFETCH_WORDS):
        self.prefetch_calls = prefetch_calls
        self.max_prefetch_words = max_prefetch_words
        native_copy = load_ppu_tensor_copy()
        self.copy = native_copy.copy_tensor
        self.cached_output_type = native_copy.CachedOutput
        self.cached_output = None
        self.stream = None
        self.stream_identity = None
        self.values = None
        self.copy_source = None


def _buffer_state(engine):
    buffer = getattr(engine, "_thead_buffer", None)
    if buffer is None:
        buffer = _BufferState()
        engine._thead_buffer = buffer
    return buffer


def configure_buffer(engine, *, prefetch_calls=PREFETCH_CALLS, max_prefetch_words=MAX_PREFETCH_WORDS):
    """Set the finite prefetch policy without discarding unconsumed words."""
    buffer = _buffer_state(engine)
    buffer.prefetch_calls = prefetch_calls
    buffer.max_prefetch_words = max_prefetch_words


def reset_buffer(engine):
    """Drop derived references while retaining the last stream's ordering."""
    buffer = getattr(engine, "_thead_buffer", None)
    if buffer is not None:
        buffer.values = None
        buffer.copy_source = None
        buffer.cached_output = None


def _record_stream_use(engine, buffer, device):
    identity = (device.index, int(driver.active.get_current_stream(device.index)))
    if buffer.stream is not None and identity == buffer.stream_identity:
        return
    current = torch.cuda.current_stream(device)
    previous = buffer.stream
    if previous is not None and previous.cuda_stream != current.cuda_stream:
        current.wait_stream(previous)
        engine._working_state.record_stream(current)
        for tensor in engine._ws_params:
            tensor.record_stream(current)
        cache = engine._chunk_cache
        if cache is not None:
            cache.record_stream(current)
        if buffer.values is not None:
            for tensor in buffer.values[2].values():
                tensor.record_stream(current)
    buffer.stream = current
    buffer.stream_identity = (device.index, int(current.cuda_stream))


def prepare_stream(engine, device):
    """Order a PPU fallback/fused call after this engine's preceding stream."""
    ensure_working_state(engine, engine.seed, str(device))
    buffer = _buffer_state(engine)
    _record_stream_use(engine, buffer, device)
    return buffer


def _copy_tensor(buffer, out, source, source_elements, count):
    metadata = buffer.copy_source
    if metadata is None or metadata[0] is not source:
        metadata = (source, source.data_ptr(), source.element_size())
        buffer.copy_source = metadata
    buffer.copy(out, source, source_elements, count, buffer.stream)
    # Internal sources are created on this stream or recorded by
    # _record_stream_use before their owner/cache can be consumed or replaced.


def raw_window(engine, words, device, *, uniform64=False):
    buffer = prepare_stream(engine, device)
    cache = engine._chunk_cache
    old_offset = 0 if cache is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if cache is None else cache.numel() - old_offset
    if remaining < words:
        target = max(words, min(buffer.prefetch_calls * words, buffer.max_prefetch_words))
        blocks = (target - remaining + 255) // 256
        count = remaining + blocks * 256
        fresh = torch.empty(count, dtype=torch.int32, device=device)
        if remaining:
            _copy_tensor(buffer, fresh, cache, old_offset, remaining)
        if uniform64 and not remaining:
            from ._mtgp32_thead_refill_uniform64 import launch_refill

            values = torch.empty(count // 2, dtype=torch.float64, device=device)
            launch_refill(
                fresh,
                values,
                engine._working_state,
                *engine._ws_params,
                start_block=engine._ws_next_block_start // 256,
                block_count=blocks,
            )
            derived = (fresh, ("uniform", torch.float64), {0: values})
        else:
            launch_mtgp32_contiguous(
                fresh[remaining:],
                engine._working_state,
                *engine._ws_params,
                start_block=engine._ws_next_block_start // 256,
                block_count=blocks,
                num_warps=8,
                linear_table=False,
            )
            derived = None
        engine._ws_next_block_start += blocks * 256
        engine._chunk_cache = cache = fresh
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(device))
        buffer.values = derived
        old_offset = 0
    engine.offset += words
    if old_offset + words == cache.numel():
        clear_chunk_cache(engine)
    return cache, old_offset, buffer


@triton.jit(do_not_specialize=["RAW_OFFSET", "N", "MU_BITS", "SIGMA_BITS"])
def _transform_kernel(
    RAW, OUT, RAW_OFFSET, N, MU_BITS, SIGMA_BITS, MODE: tl.constexpr, F64: tl.constexpr, BLOCK: tl.constexpr
):
    if MODE == 1:
        _uniform_kernel(RAW + RAW_OFFSET, OUT, N, F64, BLOCK)
    else:
        _portable_normal(RAW + RAW_OFFSET, OUT, N, MU_BITS, SIGMA_BITS, F64, MODE == 3, BLOCK)


@triton.jit(do_not_specialize=["RAW_OFFSET", "N"])
def _poisson_kernel(
    RAW, OUT, THRESHOLD, ALIASES, RAW_OFFSET, N, FIRST: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    _raw_poisson(RAW + RAW_OFFSET, OUT, THRESHOLD, ALIASES, N, FIRST, SIZE, 4294967296, BLOCK)


_transform_launcher = CachedKernelLauncher(
    _transform_kernel,
    constexpr_names=("MODE", "F64", "BLOCK"),
    option_names=("num_warps", "enable_fp_fusion"),
)
_poisson_launcher = CachedKernelLauncher(_poisson_kernel, constexpr_names=("FIRST", "SIZE", "BLOCK"))


def _float_bits(value, f64):
    return struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", value))[0]


def _normal_key_bits(buffer, mean, stddev, f64):
    entry = getattr(buffer, "_normal_key_parameters", None)
    if entry is not None and entry[0] is f64 and entry[1] is mean and entry[2] is stddev:
        return entry[3]
    bits = (_float_bits(mean, f64), _float_bits(stddev, f64))
    if type(mean) is float and type(stddev) is float:
        buffer._normal_key_parameters = (f64, mean, stddev, bits)
    return bits


def _launch_transform(raw, raw_offset, out, distribution, mean, stddev, threshold, aliases, first, size):
    n = out.numel()
    if distribution == "poisson":
        args = (raw, out, threshold, aliases, raw_offset, n)
        constants, options = (first, size, 256), (4,)
        key = launch_specialization_key(args, 4, constants, options, _poisson_kernel.debug)
        _poisson_launcher.launch(((n + 255) // 256,), args, constants, options, specialization_key=key)
    else:
        f64 = out.dtype == torch.float64
        mode = {"uniform": 1, "normal": 2, "lognormal": 3}[distribution]
        args = (raw, out, raw_offset, n, _float_bits(mean, f64), _float_bits(stddev, f64))
        constants, options = (mode, f64, 256), (4, False)
        key = launch_specialization_key(args, 2, constants, options, _transform_kernel.debug)
        work = n if mode == 1 else (n + 1) // 2
        _transform_launcher.launch(((work + 255) // 256,), args, constants, options, specialization_key=key)


def _generate_buffered_uncached(
    engine,
    out,
    distribution="raw32",
    *,
    mean=0.0,
    stddev=1.0,
    threshold=None,
    aliases=None,
    first=0,
    size=1,
    n=None,
):
    n = out.numel() if n is None else n
    if distribution == "uniform" and out.dtype == torch.float32 and n > MAX_BUFFERED_OUTPUTS:
        from flagrand.rng._mtgp32_thead_uniform32_tail import generate_uniform_tail

        if generate_uniform_tail(engine, out, n):
            return out
    if distribution == "raw32" and n > MAX_BUFFERED_OUTPUTS:
        from flagrand.rng._mtgp32_thead_direct_tail import generate_direct_tail

        if generate_direct_tail(engine, out, n):
            return out
    f64 = out.dtype == torch.float64
    if distribution == "uniform" and f64 and n > MAX_BUFFERED_OUTPUTS:
        buffer = _buffer_state(engine)
        words = 2 * n
        # Keep the no-cache bound; an existing cache may use a larger raw tail.
        if (
            words + 256 >= min(buffer.prefetch_calls * words, buffer.max_prefetch_words)
            or getattr(engine, "_chunk_cache", None) is not None
        ) and engine.offset % 2 == 0:
            from flagrand.rng._mtgp32_thead_uniform64_tail import generate_tail

            if generate_tail(engine, out, n):
                return out
    pair_distribution = distribution in ("normal", "lognormal")
    stride = (4 if f64 else 2) if pair_distribution else (2 if f64 or distribution == "poisson" else 1)
    outputs_per_unit = 2 if pair_distribution else 1
    words = stride * ((n + 1) // 2 if pair_distribution else n)
    raw, offset, buffer = raw_window(
        engine, words, out.device, uniform64=distribution == "uniform" and f64 and n <= MAX_DERIVED_OUTPUTS
    )
    if distribution == "raw32":
        _copy_tensor(buffer, out, raw, offset, n)
    elif n > (524288 if pair_distribution and not f64 else MAX_DERIVED_OUTPUTS):
        _launch_transform(raw, offset, out, distribution, mean, stddev, threshold, aliases, first, size)
    else:
        parameters = (distribution, out.dtype)
        if pair_distribution:
            parameters += _normal_key_bits(buffer, mean, stddev, f64)
        if distribution == "poisson":
            parameters += (id(threshold), id(aliases), first, size)
        entry = buffer.values
        if entry is None or entry[0] is not raw or entry[1] != parameters:
            entry = (raw, parameters, {})
        residue = offset % stride
        values = entry[2].get(residue)
        if values is None:
            count = outputs_per_unit * ((raw.numel() - residue) // stride)
            values = torch.empty(count, dtype=out.dtype, device=out.device)
            _launch_transform(raw, residue, values, distribution, mean, stddev, threshold, aliases, first, size)
            entry[2][residue] = values
        output_start = outputs_per_unit * ((offset - residue) // stride)
        _copy_tensor(buffer, out, values, output_start, n)
        if engine._chunk_cache is not None:
            buffer.values = entry
    if engine._chunk_cache is None:
        buffer.values = None
    return out


def _bind_cached_output(engine, out, distribution, n):
    if n <= 0 or n > MAX_BUFFERED_OUTPUTS or distribution not in ("raw32", "uniform", "normal", "lognormal"):
        return
    buffer = engine._thead_buffer
    raw = engine._chunk_cache
    if raw is None:
        return
    pair = distribution in ("normal", "lognormal")
    f64 = out.dtype == torch.float64
    stride = (4 if f64 else 2) if pair else (2 if f64 else 1)
    outputs_per_unit = 2 if pair else 1
    words = stride * ((n + 1) // 2 if pair else n)
    raw_start = engine._chunk_cache_start
    raw_end = raw_start + raw.numel()
    residue = (engine.offset - words - raw_start) % stride
    entry = buffer.values
    if distribution == "raw32":
        source, mean_bits, stddev_bits = raw, 0, 0
    else:
        source = entry[2][residue]
        mean_bits, stddev_bits = entry[1][2:] if pair else (0, 0)
    context = buffer.cached_output_type(
        raw,
        source,
        entry,
        buffer.stream,
        engine.seed,
        raw_start,
        distribution,
        stride,
        residue,
        outputs_per_unit,
        mean_bits,
        stddev_bits,
    )
    buffer.cached_output = (context, raw_end)


def generate_buffered(
    engine,
    out,
    distribution="raw32",
    *,
    mean=0.0,
    stddev=1.0,
    threshold=None,
    aliases=None,
    first=0,
    size=1,
    n=None,
):
    buffer = getattr(engine, "_thead_buffer", None)
    cached = None if buffer is None else getattr(buffer, "cached_output", None)
    if cached is not None:
        consumed = cached[0].try_copy(
            out,
            engine._chunk_cache,
            buffer.values,
            buffer.stream,
            engine.seed,
            engine.offset,
            distribution,
            mean,
            stddev,
        )
        if consumed:
            engine.offset += consumed
            if engine.offset == cached[1]:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.cached_output = None
            return out
    if buffer is not None:
        buffer.cached_output = None
    result = _generate_buffered_uncached(
        engine,
        out,
        distribution,
        mean=mean,
        stddev=stddev,
        threshold=threshold,
        aliases=aliases,
        first=first,
        size=size,
        n=n,
    )
    _bind_cached_output(engine, out, distribution, out.numel() if n is None else n)
    return result
