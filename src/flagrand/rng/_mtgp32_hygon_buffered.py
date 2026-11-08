"""Bounded Hygon MTGP raw and distribution caches for partial calls.

The raw cache is the unread part of the same MTGP sequence. The physical
state can lead logical offset, as in the existing MetaX/PPU/CoreX paths.
"""

import struct

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

from flagrand.fused.poisson import _DEVICE_TABLES, _raw_poisson
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._hygon_launch import prepare_hygon
from ._mtgp32_hygon import _Stage, generate_aligned
from ._mtgp32_hygon_copy import CachedCopy
from ._mtgp32_sequence import clear_chunk_cache, ensure_working_state, generate_mtgp32_contiguous
from ._mtgp32_transforms import _normal_kernel, _uniform_kernel
from ._poisson_table import poisson_alias_table

PREFETCH_CALLS = 256
MAX_PREFETCH_WORDS = 8388608
MAX_DERIVED_OUTPUTS = 2097152


@triton.jit(do_not_specialize=["OFFSET", "N", "MU_BITS", "STD_BITS"])
def _transform(
    RAW,
    OUT,
    THRESHOLD,
    ALIASES,
    OFFSET,
    N,
    MU_BITS,
    STD_BITS,
    MODE: tl.constexpr,
    F64: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
):
    if MODE == 1:
        _uniform_kernel(RAW + OFFSET, OUT, N, F64, 256)
    elif MODE == 4:
        _raw_poisson(RAW + OFFSET, OUT, THRESHOLD, ALIASES, N, FIRST, SIZE, 4294967296, 256)
    else:
        _normal_kernel(RAW + OFFSET, OUT, N, MU_BITS, STD_BITS, F64, MODE == 3, 256)


_transform_launcher = CachedKernelLauncher(
    _transform,
    constexpr_names=("MODE", "F64", "FIRST", "SIZE"),
    option_names=("num_warps", "enable_fp_fusion"),
)


class _Buffer:
    def __init__(self):
        self.stream = None
        self.device = None
        self.values = None
        self.plans = {}
        self.refills = 0
        self.generated_words = 0
        self.peak_raw_words = 0
        self.derived_builds = 0
        self.cached_copy = None


def _prepare(engine, device):
    buffer = getattr(engine, "_hygon_mtgp_buffer", None)
    if buffer is None:
        buffer = _Buffer()
        engine._hygon_mtgp_buffer = buffer
    handle = driver.active.get_current_stream(device.index)
    previous = buffer.stream
    if previous is None or buffer.device != device or previous.cuda_stream != handle:
        current = torch.cuda.current_stream(device)
        if previous is not None:
            if buffer.device != device:
                previous.synchronize()
                buffer.values = None
                buffer.plans.clear()
            else:
                current.wait_stream(previous)
        ensure_working_state(engine, engine.seed, str(device))
        engine._working_state.record_stream(current)
        for tensor in engine._ws_params:
            tensor.record_stream(current)
        if engine._chunk_cache is not None:
            engine._chunk_cache.record_stream(current)
        if buffer.values is not None:
            for tensor in buffer.values[2].values():
                tensor.record_stream(current)
        buffer.stream, buffer.device = current, device
    else:
        ensure_working_state(engine, engine.seed, str(device))
    return buffer


def _raw_window(engine, words, device):
    buffer = _prepare(engine, device)
    raw = engine._chunk_cache
    offset = 0 if raw is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if raw is None else raw.numel() - offset
    if remaining < words:
        target = max(words, min(PREFETCH_CALLS * words, MAX_PREFETCH_WORDS))
        end = ((engine.offset + target + 255) // 256) * 256
        count = end - engine.offset
        fresh = torch.empty(count, dtype=torch.int32, device=device)
        before = engine._ws_next_block_start
        logical_offset = engine.offset
        start = logical_offset + remaining
        if start == before and start % 256 == 0:
            if remaining:
                fresh[:remaining].copy_(raw[offset:])
            engine.offset = start
            generate_aligned(engine, fresh[remaining:], num_warps=4, linear_table=True)
            engine.offset = logical_offset
        else:
            generate_mtgp32_contiguous(engine, fresh, engine.seed, engine.offset, 4, linear_table=True)
        engine._chunk_cache = raw = fresh
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(device))
        buffer.values = None
        buffer.refills += 1
        buffer.generated_words += engine._ws_next_block_start - before
        buffer.peak_raw_words = max(buffer.peak_raw_words, count)
        offset = 0
    engine.offset += words
    if offset + words == raw.numel():
        clear_chunk_cache(engine)
    return raw, offset, buffer


def _table(lam, device):
    key = (float(lam), 1 << 32, str(device))
    if key not in _DEVICE_TABLES:
        cutoff, alias, first = poisson_alias_table(float(lam), 1 << 32)
        signed = [x if x < 1 << 31 else x - (1 << 32) for x in cutoff]
        _DEVICE_TABLES[key] = (
            torch.tensor(signed, dtype=torch.int32, device=device),
            torch.tensor(alias, dtype=torch.int32, device=device),
            first,
        )
    return _DEVICE_TABLES[key]


def _launch(buffer, raw, offset, out, mode, parameters, tables):
    f64, n = out.dtype == torch.float64, out.numel()
    threshold, aliases, first = tables if tables is not None else (raw, raw, 0)
    size = threshold.numel() if tables is not None else 1
    constants, options = (mode, f64, first, size), (4, False)
    pairs = (n + 1) // 2 if mode in (2, 3) else n
    key = (
        n,
        out.dtype,
        parameters,
        constants,
        out.data_ptr() % 16 == 0,
        out.untyped_storage().size() <= (1 << 31) - 1,
        raw.untyped_storage().size() <= (1 << 31) - 1,
        integer_specialization_key(offset),
    )
    stage = buffer.plans.get(key)
    if stage is None:
        stage = _Stage(prepare_hygon(_transform_launcher, ((pairs + 255) // 256,)), constants, options)
        buffer.plans[key] = stage
    stage.bound.launch(
        (raw, out, threshold, aliases, offset, n) + parameters, constants, options, specialization_key=stage.key
    )


def generate_buffered(engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0):
    n, f64 = out.numel(), out.dtype == torch.float64
    if not n:
        return out
    buffer = getattr(engine, "_hygon_mtgp_buffer", None)
    cached = None if buffer is None else buffer.cached_copy
    if cached is not None:
        words = cached.try_copy(engine, out, n, api, mean, stddev, lam)
        if words:
            engine.offset += words
            if engine.offset == cached.end:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.cached_copy = None
            return out
        buffer.cached_copy = None
    pair = api in ("normal", "lognormal")
    stride = (4 if f64 else 2) if pair else 2 if (api == "poisson" or f64) else 1
    outputs_per_unit = 2 if pair else 1
    words = stride * ((n + 1) // 2 if pair else n)
    raw, offset, buffer = _raw_window(engine, words, out.device)
    source, residue = None, offset % stride
    if api == "raw32":
        target = out.view(torch.int32) if out.dtype == torch.uint32 else out
        target.view(-1).copy_(raw[offset : offset + n])
        source = raw
    else:
        mode = {"uniform": 1, "normal": 2, "lognormal": 3, "poisson": 4}[api]
        fmt, ifmt = ("<d", "<Q") if f64 else ("<f", "<I")
        parameters = (struct.unpack(ifmt, struct.pack(fmt, mean))[0], struct.unpack(ifmt, struct.pack(fmt, stddev))[0])
        tables = _table(lam, out.device) if api == "poisson" else None
        if tables is not None:
            tables[0].record_stream(buffer.stream)
            tables[1].record_stream(buffer.stream)
        if n > MAX_DERIVED_OUTPUTS:
            _launch(buffer, raw, offset, out, mode, parameters, tables)
        else:
            config = (api, out.dtype, parameters, float(lam) if mode == 4 else None)
            entry = buffer.values
            if entry is None or entry[0] is not raw or entry[1] != config:
                entry = (raw, config, {})
            residue = offset % stride
            values = entry[2].get(residue)
            if values is None:
                count = outputs_per_unit * ((raw.numel() - residue) // stride)
                values = torch.empty(count, dtype=out.dtype, device=out.device)
                _launch(buffer, raw, residue, values, mode, parameters, tables)
                entry[2][residue] = values
                buffer.derived_builds += 1
            first = outputs_per_unit * ((offset - residue) // stride)
            out.view(-1).copy_(values[first : first + n])
            buffer.values = entry
            source = values
    out.record_stream(buffer.stream)
    if engine._chunk_cache is None:
        buffer.values = None
        buffer.cached_copy = None
    elif source is not None and (source.dtype == out.dtype or (api == "raw32" and out.dtype == torch.uint32)):
        parameter_bytes = (struct.pack(fmt, mean), struct.pack(fmt, stddev)) if pair else None
        buffer.cached_copy = CachedCopy(
            engine, raw, source, api, stride, residue, outputs_per_unit, parameter_bytes, lam, buffer.stream, out.dtype
        )
    return out
