"""Bounded CoreX MTGP32 raw and FP32 distribution caches.

Existing contiguous generation owns the recurrence and partial-block cache.
This module retains its generated words and copies dynamic tensor slices.
"""

import struct

import torch

from flagrand.fused.poisson import _DEVICE_TABLES, _raw_poisson
from flagrand.rng import mtgp32
from flagrand.rng._mtgp32_sequence import clear_chunk_cache, ensure_working_state, generate_mtgp32_contiguous
from flagrand.rng._poisson_table import poisson_alias_table
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.iluvatar_tensor_copy import load_iluvatar_tensor_copy

PREFETCH_CALLS = 32
MAX_PREFETCH_WORDS = 1048576
MAX_DERIVED_OUTPUTS = 65536
MAX_POISSON_OUTPUTS = 1048577
MAX_POISSON_PREFETCH_WORDS = 8388608
_poisson_launcher = CachedKernelLauncher(_raw_poisson, constexpr_names=("FIRST", "SIZE", "RAW_RANGE", "BLOCK"))


class _BufferState:
    def __init__(self):
        native = load_iluvatar_tensor_copy()
        self.copy = native.copy_tensor
        self.cached_output_type = native.CachedOutput
        self.cached_output = None
        self.stream = None
        self.device = None
        self.values = None
        self.refills = 0
        self.generated_words = 0
        self.peak_raw_words = 0
        self.derived_builds = 0


def prepare_stream(engine, device):
    buffer = getattr(engine, "_iluvatar_buffer", None)
    if buffer is None:
        buffer = _BufferState()
        engine._iluvatar_buffer = buffer
    current = torch.cuda.current_stream(device)
    previous = buffer.stream
    if previous is not None and buffer.device != device:
        previous.synchronize()
        buffer.values = None
    elif previous is not None and previous.cuda_stream != current.cuda_stream:
        current.wait_stream(previous)
    ensure_working_state(engine, engine.seed, str(device))
    if previous is None or buffer.device != device or previous.cuda_stream != current.cuda_stream:
        engine._working_state.record_stream(current)
        for tensor in engine._ws_params:
            tensor.record_stream(current)
        if engine._chunk_cache is not None:
            engine._chunk_cache.record_stream(current)
        if buffer.values is not None:
            for tensor in buffer.values[2].values():
                tensor.record_stream(current)
    buffer.stream, buffer.device = current, device
    return buffer


def raw_window(engine, words, device, max_prefetch_words=MAX_PREFETCH_WORDS):
    buffer = prepare_stream(engine, device)
    cache = engine._chunk_cache
    offset = 0 if cache is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if cache is None else cache.numel() - offset
    if remaining < words:
        target = max(words, min(PREFETCH_CALLS * words, max_prefetch_words))
        # End at a global block boundary so the old contiguous path fully
        # consumes its temporary chunk cache before this raw window takes over.
        end = ((engine.offset + target + 255) // 256) * 256
        count = end - engine.offset
        fresh = torch.empty(count, dtype=torch.int32, device=device)
        before = engine._ws_next_block_start
        generate_mtgp32_contiguous(
            engine,
            fresh,
            engine.seed,
            engine.offset,
            4,
            keyed_launch=False,
            linear_table=count >= MAX_PREFETCH_WORDS,
        )
        engine._chunk_cache = cache = fresh
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(device))
        buffer.values = None
        buffer.refills += 1
        buffer.generated_words += engine._ws_next_block_start - before
        buffer.peak_raw_words = max(buffer.peak_raw_words, count)
        offset = 0
    engine.offset += words
    if offset + words == cache.numel():
        clear_chunk_cache(engine)
    return cache, offset, buffer


def table(lam, device):
    key = (float(lam), 1 << 32, str(device))
    if key not in _DEVICE_TABLES:
        cutoff, alias, first = poisson_alias_table(float(lam), 1 << 32)
        signed = [x if x < (1 << 31) else x - (1 << 32) for x in cutoff]
        _DEVICE_TABLES[key] = (
            torch.tensor(signed, dtype=torch.int32, device=device),
            torch.tensor(alias, dtype=torch.int32, device=device),
            first,
        )
    return _DEVICE_TABLES[key]


def launch_transform(raw, offset, out, api, mean, stddev, tables):
    source = raw[offset:]
    n = out.numel()
    if api == "uniform":
        mtgp32._uniform_launcher.launch(((n + 255) // 256,), (source, out, n), (False, 256), (4,))
    elif api == "poisson":
        threshold, aliases, first = tables
        _poisson_launcher.launch(
            ((n + 255) // 256,), (source, out, threshold, aliases, n), (first, threshold.numel(), 1 << 32, 256), (4,)
        )
    else:
        args = (source, out, n, mtgp32._float_bits(mean, False), mtgp32._float_bits(stddev, False))
        mtgp32._normal_launcher(False).launch(
            (((n + 1) // 2 + 255) // 256,), args, (False, api == "lognormal", 256), (4, False)
        )


def _generate_buffered_uncached(engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0):
    n = out.numel()
    if not n:
        return out
    pair = api in ("normal", "lognormal")
    stride = 2 if pair or api == "poisson" else 1
    outputs_per_unit = 2 if pair else 1
    words = stride * ((n + 1) // 2 if pair else n)
    if api == "poisson":
        raw, offset, buffer = raw_window(engine, words, out.device, MAX_POISSON_PREFETCH_WORDS)
    else:
        raw, offset, buffer = raw_window(engine, words, out.device)
    max_outputs = MAX_POISSON_OUTPUTS if api == "poisson" else MAX_DERIVED_OUTPUTS
    tables = table(lam, out.device) if api == "poisson" else None
    if tables is not None:
        tables[0].record_stream(buffer.stream)
        tables[1].record_stream(buffer.stream)
    if api == "raw32":
        buffer.copy(out, raw, offset, n)
    elif n > max_outputs:
        launch_transform(raw, offset, out, api, mean, stddev, tables)
    else:
        parameters = (api, out.dtype)
        if pair:
            parameters += (struct.pack("<f", mean), struct.pack("<f", stddev))
        elif api == "poisson":
            parameters += (float(lam), id(tables[0]), id(tables[1]), tables[2], tables[0].numel())
        entry = buffer.values
        if entry is None or entry[0] is not raw or entry[1] != parameters:
            entry = (raw, parameters, {})
        residue = offset % stride
        values = entry[2].get(residue)
        if values is None:
            count = outputs_per_unit * ((raw.numel() - residue) // stride)
            values = torch.empty(count, dtype=out.dtype, device=out.device)
            launch_transform(raw, residue, values, api, mean, stddev, tables)
            entry[2][residue] = values
            buffer.derived_builds += 1
        output_start = outputs_per_unit * ((offset - residue) // stride)
        buffer.copy(out, values, output_start, n)
        buffer.values = entry
    if api != "raw32" and n > max_outputs:
        out.record_stream(buffer.stream)
    if engine._chunk_cache is None:
        buffer.values = None
    return out


def reset_buffer(engine):
    """Public reset integration calls this after clearing raw state/cache."""
    buffer = getattr(engine, "_iluvatar_buffer", None)
    if buffer is not None:
        buffer.values = None
        buffer.cached_output = None


def _bind_cached_output(engine, out, api, mean, stddev, lam):
    raw = engine._chunk_cache
    n = out.numel()
    if raw is None or n <= 0 or n > (MAX_POISSON_OUTPUTS if api == "poisson" else MAX_DERIVED_OUTPUTS):
        return
    buffer = engine._iluvatar_buffer
    pair = api in ("normal", "lognormal")
    stride = 2 if pair or api == "poisson" else 1
    outputs_per_unit = 2 if pair else 1
    words = stride * ((n + 1) // 2 if pair else n)
    raw_start = engine._chunk_cache_start
    residue = (engine.offset - words - raw_start) % stride
    entry = buffer.values
    source = raw if api == "raw32" else entry[2][residue]
    mean_bits = int.from_bytes(entry[1][2], "little") if pair else 0
    stddev_bits = int.from_bytes(entry[1][3], "little") if pair else 0
    lambda_bits = struct.unpack("<Q", struct.pack("<d", float(lam)))[0] if api == "poisson" else 0
    context = buffer.cached_output_type(
        raw,
        source,
        entry,
        engine.seed,
        raw_start,
        api,
        stride,
        residue,
        outputs_per_unit,
        mean_bits,
        stddev_bits,
        lambda_bits,
    )
    buffer.cached_output = (context, raw_start + raw.numel())


def generate_buffered(engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0):
    buffer = getattr(engine, "_iluvatar_buffer", None)
    cached = None if buffer is None else getattr(buffer, "cached_output", None)
    if cached is not None:
        words = cached[0].try_copy(
            out, engine._chunk_cache, buffer.values, engine.seed, engine.offset, api, mean, stddev, lam
        )
        if words:
            engine.offset += words
            if engine.offset == cached[1]:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.cached_output = None
            return out
    if buffer is not None:
        buffer.cached_output = None
    result = _generate_buffered_uncached(engine, out, api, mean=mean, stddev=stddev, lam=lam)
    _bind_cached_output(engine, out, api, mean, stddev, lam)
    return result
