"""Bounded raw and derived MTGP32 output caches for Ascend.

Raw words own the continuation. Physical state advances only on refill;
public offset advances by the words consumed by each public call. Derived
values refer to that same raw window, including its pairing residue.
"""

import struct

import torch
import triton
import triton.language as tl

from flagrand.rng import _mtgp32_ascend as ascend
from flagrand.rng._mtgp32_sequence import clear_chunk_cache, ensure_working_state, generate_mtgp32_contiguous
from flagrand.rng._xorwow.packed64 import uniform64_words
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._mtgp32_ascend_copy import CachedCopy
from ._mtgp32_ascend_fused import aligned

PREFETCH_CALLS = 256
MAX_PREFETCH_WORDS = 4 * 1048576
MAX_DERIVED_OUTPUTS = 524288
MAX_BUFFERED_OUTPUTS = 524288
_uniform_kernel = ascend._uniform
_normal_kernel = ascend._normal32


class Buffer:
    def __init__(self):
        self.stream = None
        self.device = None
        self.values = None
        self.acl_copy = None


def prepare_stream(engine, device):
    buffer = getattr(engine, "_ascend_mtgp_buffer", None)
    if buffer is None:
        buffer = Buffer()
        engine._ascend_mtgp_buffer = buffer
    current = torch.npu.current_stream(device)
    previous = buffer.stream
    changed = previous is None or buffer.device != device or previous.npu_stream != current.npu_stream
    if previous is not None and changed:
        if buffer.device != device:
            previous.synchronize()
            buffer.values = None
        else:
            current.wait_stream(previous)
    ensure_working_state(engine, engine.seed, str(device))
    if changed:
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


def raw_window(engine, words, device):
    buffer = prepare_stream(engine, device)
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
            aligned(engine, fresh[remaining:], 0)
            engine.offset = logical_offset
        else:
            generate_mtgp32_contiguous(engine, fresh, engine.seed, engine.offset, 4)
        engine._chunk_cache = raw = fresh
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, str(device))
        buffer.values = None
        offset = 0
    engine.offset += words
    if offset + words == raw.numel():
        clear_chunk_cache(engine)
    return raw, offset, buffer


@triton.jit(do_not_specialize=["RAW_OFFSET", "N"])
def _transform(RAW, OUT, RAW_OFFSET, N, MODE: tl.constexpr, MU: tl.constexpr, SIGMA: tl.constexpr):
    if MODE == 4:
        lane = tl.arange(0, 512)
        index = tl.program_id(0) * 512 + lane
        words = tl.load(RAW + RAW_OFFSET + index, index < 2 * N, other=0)
        a, b = tl.split(tl.reshape(words, (256, 2)))
        lo, hi = uniform64_words(a, b)
        tl.store(OUT + index, tl.interleave(lo, hi), index < 2 * N)
    elif MODE == 1:
        _uniform_kernel(RAW + RAW_OFFSET, OUT, N, False, 256)
    else:
        _normal_kernel(RAW + RAW_OFFSET, OUT, N, MU, SIGMA, MODE == 3, 256)


_transform_launcher = CachedKernelLauncher(
    _transform,
    constexpr_names=("MODE", "MU", "SIGMA"),
    option_names=("num_warps", "enable_fp_fusion", "sync_solver"),
)


def launch_transform(raw, offset, out, mode, mean, stddev):
    n = out.numel()
    target = out.view(torch.int32) if mode == 4 else out
    work = (n + 1) // 2 if mode in (2, 3) else n
    constants = (mode, float(mean), float(stddev))
    _transform_launcher.launch(
        (triton.cdiv(work, 256),),
        (raw, target, offset, n),
        constants,
        (4, False, False),
        specialization_key=(
            target.device,
            target.dtype,
            target.data_ptr() % 16,
            raw.data_ptr() % 16,
            n,
            integer_specialization_key(offset),
            constants,
        ),
    )


def _buffered(engine, out, api="raw32", mean=0.0, stddev=1.0):
    n, f64 = out.numel(), out.dtype == torch.float64
    paired = api in ("normal", "lognormal")
    stride = 2 if paired or f64 else 1
    units = 2 if paired else 1
    words = stride * ((n + 1) // 2 if paired else n)
    raw, offset, buffer = raw_window(engine, words, out.device)
    if api == "raw32":
        target = out.view(torch.int32) if out.dtype == torch.uint32 else out
        target.view(-1).copy_(raw[offset : offset + n])
    else:
        mode = 4 if f64 else {"uniform": 1, "normal": 2, "lognormal": 3}[api]
        if n > MAX_DERIVED_OUTPUTS:
            launch_transform(raw, offset, out, mode, mean, stddev)
        else:
            parameters = (struct.pack("<d", float(mean)), struct.pack("<d", float(stddev))) if paired else None
            config = (api, out.dtype, parameters)
            entry = buffer.values
            if entry is None or entry[0] is not raw or entry[1] != config:
                entry = (raw, config, {})
            residue = offset % stride
            values = entry[2].get(residue)
            if values is None:
                count = units * ((raw.numel() - residue) // stride)
                values = torch.empty(count, dtype=out.dtype, device=out.device)
                launch_transform(raw, residue, values, mode, mean, stddev)
                entry[2][residue] = values
            first = units * ((offset - residue) // stride)
            out.view(-1).copy_(values[first : first + n])
            buffer.values = entry
    out.record_stream(buffer.stream)
    if engine._chunk_cache is None:
        buffer.values = None
    return out


def buffered(engine, out, api="raw32", mean=0.0, stddev=1.0):
    buffer = getattr(engine, "_ascend_mtgp_buffer", None)
    cached = None if buffer is None else getattr(buffer, "acl_copy", None)
    if cached is not None:
        words = cached.try_copy(engine, out, out.numel(), api, mean, stddev)
        if words:
            engine.offset += words
            if engine.offset == cached.end:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.acl_copy = None
            return out
        buffer.acl_copy = None
    result = _buffered(engine, out, api, mean, stddev)
    raw = engine._chunk_cache
    if raw is not None:
        buffer = engine._ascend_mtgp_buffer
        paired, f64, n = api in ("normal", "lognormal"), out.dtype == torch.float64, out.numel()
        stride = 2 if paired or f64 else 1
        units = 2 if paired else 1
        words = stride * ((n + 1) // 2 if paired else n)
        residue = (engine.offset - words - engine._chunk_cache_start) % stride
        source = raw if api == "raw32" else None
        if api != "raw32" and n <= MAX_DERIVED_OUTPUTS and buffer.values is not None:
            source = buffer.values[2].get(residue)
        if source is not None:
            parameters = struct.pack("<dd", float(mean), float(stddev)) if paired else None
            buffer.acl_copy = CachedCopy(
                engine, raw, source, api, out.dtype, stride, units, residue, parameters, buffer.stream
            )
    return result
