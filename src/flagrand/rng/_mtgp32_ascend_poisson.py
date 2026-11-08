"""Poisson values derived from the bounded Ascend MTGP32 raw window."""

import struct

import torch
import triton
import triton.language as tl

from flagrand.fused._internal import poisson_ascend as poisson
from flagrand.rng import _mtgp32_ascend_buffered as buffered
from flagrand.rng._mtgp32_sequence import clear_chunk_cache
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

MAX_DERIVED_OUTPUTS = 65536
_raw_poisson = poisson._raw_poisson


@triton.jit(do_not_specialize=["RAW_OFFSET", "N"])
def _transform(RAW, OUT, THRESHOLD, ALIASES, RAW_OFFSET, N, FIRST: tl.constexpr, SIZE: tl.constexpr):
    _raw_poisson(RAW + RAW_OFFSET, OUT, THRESHOLD, ALIASES, N, FIRST, SIZE, 256)


_launcher = CachedKernelLauncher(
    _transform,
    constexpr_names=("FIRST", "SIZE"),
    option_names=("num_warps", "sync_solver"),
)


def transform(raw, offset, out, threshold, aliases, first):
    n = out.numel()
    target = out.view(torch.int32) if out.dtype == torch.uint32 else out
    constants = (first, threshold.numel())
    _launcher.launch(
        (triton.cdiv(n, 256),),
        (raw, target, threshold, aliases, offset, n),
        constants,
        (4, False),
        specialization_key=(
            out.device,
            target.dtype,
            target.data_ptr() % 16,
            raw.data_ptr() % 16,
            threshold.data_ptr() % 16,
            aliases.data_ptr() % 16,
            n,
            integer_specialization_key(offset),
            constants,
        ),
    )


def generate(engine, out, lam):
    n = out.numel()
    if not n:
        return out
    # CachedCopy compares api by equality. Include the exact binary64 lambda
    # here so values from an earlier distribution cannot be reused by mistake.
    api = ("poisson", struct.pack("<d", float(lam)))
    buffer = getattr(engine, "_ascend_mtgp_buffer", None)
    cached = None if buffer is None else buffer.acl_copy
    if cached is not None:
        words = cached.try_copy(engine, out, n, api, 0.0, 1.0) if n <= MAX_DERIVED_OUTPUTS else 0
        if words:
            engine.offset += words
            if engine.offset == cached.end:
                clear_chunk_cache(engine)
                buffer.values = None
                buffer.acl_copy = None
            return out
        buffer.acl_copy = None
    raw, offset, buffer = buffered.raw_window(engine, 2 * n, out.device)
    threshold, aliases, first = poisson._tables(float(lam), out.device, 1 << 32)
    threshold.record_stream(buffer.stream)
    aliases.record_stream(buffer.stream)
    if n > MAX_DERIVED_OUTPUTS:
        transform(raw, offset, out, threshold, aliases, first)
    else:
        config = (api, out.dtype, None)
        entry = buffer.values
        if entry is None or entry[0] is not raw or entry[1] != config:
            entry = (raw, config, {})
        residue = offset % 2
        values = entry[2].get(residue)
        if values is None:
            count = (raw.numel() - residue) // 2
            values = torch.empty(count, dtype=torch.int32, device=out.device)
            transform(raw, residue, values, threshold, aliases, first)
            entry[2][residue] = values
        start = (offset - residue) // 2
        target = out.view(torch.int32) if out.dtype == torch.uint32 else out
        target.view(-1).copy_(values[start : start + n])
        buffer.values = entry
        if engine._chunk_cache is not None:
            buffer.acl_copy = buffered.CachedCopy(
                engine, raw, values, api, out.dtype, 2, 1, residue, None, buffer.stream
            )
    out.record_stream(buffer.stream)
    if engine._chunk_cache is None:
        buffer.values = None
        buffer.acl_copy = None
    return out
