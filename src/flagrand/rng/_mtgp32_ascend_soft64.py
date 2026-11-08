"""Standard MTGP FP64 distributions over the existing bounded raw window."""

import struct
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flagrand.rng import _mtgp32_ascend_buffered as buffered
from flagrand.rng._mtgp32_sequence import clear_chunk_cache
from flagrand.rng._xorwow._binding import bind_kernel
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._xorwow.ascend_bounded_math import bind_bounded_math

MAX_DERIVED_OUTPUTS = 65536
OPTIONS = (4, False, False)


@lru_cache(maxsize=1)
def normal_kernels():
    math = bind_bounded_math()
    from_words = math.from_words53
    log, sqrt, sincos = math.log, math.sqrt, math.sincos_turns
    mul_scaled, pack = math.mul_scaled, math.pack_binary64

    def radius(RAW, R, P, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        a = tl.load(RAW + 4 * i, i < P, other=0)
        b = tl.load(RAW + 4 * i + 1, i < P, other=0)
        u0, u1, u2 = from_words(a, b, True)
        l0, l1, l2 = log(u0, u1, u2)
        r0, r1, r2 = sqrt(-2.0 * l0, -2.0 * l1, -2.0 * l2)
        tl.store(R + i, r0, i < P)
        tl.store(R + P + i, r1, i < P)
        tl.store(R + 2 * P + i, r2, i < P)

    def angle(RAW, T, P, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        c = tl.load(RAW + 4 * i + 2, i < P, other=0)
        d = tl.load(RAW + 4 * i + 3, i < P, other=0)
        t0, t1, t2 = from_words(c, d, False)
        s0, s1, s2, c0, c1, c2 = sincos(t0, t1, t2, True)
        tl.store(T + i, s0, i < P)
        tl.store(T + P + i, s1, i < P)
        tl.store(T + 2 * P + i, s2, i < P)
        tl.store(T + 3 * P + i, c0, i < P)
        tl.store(T + 4 * P + i, c1, i < P)
        tl.store(T + 5 * P + i, c2, i < P)

    return bind_kernel(radius), bind_kernel(angle)


@lru_cache(maxsize=1)
def output_kernel():
    math = bind_bounded_math()
    mul_scaled, pack, exp = math.mul_scaled, math.pack_binary64, math.exp_from_scaled

    def output(R, T, OUT, N, P, LOG: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        r0 = tl.load(R + i, i < P, other=0)
        r1 = tl.load(R + P + i, i < P, other=0)
        r2 = tl.load(R + 2 * P + i, i < P, other=0)
        s0 = tl.load(T + i, i < P, other=0)
        s1 = tl.load(T + P + i, i < P, other=0)
        s2 = tl.load(T + 2 * P + i, i < P, other=0)
        c0 = tl.load(T + 3 * P + i, i < P, other=0)
        c1 = tl.load(T + 4 * P + i, i < P, other=0)
        c2 = tl.load(T + 5 * P + i, i < P, other=0)
        m0, m1, m2, e0 = mul_scaled(r0, r1, r2, c0, c1, c2)
        n0, n1, n2, e1 = mul_scaled(r0, r1, r2, s0, s1, s2)
        if LOG:
            # MTGP applies exp to the three-component normal, before packing.
            m0, m1, m2, e0 = exp(m0, m1, m2, e0)
            n0, n1, n2, e1 = exp(n0, n1, n2, e1)
        l0, h0 = pack(m0, m1, m2, e0)
        l1, h1 = pack(n0, n1, n2, e1)
        words = tl.interleave(tl.interleave(l0, l1), tl.interleave(h0, h1))
        j = tl.program_id(0) * (4 * B) + tl.arange(0, 4 * B)
        tl.store(OUT + j, words, j < 2 * N)

    return bind_kernel(output)


@lru_cache(maxsize=1)
def launchers():
    radius, angle = normal_kernels()
    names = ("num_warps", "enable_fp_fusion", "sync_solver")
    return (
        CachedKernelLauncher(radius, constexpr_names=("B",), option_names=names),
        CachedKernelLauncher(angle, constexpr_names=("B",), option_names=names),
        CachedKernelLauncher(output_kernel(), constexpr_names=("LOG", "B"), option_names=names),
    )


def transform(engine, raw, offset, out, lognormal):
    n, device = out.numel(), out.device
    pairs = (n + 1) // 2
    stream = torch.npu.current_stream(device)
    key = (pairs, device, stream.npu_stream)
    plan = getattr(engine, "_ascend_standard64_scratch", None)
    if plan is None or plan[0] != key:
        plan = (
            key,
            torch.empty((3, pairs), dtype=torch.float32, device=device),
            torch.empty((6, pairs), dtype=torch.float32, device=device),
        )
        engine._ascend_standard64_scratch = plan
    _, r, t = plan
    source = raw[offset : offset + 4 * pairs]
    radius, angle, output = launchers()
    k = (device, source.data_ptr() % 16, integer_specialization_key(pairs))
    radius.launch((triton.cdiv(pairs, 128),), (source, r, pairs), (128,), OPTIONS, specialization_key=k)
    angle.launch((triton.cdiv(pairs, 128),), (source, t, pairs), (128,), OPTIONS, specialization_key=k)
    block = 64 if lognormal else 256
    target = out.view(torch.int32)
    output.launch(
        (triton.cdiv(pairs, block),),
        (r, t, target, n, pairs),
        (lognormal, block),
        OPTIONS,
        specialization_key=(
            device,
            target.data_ptr() % 16,
            integer_specialization_key(n),
            integer_specialization_key(pairs),
            lognormal,
            block,
        ),
    )
    for tensor in (raw, r, t, out):
        tensor.record_stream(stream)


def standard(engine, out, distribution):
    n = out.numel()
    if not n:
        return out
    state = getattr(engine, "_ascend_mtgp_buffer", None)
    cached = None if state is None else state.acl_copy
    if cached is not None:
        words = cached.try_copy(engine, out, n, distribution, 0.0, 1.0) if n <= MAX_DERIVED_OUTPUTS else 0
        if words:
            engine.offset += words
            if engine.offset == cached.end:
                clear_chunk_cache(engine)
                state.values = None
                state.acl_copy = None
            return out
        state.acl_copy = None
    raw, offset, state = buffered.raw_window(engine, 4 * ((n + 1) // 2), out.device)
    if n > MAX_DERIVED_OUTPUTS:
        transform(engine, raw, offset, out, distribution == "lognormal")
    else:
        config = (distribution, out.dtype, (struct.pack("<d", 0.0), struct.pack("<d", 1.0)))
        entry = state.values
        if entry is None or entry[0] is not raw or entry[1] != config:
            entry = (raw, config, {})
        residue = offset % 4
        values = entry[2].get(residue)
        if values is None:
            count = 2 * ((raw.numel() - residue) // 4)
            values = torch.empty(count, dtype=out.dtype, device=out.device)
            transform(engine, raw, residue, values, distribution == "lognormal")
            entry[2][residue] = values
        start = 2 * ((offset - residue) // 4)
        out.view(-1).copy_(values[start : start + n])
        state.values = entry
        if engine._chunk_cache is not None:
            state.acl_copy = buffered.CachedCopy(
                engine, raw, values, distribution, out.dtype, 4, 2, residue, struct.pack("<dd", 0.0, 1.0), state.stream
            )
    out.record_stream(state.stream)
    if engine._chunk_cache is None:
        state.values = None
        state.acl_copy = None
    return out
