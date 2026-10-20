"""Bounded raw-word prefetch for partial MetaX MTGP32 calls.

The physical 192-state frontier can lead the consumed offset. The raw cache
is part of the generator state and is consumed before any aligned fast path.
Refills preserve the contiguous kernel's state mapping and recurrence.
"""

import struct
from functools import lru_cache
from weakref import ref

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

from flagrand.fused.poisson import _raw_poisson
from flagrand.metax.memory import device_memcpy_async
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _metax_tle_launcher, _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear
from ._mtgp32_sequence import clear_chunk_cache, ensure_working_state
from ._mtgp32_transforms import _uniform_kernel, build_normal_kernel, exp_finite_range

PREFETCH_CALLS = 32
MAX_PREFETCH_WORDS = 1048576
MAX_UNIFORM64_CACHE_OUTPUTS = 65536
_native_normal = build_normal_kernel(True)
_native_normal_finite = build_normal_kernel(True, True)


@triton.jit
def _write_refill_output(OUT, index, word, count, MODE: tl.constexpr):
    if MODE == 0:
        value = word.to(tl.int32, bitcast=True)
    else:
        value = ((word.to(tl.uint32, bitcast=True) >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
    tl.store(OUT + index, value, index < count)


@triton.jit(do_not_specialize=["OLD_OFFSET", "PREFIX", "START_BLOCK", "BLOCK_COUNT", "N"])
def _refill_kernel(
    OUT,
    OLD,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    PUBLIC_OUT,
    OLD_OFFSET,
    PREFIX,
    START_BLOCK,
    BLOCK_COUNT,
    N,
    OUTPUT_MODE: tl.constexpr,
    N_STATES: tl.constexpr,
    BLOCK: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    p = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    copy_index = p * BLOCK + lane
    old = tl.load(OLD + OLD_OFFSET + copy_index, copy_index < PREFIX, other=0)
    tl.store(OUT + copy_index, old, copy_index < PREFIX)
    if OUTPUT_MODE >= 0:
        _write_refill_output(PUBLIC_OUT, copy_index, old, PREFIX, OUTPUT_MODE)

    if p < tl.minimum(BLOCK_COUNT, N_STATES):
        first_state = (START_BLOCK % N_STATES).to(tl.int32)
        state_id = (first_state + p) % N_STATES
        first_round = ((START_BLOCK // N_STATES) & 3).to(tl.int32)
        first_round += (first_state + p) // N_STATES
        iterations = tl.cdiv(BLOCK_COUNT - p, N_STATES)
        pos = tl.load(POS + state_id)
        sh1 = tl.load(SH1 + state_id)
        sh2 = tl.load(SH2 + state_id)
        s_base = state_id * (STATE_MASK + 1)
        p_base = state_id * 16
        p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
        t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
        for k in range(iterations):
            offset = ((first_round + k) * BLOCK) & STATE_MASK
            x1 = tl.load(STATE + s_base + ((lane + offset) & STATE_MASK)).to(tl.uint32, bitcast=True)
            x2 = tl.load(STATE + s_base + ((lane + offset + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
            y = tl.load(STATE + s_base + ((lane + offset + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
            y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
            value = y ^ _linear_table_value(y, p0, p1, p2, p3)
            t = tl.load(STATE + s_base + ((lane + offset + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
            word = _temper_linear(value, t, t0, t1, t2, t3)
            tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
            index = PREFIX + (k.to(tl.int64) * N_STATES + p) * BLOCK + lane
            tl.store(OUT + index, word.to(tl.int32, bitcast=True))
            if OUTPUT_MODE >= 0:
                _write_refill_output(PUBLIC_OUT, index, word, N, OUTPUT_MODE)
            if k + 1 < iterations:
                tl.debug_barrier()


_refill_launcher = CachedKernelLauncher(
    _refill_kernel,
    constexpr_names=("OUTPUT_MODE", "N_STATES", "BLOCK", "STATE_MASK", "MASK", "N_RECUR"),
)


@lru_cache(maxsize=1)
def _metax_shared_refill():
    if _metax_tle_launcher("metax") is None:
        return None
    from ._mtgp32_tle_refill import _refill_launcher as shared_refill

    return shared_refill


@triton.jit(do_not_specialize=["RAW_OFFSET", "N", "MU_BITS", "SIGMA_BITS"])
def _transform_kernel(
    RAW,
    OUT,
    RAW_OFFSET,
    N,
    MU_BITS,
    SIGMA_BITS,
    MODE: tl.constexpr,
    F64: tl.constexpr,
    BLOCK: tl.constexpr,
    FINITE_EXP: tl.constexpr = False,
):
    source = RAW + RAW_OFFSET
    if MODE == 0:
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(source + i, i < N, other=0)
        tl.store(OUT + i, value, i < N)
    elif MODE == 1:
        _uniform_kernel(source, OUT, N, F64, BLOCK)
    else:
        if FINITE_EXP:
            _native_normal_finite(source, OUT, N, MU_BITS, SIGMA_BITS, F64, MODE == 3, BLOCK)
        else:
            _native_normal(source, OUT, N, MU_BITS, SIGMA_BITS, F64, MODE == 3, BLOCK)


_transform_launcher = CachedKernelLauncher(
    _transform_kernel,
    constexpr_names=("MODE", "F64", "BLOCK", "FINITE_EXP"),
    option_names=("num_warps", "enable_fp_fusion"),
)


@triton.jit(do_not_specialize=["RAW_OFFSET", "N"])
def _poisson_transform_kernel(
    RAW,
    OUT,
    THRESHOLD,
    ALIASES,
    RAW_OFFSET,
    N,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _raw_poisson(RAW + RAW_OFFSET, OUT, THRESHOLD, ALIASES, N, FIRST, SIZE, 4294967296, BLOCK)


_poisson_transform_launcher = CachedKernelLauncher(
    _poisson_transform_kernel, constexpr_names=("FIRST", "SIZE", "BLOCK")
)


def _dns_integer_abi(value):
    """MetaX's modern binder keeps only the ABI for do_not_specialize ints."""
    if -(1 << 31) <= value < (1 << 31):
        return "i32"
    if value < (1 << 63):
        return "i64"
    return "u64"


class _BufferedTransformKey:
    """Reuse the owned raw cache's metadata across distribution transforms."""

    def __init__(self):
        self.raw_ref = None
        self.raw_signature = None

    def __call__(self, args, tensor_count, constants, options, debug):
        raw = args[0]
        if self.raw_ref is None or self.raw_ref() is not raw:
            self.raw_ref = ref(raw)
            self.raw_signature = (raw.dtype, raw.data_ptr() % 16 == 0)
        out = args[1]
        out_signature = (out.dtype, out.data_ptr() % 16 == 0)
        if tensor_count == 2:
            tensors = (self.raw_signature, out_signature)
            scalar_types = (
                _dns_integer_abi(args[2]),
                _dns_integer_abi(args[3]),
                _dns_integer_abi(args[4]),
                _dns_integer_abi(args[5]),
            )
        else:
            tensors = (
                self.raw_signature,
                out_signature,
                (args[2].dtype, args[2].data_ptr() % 16 == 0),
                (args[3].dtype, args[3].data_ptr() % 16 == 0),
            )
            scalar_types = (_dns_integer_abi(args[4]), _dns_integer_abi(args[5]))
        return (
            "mtgp32_buffered",
            driver.active.get_current_device(),
            tensors,
            scalar_types,
            constants,
            options,
            debug,
        )


_buffered_specialization_key = (
    _BufferedTransformKey() if hasattr(_transform_kernel, "device_caches") else launch_specialization_key
)


def _record_stream_use(engine, device):
    stream_id = driver.active.get_current_stream(device.index)
    if getattr(engine, "_buffer_stream", stream_id) != stream_id:
        stream = torch.cuda.current_stream(device)
        engine._working_state.record_stream(stream)
        cache = getattr(engine, "_chunk_cache", None)
        if cache is not None:
            cache.record_stream(stream)
        for param in engine._ws_params:
            param.record_stream(stream)
    engine._buffer_stream = stream_id


def raw_window(engine, words, device, output=None, output_mode=-1):
    """Consume words; optionally write raw/FP32-uniform output during refill."""
    device_str = str(device)
    ensure_working_state(engine, engine.seed, device_str)
    _record_stream_use(engine, device)
    cache = engine._chunk_cache
    old_offset = 0 if cache is None else engine.offset - engine._chunk_cache_start
    remaining = 0 if cache is None else cache.numel() - old_offset
    output_written = False
    stats = getattr(engine, "_buffer_stats", None)
    if stats is None:
        stats = dict(refills=0, generated_words=0, consumed_words=0, peak_cache_words=0)
        engine._buffer_stats = stats
    if remaining < words:
        engine._uniform64_cache = None
        target = max(words, min(PREFETCH_CALLS * words, MAX_PREFETCH_WORDS))
        blocks = (target - remaining + 255) // 256
        count = remaining + blocks * 256
        new_cache = torch.empty(count, dtype=torch.int32, device=device)
        old = new_cache if cache is None else cache
        params = engine._ws_params
        arguments = (
            new_cache,
            old,
            engine._working_state,
            *params,
            new_cache if output is None else output,
            old_offset,
            remaining,
            engine._ws_next_block_start // 256,
            blocks,
            words,
        )
        constants = (output_mode, 192, 256, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
        key = launch_specialization_key(arguments, 9, constants, (4,), _refill_kernel.debug)
        grid = (max((remaining + 255) // 256, min(blocks, 192)),)
        launcher = _refill_launcher
        # Shared state benefits the measured ~2M-word cache-only refills;
        # shorter refills and fused raw/FP32 output keep the global path.
        if output_mode < 0 and blocks >= 8192:
            shared_refill = _metax_shared_refill()
            if shared_refill is not None:
                launcher = shared_refill
        launcher.launch(grid, arguments, constants, (4,), specialization_key=key)
        engine._ws_next_block_start += blocks * 256
        engine._chunk_cache = cache = new_cache
        engine._chunk_cache_start = engine.offset
        engine._chunk_cache_key = (engine.seed, device_str)
        old_offset = 0
        output_written = output_mode >= 0
        stats["refills"] += 1
        stats["generated_words"] += blocks * 256
        stats["peak_cache_words"] = max(stats["peak_cache_words"], count)
    engine.offset += words
    stats["consumed_words"] += words
    if old_offset + words == cache.numel():
        clear_chunk_cache(engine)
    return cache, old_offset, output_written


def _generate_uniform64_cached(engine, out, n):
    # Keep the last window alive locally when raw_window consumes its final
    # words and clears the engine cache. Only a live raw cache retains it.
    entry = getattr(engine, "_uniform64_cache", None)
    raw, offset, _ = raw_window(engine, 2 * n, out.device)
    if entry is None or entry[0] is not raw:
        entry = (raw, {}, engine._buffer_stream)
    elif entry[2] != engine._buffer_stream:
        stream = torch.cuda.current_stream(out.device)
        for values in entry[1].values():
            values.record_stream(stream)
        entry = (raw, entry[1], engine._buffer_stream)
    parity = offset & 1
    values = entry[1].get(parity)
    if values is None:
        count = (raw.numel() - parity) // 2
        values = torch.empty(count, dtype=torch.float64, device=out.device)
        arguments = (raw, values, parity, count, 0, 0)
        constants, options = (1, True, 256, False), (4, False)
        key = _buffered_specialization_key(arguments, 2, constants, options, _transform_kernel.debug)
        _transform_launcher.launch(((count + 255) // 256,), arguments, constants, options, specialization_key=key)
        entry[1][parity] = values
    status = device_memcpy_async()(
        out.data_ptr(), values.data_ptr() + (offset // 2) * 8, n * 8, 3, engine._buffer_stream
    )
    if status != 0:
        raise RuntimeError(f"mcMemcpyAsync returned {status}")
    if engine._chunk_cache is not None:
        engine._uniform64_cache = entry
    return out


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
):
    n = out.numel()
    f64 = out.dtype == torch.float64
    mode = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3, "poisson": 4}[distribution]
    if mode == 1 and f64 and n <= MAX_UNIFORM64_CACHE_OUTPUTS:
        cache = getattr(engine, "_chunk_cache", None)
        if cache is None or cache.numel() <= MAX_PREFETCH_WORDS + 255:
            return _generate_uniform64_cached(engine, out, n)
    if mode == 0:
        words = n
    elif mode == 1:
        words = n * (2 if f64 else 1)
    elif mode == 4:
        words = 2 * n
    else:
        words = ((n + 1) // 2) * (4 if f64 else 2)
    output_mode = mode if mode == 0 or (mode == 1 and not f64) else -1
    raw, raw_offset, output_written = raw_window(engine, words, out.device, out, output_mode)
    if output_written:
        return out
    if mode == 4:
        arguments = (raw, out, threshold, aliases, raw_offset, n)
        constants, options = (first, size, 256), (4,)
        key = _buffered_specialization_key(arguments, 4, constants, options, _poisson_transform_kernel.debug)
        _poisson_transform_launcher.launch(((n + 255) // 256,), arguments, constants, options, specialization_key=key)
        return out
    mu_bits = sigma_bits = 0
    if mode in (2, 3):
        mu_bits = struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", mean))[0]
        sigma_bits = struct.unpack("<Q" if f64 else "<I", struct.pack("<d" if f64 else "<f", stddev))[0]
    arguments = (raw, out, raw_offset, n, mu_bits, sigma_bits)
    finite_exp = mode == 3 and f64 and exp_finite_range(mean, stddev)
    constants, options = (mode, f64, 256, finite_exp), (4, False)
    key = _buffered_specialization_key(arguments, 2, constants, options, _transform_kernel.debug)
    count = (n + 1) // 2 if mode in (2, 3) else n
    _transform_launcher.launch(((count + 255) // 256,), arguments, constants, options, specialization_key=key)
    return out
