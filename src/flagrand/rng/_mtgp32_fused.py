"""Complete-block MTGP32 generation and distribution fusion candidates.

The 192 states and 256-word step are fixed. Gather only pairs raw words
within one generated block; it does not change the global sequence order.
Unaligned calls retain the existing cached path.
"""

import struct

import torch
import triton
import triton.language as tl

from flagrand.fused.poisson import _sample_alias
from flagrand.runtime.backend import get_vendor_info
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_kernel import _mtgp32_recurrence, _mtgp32_temper, launch_mtgp32_contiguous, mtgp32_launch_warps
from ._mtgp32_sequence import ensure_working_state
from ._mtgp32_transforms import _uniform64
from ._mtgp32_uniform_f32 import _uniform32
from ._xorwow.metax_math import sqrt_radial64
from ._xorwow.native_math import sincos_turns
from ._xorwow.range_math64 import exp64, log_unit64


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _mtgp32_fused_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START_BLOCK,
    BLOCK_COUNT,
    N,
    MU_BITS,
    SIGMA_BITS,
    THRESHOLD,
    ALIASES,
    N_STATES: tl.constexpr,
    MODE: tl.constexpr,
    F64: tl.constexpr,
    FIRST: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
):
    p = tl.program_id(0)
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
    lane = tl.arange(0, BLOCK_SIZE)

    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(STATE + s_base + ((lane + offset) & STATE_MASK)).to(tl.uint32, bitcast=True)
        x2 = tl.load(STATE + s_base + ((lane + offset + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = tl.load(STATE + s_base + ((lane + offset + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        matrix = tl.load(PARAM + p_base + (y & 15)).to(tl.uint32, bitcast=True)
        value = y ^ matrix
        t = tl.load(STATE + s_base + ((lane + offset + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        word = _mtgp32_temper(value, t, TEMPER + p_base)
        tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
        block = k.to(tl.int64) * N_STATES + p

        if MODE == 1:
            if F64:
                j = tl.arange(0, BLOCK_SIZE // 2)
                a = tl.gather(word, 2 * j, 0)
                b = tl.gather(word, 2 * j + 1, 0)
                tl.store(OUT + block * (BLOCK_SIZE // 2) + j, _uniform64(a, b))
            else:
                tl.store(OUT + block * BLOCK_SIZE + lane, _uniform32(word))
        elif MODE == 4:
            j = tl.arange(0, BLOCK_SIZE // 2)
            a = tl.gather(word, 2 * j, 0)
            b = tl.gather(word, 2 * j + 1, 0)
            result = _sample_alias(a, b, THRESHOLD, ALIASES, FIRST, SIZE, 4294967296)
            tl.store(OUT + block * (BLOCK_SIZE // 2) + j, result)
        else:
            if F64:
                j = tl.arange(0, BLOCK_SIZE // 4)
                a = tl.gather(word, 4 * j, 0)
                b = tl.gather(word, 4 * j + 1, 0)
                c = tl.gather(word, 4 * j + 2, 0)
                d = tl.gather(word, 4 * j + 3, 0)
                u1 = _uniform64(a, b)
                u2 = _uniform64(c, d)
                radius = sqrt_radial64(-2.0 * log_unit64(u1))
                mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 4) + j
            else:
                j = tl.arange(0, BLOCK_SIZE // 2)
                a = tl.gather(word, 2 * j, 0)
                b = tl.gather(word, 2 * j + 1, 0)
                u1 = _uniform32(a)
                u2 = _uniform32(b)
                radius = tl.sqrt(-2.0 * tl.log(u1))
                mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 2) + j
            sine, cosine = sincos_turns(u2, F64)
            a = mu + sigma * radius * cosine
            b = mu + sigma * radius * sine
            if MODE == 3:
                if F64:
                    a = exp64(a)
                    b = exp64(b)
                else:
                    a = tl.exp(a)
                    b = tl.exp(b)
            tl.store(OUT + 2 * output_pair, a, 2 * output_pair < N)
            tl.store(OUT + 2 * output_pair + 1, b, 2 * output_pair + 1 < N)
        if k + 1 < iterations:
            tl.debug_barrier()


_fused_launcher = CachedKernelLauncher(
    _mtgp32_fused_kernel,
    constexpr_names=("N_STATES", "MODE", "F64", "FIRST", "SIZE", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
    option_names=("num_warps", "enable_fp_fusion"),
)


def generate_mtgp32_aligned(
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
    num_warps=8,
    cached_launch=True,
    keyed_launch=True,
    linear_table=False,
    cache_tensor_signature=True,
):
    """Generate a nonempty call consuming complete blocks; advance counters.

    Caller selects aligned engine.offset and raw-word consumption. This
    explicit entry remains available for comparisons with the two-pass path.
    """
    n = out.numel()
    f64 = out.dtype == torch.float64
    mode = {"raw32": 0, "uniform": 1, "normal": 2, "lognormal": 3, "poisson": 4}[distribution]
    if mode == 0:
        words = n
    elif mode == 1:
        words = n * (2 if f64 else 1)
    elif mode == 4:
        words = 2 * n
    else:
        words = ((n + 1) // 2) * (4 if f64 else 2)
    device = str(out.device)
    ensure_working_state(engine, engine.seed, device)
    pos, sh1, sh2, param, temper = engine._ws_params
    blocks = words // MTGP32_BLOCK_SIZE
    states = engine._working_state.shape[0]
    tensor_signatures = None
    if cached_launch and keyed_launch and cache_tensor_signature:
        out_signature = (out.dtype, out.data_ptr() % 16 == 0)
        tensor_signatures = (out_signature,) + engine._ws_tensor_signatures
    if mode == 0:
        launch_mtgp32_contiguous(
            out,
            engine._working_state,
            pos,
            sh1,
            sh2,
            param,
            temper,
            start_block=engine.offset // MTGP32_BLOCK_SIZE,
            block_count=blocks,
            num_warps=num_warps,
            cached_launch=cached_launch,
            keyed_launch=keyed_launch,
            linear_table=linear_table,
            tensor_signatures=tensor_signatures,
        )
    else:
        kernel, launcher = _mtgp32_fused_kernel, _fused_launcher
        corex_f32 = not f64 and get_vendor_info().vendor_name == "iluvatar"
        if corex_f32:
            from ._mtgp32_iluvatar_fused import _iluvatar_fused_launcher, _mtgp32_iluvatar_fused_kernel

            kernel, launcher = _mtgp32_iluvatar_fused_kernel, _iluvatar_fused_launcher
        fmt, integer_fmt = ("<d", "<Q") if f64 else ("<f", "<I")
        mu = struct.unpack(integer_fmt, struct.pack(fmt, mean))[0]
        sigma = struct.unpack(integer_fmt, struct.pack(fmt, stddev))[0]
        arguments = (
            out,
            engine._working_state,
            pos,
            sh1,
            sh2,
            param,
            temper,
            engine.offset // MTGP32_BLOCK_SIZE,
            blocks,
            n,
            mu,
            sigma,
            out if threshold is None else threshold,
            out if aliases is None else aliases,
        )
        constants = (states, mode, f64, first, size, MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N)
        if corex_f32:
            constants += (linear_table,)
        options = (mtgp32_launch_warps(blocks, (blocks + states - 1) // states, num_warps), mode not in (2, 3))
        if cached_launch:
            # Key tensors precede integers; the launch preserves the kernel ABI.
            key_args = arguments[:7] + arguments[12:] + arguments[7:12]
            if tensor_signatures is not None:
                tensor_signatures += (
                    out_signature if threshold is None else (threshold.dtype, threshold.data_ptr() % 16 == 0),
                    out_signature if aliases is None else (aliases.dtype, aliases.data_ptr() % 16 == 0),
                )
            key = (
                launch_specialization_key(
                    key_args, 9, constants, options, kernel.debug, tensor_signatures=tensor_signatures
                )
                if keyed_launch
                else None
            )
            launcher.launch((min(blocks, states),), arguments, constants, options, specialization_key=key)
        else:
            keywords = dict(zip(launcher._constexpr_names, constants))
            keywords.update(zip(launcher._option_names, options))
            kernel[(min(blocks, states),)](*arguments, **keywords)
    engine.offset += words
    engine._ws_next_block_start += words
    return out
