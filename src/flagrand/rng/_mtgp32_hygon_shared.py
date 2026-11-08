"""Hygon candidate: retain the complete MTGP ring in TLE shared memory."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import integer_specialization_key

from ._hygon_launch import prepare_hygon
from ._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from ._mtgp32_hygon import _plan, _Stage
from ._mtgp32_kernel import _mtgp32_recurrence
from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT"])
def _shared_raw_kernel(
    OUT,
    STATE,
    POS,
    SH1,
    SH2,
    PARAM,
    TEMPER,
    START_BLOCK,
    BLOCK_COUNT,
    N_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    LINEAR_TABLE: tl.constexpr,
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
    p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
    t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
    lane = tl.arange(0, BLOCK_SIZE)
    ring_lane = tl.arange(0, STATE_MASK + 1)
    buffer = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
    ring_ptr = tle.gpu.local_ptr(buffer, indices=(ring_lane,))
    initial = tl.load(STATE + s_base + ring_lane).to(tl.uint32, bitcast=True)
    tl.store(ring_ptr, initial)
    tl.debug_barrier()
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane) & STATE_MASK,)))
        x2 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + 1) & STATE_MASK,)))
        y = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos) & STATE_MASK,)))
        t = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos - 1) & STATE_MASK,)))
        r = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        value = r ^ _linear_table_value(r, p0, p1, p2, p3)
        word = _temper_linear(value, t, t0, t1, t2, t3)
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane
        if OUT.dtype.element_ty == tl.float32:
            result = ((word >> 8) + 1).to(tl.float32) * 5.960464477539063e-8
        else:
            result = word.to(tl.int32, bitcast=True)
        tl.store(OUT + index, result)
        destination = tle.gpu.local_ptr(buffer, indices=((offset + lane + N_RECUR) & STATE_MASK,))
        tl.store(destination, value)
        # All reads for this step precede its state stores. This synchronizes
        # the next step and also the final complete backing-state writeback.
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + s_base + ring_lane, final.to(tl.int32, bitcast=True))


_shared_raw_launcher = CachedKernelLauncher(
    _shared_raw_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR", "LINEAR_TABLE"),
)

from ._mtgp32_fused import _sample_alias, _uniform32, _uniform64, exp64, log_unit64, sincos_turns, sqrt_radial64


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT", "N", "MU_BITS", "SIGMA_BITS"])
def _shared_fused_kernel(
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
    p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
    t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)
    lane = tl.arange(0, BLOCK_SIZE)
    ring_lane = tl.arange(0, STATE_MASK + 1)
    buffer = tle.gpu.alloc([STATE_MASK + 1], tl.uint32, nv_mma_shared_layout=False)
    ring_ptr = tle.gpu.local_ptr(buffer, indices=(ring_lane,))
    initial = tl.load(STATE + s_base + ring_lane).to(tl.uint32, bitcast=True)
    tl.store(ring_ptr, initial)
    tl.debug_barrier()
    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane) & STATE_MASK,)))
        x2 = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + 1) & STATE_MASK,)))
        y = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos) & STATE_MASK,)))
        t = tl.load(tle.gpu.local_ptr(buffer, indices=((offset + lane + pos - 1) & STATE_MASK,)))
        r = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        value = r ^ _linear_table_value(r, p0, p1, p2, p3)
        word = _temper_linear(value, t, t0, t1, t2, t3)
        destination = tle.gpu.local_ptr(buffer, indices=((offset + lane + N_RECUR) & STATE_MASK,))
        tl.store(destination, value)
        block = k.to(tl.int64) * N_STATES + p
        if MODE == 1:
            j = tl.arange(0, BLOCK_SIZE // 2)
            a = tl.gather(word, 2 * j, 0)
            b = tl.gather(word, 2 * j + 1, 0)
            tl.store(OUT + block * (BLOCK_SIZE // 2) + j, _uniform64(a, b))
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
                u1, u2 = _uniform64(a, b), _uniform64(c, d)
                radius = sqrt_radial64(-2.0 * log_unit64(u1))
                mu = MU_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint64).to(tl.float64, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 4) + j
            else:
                j = tl.arange(0, BLOCK_SIZE // 2)
                a = tl.gather(word, 2 * j, 0)
                b = tl.gather(word, 2 * j + 1, 0)
                u1, u2 = _uniform32(a), _uniform32(b)
                radius = tl.sqrt(-2.0 * tl.log(u1))
                mu = MU_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                sigma = SIGMA_BITS.to(tl.uint32).to(tl.float32, bitcast=True)
                output_pair = block * (BLOCK_SIZE // 2) + j
            sine, cosine = sincos_turns(u2, F64)
            a, b = mu + sigma * radius * cosine, mu + sigma * radius * sine
            if MODE == 3:
                if F64:
                    a, b = exp64(a), exp64(b)
                else:
                    a, b = tl.exp(a), tl.exp(b)
            tl.store(OUT + 2 * output_pair, a, 2 * output_pair < N)
            tl.store(OUT + 2 * output_pair + 1, b, 2 * output_pair + 1 < N)
        tl.debug_barrier()
    final = tl.load(ring_ptr)
    tl.store(STATE + s_base + ring_lane, final.to(tl.int32, bitcast=True))


_shared_fused_launcher = CachedKernelLauncher(
    _shared_fused_kernel,
    constexpr_names=("N_STATES", "MODE", "F64", "FIRST", "SIZE", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR"),
    option_names=("num_warps", "enable_fp_fusion"),
)


def generate_shared(
    engine, out, api="raw32", *, mean=0.0, stddev=1.0, lam=10.0, fused=True, num_warps=4, linear_table=True
):
    plan = _plan(engine, out, api, mean, stddev, lam, fused, num_warps, True)
    start = engine.offset // 256
    signature = (
        "shared",
        out.data_ptr() % 16 == 0,
        out.untyped_storage().size() <= (1 << 31) - 1,
        integer_specialization_key(start),
    )
    stages = plan.entries.get(signature)
    simple = plan.scratch is not None or plan.mode == 0 or (plan.mode == 1 and out.dtype == torch.float32)
    if stages is None:
        launcher = _shared_raw_launcher if simple else _shared_fused_launcher
        stages = (_Stage(prepare_hygon(launcher, plan.grids[0]), plan.constants[0], plan.options[0]),)
        if plan.scratch is not None:
            stages += (_Stage(prepare_hygon(plan.launchers[1], plan.grids[1]), plan.constants[1], plan.options[1]),)
        plan.entries[signature] = stages
    if plan.scratch is not None:
        args = (plan.scratch,) + plan.tensors + (start, plan.blocks)
    elif simple:
        args = (out,) + plan.tensors + (start, plan.blocks)
    else:
        threshold, aliases = (plan.threshold, plan.aliases) if plan.mode == 4 else (out, out)
        args = (out,) + plan.tensors + (start, plan.blocks, out.numel()) + plan.parameters + (threshold, aliases)
    stage = stages[0]
    stage.bound.launch(args, stage.constants, stage.options, specialization_key=stage.key)
    if plan.scratch is not None:
        if plan.mode == 1:
            args = (plan.scratch, out, out.numel())
        elif plan.mode == 4:
            args = (plan.scratch, out, plan.threshold, plan.aliases, out.numel())
        else:
            args = (plan.scratch, out, out.numel()) + plan.parameters
        stage = stages[1]
        stage.bound.launch(args, stage.constants, stage.options, specialization_key=stage.key)
    engine.offset += plan.words
    engine._ws_next_block_start += plan.words
    return out
