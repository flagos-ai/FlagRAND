from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec

import torch
import triton
import triton.language as tl

from flagrand.rng._mtgp32_data import MTGP32_BLOCK_SIZE, MTGP32_MASK, MTGP32_STATE_MASK, MTGPDC_N
from flagrand.runtime.backend import get_vendor_info
from flagrand.runtime.compiled_launcher import CachedKernelLauncher
from flagrand.runtime.specialization import launch_specialization_key

from ._mtgp32_linear_table import _linear_table_value, _load_linear_table_basis, _temper_linear


@triton.jit
def _mtgp32_recurrence(X1, X2, Y, sh1, sh2, MASK):
    X = (X1 & MASK) ^ X2
    X = X ^ ((X << sh1) & 0xFFFFFFFF)
    return X ^ (Y >> sh2)


@triton.jit
def _mtgp32_temper(r, T, temper_ptr):
    T = T ^ (T >> 16)
    T = T ^ (T >> 8)
    MAT = tl.load(temper_ptr + (T & 0x0F)).to(tl.uint32, bitcast=True)
    return r ^ MAT


@triton.jit
def _mtgp32_uint32_to_uniform(x):
    u = tl.uint_to_uniform_float(x.to(tl.uint32, bitcast=True))
    return tl.maximum(u, 2.3283064365386963e-10)


@triton.jit
def _mtgp32_kernel(
    out_ptr,
    state_ptr,
    pos_ptr,
    sh1_ptr,
    sh2_ptr,
    param_ptr,
    temper_ptr,
    n_elements,
    NUM_ITERS: tl.constexpr,
    START_ITER: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STATE_MASK: tl.constexpr,
    MASK: tl.constexpr,
    N_RECUR: tl.constexpr,
    LINEAR_TABLE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    pos = tl.load(pos_ptr + pid)
    sh1 = tl.load(sh1_ptr + pid)
    sh2 = tl.load(sh2_ptr + pid)

    s_base = pid * (STATE_MASK + 1)
    p_base = pid * 16
    offs = tl.arange(0, BLOCK_SIZE)
    if LINEAR_TABLE:
        p0, p1, p2, p3 = _load_linear_table_basis(param_ptr, p_base)
        t0, t1, t2, t3 = _load_linear_table_basis(temper_ptr, p_base)

    for k in range(NUM_ITERS):
        STATE_OFFSET = ((START_ITER + k) * BLOCK_SIZE) & STATE_MASK

        X1 = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET) & STATE_MASK)).to(tl.uint32, bitcast=True)
        X2 = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        Y = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)

        Y = _mtgp32_recurrence(X1, X2, Y, sh1, sh2, MASK)

        if LINEAR_TABLE:
            MAT = _linear_table_value(Y, p0, p1, p2, p3)
        else:
            MAT = tl.load(param_ptr + p_base + (Y & 0x0F)).to(tl.uint32, bitcast=True)
        r = Y ^ MAT

        new_state_idx = (offs + STATE_OFFSET + N_RECUR) & STATE_MASK
        tl.store(state_ptr + s_base + new_state_idx, r.to(tl.int32, bitcast=True))

        T = tl.load(state_ptr + s_base + ((offs + STATE_OFFSET + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        if LINEAR_TABLE:
            o = _temper_linear(r, T, t0, t1, t2, t3)
        else:
            o = _mtgp32_temper(r, T, temper_ptr + p_base)

        out_idx = (k * N_BLOCKS + pid) * BLOCK_SIZE + offs
        out_mask = out_idx < n_elements
        if OUTPUT_MODE == 1:
            tl.store(out_ptr + out_idx, _mtgp32_uint32_to_uniform(o), mask=out_mask)
        else:
            tl.store(out_ptr + out_idx, o.to(tl.int32, bitcast=True), mask=out_mask)

        if k + 1 < NUM_ITERS:
            tl.debug_barrier()


def launch_mtgp32_blocks(
    out: torch.Tensor,
    state: torch.Tensor,
    pos: torch.Tensor,
    sh1: torch.Tensor,
    sh2: torch.Tensor,
    param: torch.Tensor,
    temper: torch.Tensor,
    *,
    output_elements: int,
    chunks: int,
    start_iter: int,
    block_count: int,
    output_mode: int,
    requested_num_warps: int,
    linear_table: bool = False,
) -> None:
    grid = (block_count,)
    _mtgp32_kernel[grid](
        out,
        state,
        pos,
        sh1,
        sh2,
        param,
        temper,
        output_elements,
        int(chunks),
        start_iter,
        block_count,
        output_mode,
        BLOCK_SIZE=MTGP32_BLOCK_SIZE,
        STATE_MASK=MTGP32_STATE_MASK,
        MASK=MTGP32_MASK,
        N_RECUR=MTGPDC_N,
        LINEAR_TABLE=linear_table,
        num_warps=mtgp32_launch_warps(block_count, chunks, requested_num_warps),
    )


def mtgp32_launch_warps(block_count: int, chunks: int, requested: int) -> int:
    if chunks == 1 and block_count <= 32:
        # PPU's two-warp variant crashes the installed ppu-llc; its store
        # layout contains an ldmatrix conversion. Four and eight warps preserve the
        # same sequence/state and compile successfully in the small-call probe.
        return min(requested, 4 if get_vendor_info().vendor_name == "thead" else 2)
    return requested


@triton.jit(do_not_specialize=["START_BLOCK", "BLOCK_COUNT"])
def _mtgp32_contiguous_kernel(
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
    LINEAR_TABLE: tl.constexpr = False,
):
    # Programs enumerate the first output row, wrapping the parameter-set
    # number. Each program then owns all subsequent blocks for that state.
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
    if LINEAR_TABLE:
        p0, p1, p2, p3 = _load_linear_table_basis(PARAM, p_base)
        t0, t1, t2, t3 = _load_linear_table_basis(TEMPER, p_base)

    for k in range(iterations):
        offset = ((first_round + k) * BLOCK_SIZE) & STATE_MASK
        x1 = tl.load(STATE + s_base + ((lane + offset) & STATE_MASK)).to(tl.uint32, bitcast=True)
        x2 = tl.load(STATE + s_base + ((lane + offset + 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = tl.load(STATE + s_base + ((lane + offset + pos) & STATE_MASK)).to(tl.uint32, bitcast=True)
        y = _mtgp32_recurrence(x1, x2, y, sh1, sh2, MASK)
        if LINEAR_TABLE:
            matrix = _linear_table_value(y, p0, p1, p2, p3)
        else:
            matrix = tl.load(PARAM + p_base + (y & 15)).to(tl.uint32, bitcast=True)
        value = y ^ matrix
        t = tl.load(STATE + s_base + ((lane + offset + pos - 1) & STATE_MASK)).to(tl.uint32, bitcast=True)
        if LINEAR_TABLE:
            output = _temper_linear(value, t, t0, t1, t2, t3)
        else:
            output = _mtgp32_temper(value, t, TEMPER + p_base)
        tl.store(STATE + s_base + ((lane + offset + N_RECUR) & STATE_MASK), value.to(tl.int32, bitcast=True))
        index = (k.to(tl.int64) * N_STATES + p) * BLOCK_SIZE + lane
        tl.store(OUT + index, output.to(tl.int32, bitcast=True))
        if k + 1 < iterations:
            tl.debug_barrier()


_contiguous_launcher = CachedKernelLauncher(
    _mtgp32_contiguous_kernel,
    constexpr_names=("N_STATES", "BLOCK_SIZE", "STATE_MASK", "MASK", "N_RECUR", "LINEAR_TABLE"),
)


@lru_cache(maxsize=None)
def _metax_tle_launcher(vendor_name):
    if vendor_name != "metax":
        return None
    if find_spec("triton.experimental") is None or find_spec("triton.experimental.tle") is None:
        return None
    from ._mtgp32_tle_shared import launch_tle_shared

    return launch_tle_shared


def launch_mtgp32_contiguous(
    out,
    state,
    pos,
    sh1,
    sh2,
    param,
    temper,
    *,
    start_block,
    block_count,
    num_warps,
    cached_launch=True,
    keyed_launch=True,
    linear_table=False,
    tensor_signatures=None,
):
    # PPU's measured shared-state region starts at 16 Mi raw words. Its
    # original table loads outperform the four-basis form on this backend.
    if block_count >= 65536 and num_warps == 8 and not linear_table and get_vendor_info().vendor_name == "thead":
        from ._mtgp32_thead_shared import launch_thead_shared

        return launch_thead_shared(
            out,
            state,
            pos,
            sh1,
            sh2,
            param,
            temper,
            start_block=start_block,
            block_count=block_count,
            num_warps=num_warps,
            cached_launch=cached_launch,
            keyed_launch=keyed_launch,
            tensor_signatures=tensor_signatures,
        )
    # Shared-state setup pays off in the measured region of at least 1M raw
    # words. Preserve the existing path for explicit alternative configurations.
    if block_count >= 4096 and num_warps == 4 and linear_table:
        shared_launcher = _metax_tle_launcher(get_vendor_info().vendor_name)
        if shared_launcher is not None:
            return shared_launcher(
                out,
                state,
                pos,
                sh1,
                sh2,
                param,
                temper,
                start_block=start_block,
                block_count=block_count,
                num_warps=num_warps,
                cached_launch=cached_launch,
                keyed_launch=keyed_launch,
                tensor_signatures=tensor_signatures,
            )
    grid = (min(block_count, state.shape[0]),)
    arguments = (out, state, pos, sh1, sh2, param, temper, start_block, block_count)
    constants = (state.shape[0], MTGP32_BLOCK_SIZE, MTGP32_STATE_MASK, MTGP32_MASK, MTGPDC_N, linear_table)
    warps = mtgp32_launch_warps(block_count, (block_count + state.shape[0] - 1) // state.shape[0], num_warps)
    if cached_launch:
        key = (
            launch_specialization_key(
                arguments, 7, constants, (warps,), _mtgp32_contiguous_kernel.debug, tensor_signatures=tensor_signatures
            )
            if keyed_launch
            else None
        )
        _contiguous_launcher.launch(grid, arguments, constants, (warps,), specialization_key=key)
    else:
        _mtgp32_contiguous_kernel[grid](
            *arguments,
            N_STATES=constants[0],
            BLOCK_SIZE=constants[1],
            STATE_MASK=constants[2],
            MASK=constants[3],
            N_RECUR=constants[4],
            LINEAR_TABLE=linear_table,
            num_warps=warps,
        )
